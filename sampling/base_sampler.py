"""Backend-agnostic batched candidate selection and continuation."""

import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple

from tqdm.auto import tqdm

from segmentation import (
    Segmenter,
    Unit,
    display_with_boundary_space,
    normalize_generated_whitespace,
    normalize_text,
)


# Tokens generated between boundary checks.
DEFAULT_CHUNK_TOKENS = 64

# Target concurrent candidates across active documents.
DEFAULT_TARGET_WAVE_CANDIDATES = 512

_SELECTION_MODES = ("rejection", "best-of-n")


class Region(Enum):
    """Semantic region assigned to one generated candidate."""

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass(frozen=True)
class CandidateScore:
    """A candidate's region and nonnegative depth inside that region."""

    region: Region
    depth: float

    def __post_init__(self) -> None:
        if not isinstance(self.region, Region):
            raise TypeError(f"candidate region must be a Region, got {self.region!r}")
        try:
            depth = float(self.depth)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"candidate depth must be numeric, got {self.depth!r}") from exc
        if not math.isfinite(depth) or depth < 0:
            raise ValueError(f"candidate depth must be finite and nonnegative, got {depth}")
        object.__setattr__(self, "depth", depth)

    def rank_key(self) -> tuple[int, float]:
        """Order green/deep, yellow/deep, then red/shallow."""
        if self.region is Region.GREEN:
            return 2, self.depth
        if self.region is Region.YELLOW:
            return 1, self.depth
        return 0, -self.depth


ScoreFn = Callable[[Unit, List[Unit], int], List[CandidateScore]]


def score_accepts(score: CandidateScore, margin: float = 0.0) -> bool:
    """Accept only green candidates whose region depth clears ``margin``."""
    return score.region is Region.GREEN and score.depth > margin


def default_accept_fn(
    scores: Sequence[CandidateScore], *, margin: float = 0.0,
) -> List[bool]:
    """Apply region-and-margin acceptance independently to each score."""
    return [score_accepts(s, margin) for s in scores]


def _acceptor(accept_fn, margin: float):
    """Bind the external margin unless the caller supplies custom acceptance."""
    return accept_fn or partial(default_accept_fn, margin=margin)


def _validate_scores(scores, expected: int) -> List[CandidateScore]:
    """Validate the sole public score representation and candidate alignment."""
    scores = list(scores)
    if len(scores) != expected:
        raise ValueError(
            f"score_fn returned {len(scores)} scores for {expected} candidates"
        )
    if not all(isinstance(score, CandidateScore) for score in scores):
        raise TypeError("score_fn must return CandidateScore values")
    return scores


@dataclass(frozen=True)
class GeneratedCandidate:
    """One raw candidate wave item plus its segmented unit and optional score."""

    source_index: int
    raw_text: str
    finished: bool
    boundary_complete: bool
    complete: bool
    unit: Unit
    score: Optional[CandidateScore] = None


@dataclass
class _DocState:
    """Mutable per-document state for the pooled continuation driver."""

    text: str
    prompt_len: int
    predecessor: Unit
    units: list = field(default_factory=list)
    steps: list = field(default_factory=list)


def _check_selection_mode(selection_mode: str) -> None:
    if selection_mode not in _SELECTION_MODES:
        raise ValueError(
            f"selection_mode must be one of {_SELECTION_MODES}, got {selection_mode!r}"
        )


def _per_doc(value, n_docs: int, name: str) -> list:
    """Broadcast a shared value or validate per-document values."""
    if isinstance(value, (list, tuple)):
        if len(value) != n_docs:
            raise ValueError(
                f"{name}: expected one entry per document ({n_docs}), got {len(value)}"
            )
        return list(value)
    return [value] * n_docs


def resolve_gen_params(gen_config) -> dict:
    """Resolve backend-neutral generation parameters."""
    do_sample = getattr(gen_config, "do_sample", True)
    if do_sample is None:
        do_sample = True
    return {
        "temperature": gen_config.temperature if gen_config.temperature is not None else 1.0,
        "top_k": gen_config.top_k if gen_config.top_k is not None else 50,
        "top_p": gen_config.top_p if gen_config.top_p is not None else 1.0,
        "repetition_penalty": (
            gen_config.repetition_penalty
            if gen_config.repetition_penalty is not None
            else 1.0
        ),
        "do_sample": do_sample,
    }


def resolve_adapter(model_path: str):
    """Return a base model and optional PEFT adapter path."""
    try:
        from peft import PeftConfig
        cfg = PeftConfig.from_pretrained(model_path)
        return cfg.base_model_name_or_path, model_path
    except Exception:
        return model_path, None


class BaseSampler(ABC):
    """Shared selection logic over a pluggable token engine."""

    num_candidates: int
    chunk_tokens: int
    tokenizer: Any
    segmenter: Segmenter
    stop_segmenter: Segmenter

    @abstractmethod
    def generate_raw(self, prompts, n, max_tokens, gen_config):
        """Generate ``n`` raw continuations per prompt."""
        raise NotImplementedError

    def _generate_batches_with_metadata(
        self, prompts: Sequence[str], gen_config,
    ) -> List[List[GeneratedCandidate]]:
        """Generate one chunked candidate wave per prompt."""
        if not prompts:
            return []
        max_new_tokens = gen_config.max_new_tokens if gen_config.max_new_tokens else 256
        chunk = self.chunk_tokens
        n = self.num_candidates

        accumulated = [[""] * n for _ in prompts]
        done = [[False] * n for _ in prompts]
        finished_flags = [[False] * n for _ in prompts]
        boundary_flags = [[False] * n for _ in prompts]

        def _update(p: int, i: int, finished: bool) -> None:
            boundary = len(self.stop_segmenter.segment(accumulated[p][i].strip())) >= 2
            finished_flags[p][i] = finished_flags[p][i] or bool(finished)
            boundary_flags[p][i] = boundary_flags[p][i] or boundary
            done[p][i] = finished_flags[p][i] or boundary_flags[p][i]

        # Draw the first chunk for every candidate.
        first_chunk = min(chunk, max_new_tokens)
        round1 = self.generate_raw(
            list(prompts), n=n, max_tokens=first_chunk, gen_config=gen_config,
        )
        for p, outs in enumerate(round1):
            for i, (text, finished) in enumerate(outs):
                accumulated[p][i] = text
                _update(p, i, finished)
        total_generated = first_chunk

        # Continue only incomplete candidates.
        while total_generated < max_new_tokens:
            active = [(p, i) for p in range(len(prompts)) for i in range(n) if not done[p][i]]
            if not active:
                break
            remaining_chunk = min(chunk, max_new_tokens - total_generated)
            cont_prompts = [prompts[p] + accumulated[p][i] for p, i in active]
            cont = self.generate_raw(
                cont_prompts, n=1, max_tokens=remaining_chunk, gen_config=gen_config,
            )
            for (p, i), outs in zip(active, cont):
                text, finished = outs[0]
                accumulated[p][i] += text
                _update(p, i, finished)
            total_generated += remaining_chunk

        # Segment the full candidate pool in one batch.
        flat_texts = [normalize_generated_whitespace(acc) for row in accumulated for acc in row]
        flat_units = self.segmenter.first_units(flat_texts)
        return [
            [
                GeneratedCandidate(
                    source_index=i,
                    raw_text=accumulated[p][i],
                    finished=finished_flags[p][i],
                    boundary_complete=boundary_flags[p][i],
                    complete=finished_flags[p][i] or boundary_flags[p][i],
                    unit=flat_units[p * n + i],
                )
                for i in range(n)
            ]
            for p in range(len(prompts))
        ]

    def _generate_batch_with_metadata(self, prompt: str, gen_config) -> List[GeneratedCandidate]:
        """Generate one candidate wave with metadata."""
        return self._generate_batches_with_metadata([prompt], gen_config)[0]

    def _generate_batch(self, prompt: str, gen_config) -> List[Unit]:
        """Generate ``self.num_candidates`` candidate units continuing ``prompt``."""
        return [item.unit for item in self._generate_batch_with_metadata(prompt, gen_config)]

    def prompt_predecessor(self, prompt: str) -> Unit:
        """Recover the final configured prompt unit exactly once.

        Empty or otherwise unsegmentable prompts receive a synthetic Unit so
        every score callback still has one explicit predecessor value.
        """
        units = self.segmenter.segment(prompt)
        if units:
            return units[-1]
        return Unit(self.segmenter.type, normalize_text(prompt), prompt)

    def scored_candidate_wave(
        self,
        prompt: str,
        gen_config,
        score_fn: ScoreFn,
        predecessor: Optional[Unit] = None,
        unit_idx: int = 0,
    ) -> List[GeneratedCandidate]:
        """Score a candidate wave against one explicit predecessor Unit."""
        if predecessor is None:
            predecessor = self.prompt_predecessor(prompt)
        items = self._generate_batch_with_metadata(prompt, gen_config)
        scoreable = [(i, item) for i, item in enumerate(items) if item.unit.normalized]
        if not scoreable:
            return items

        positions, valid_items = zip(*scoreable)
        scores = score_fn(
            predecessor, [item.unit for item in valid_items], unit_idx,
        )
        scores = _validate_scores(scores, len(valid_items))
        scored = list(items)
        for pos, score in zip(positions, scores):
            scored[pos] = replace(scored[pos], score=score)
        return scored

    @staticmethod
    def _dedup(units) -> List[Unit]:
        """Drop empty units and duplicates (by normalized text), keeping order."""
        seen: set = set()
        unique: List[Unit] = []
        for unit in units:
            if not unit.normalized or unit.normalized in seen:
                continue
            seen.add(unit.normalized)
            unique.append(unit)
        return unique

    def _unique_candidates(self, prompt: str, gen_config) -> List[Unit]:
        """One wave of non-empty Unit candidates, deduplicated by normalized text."""
        return self._dedup(self._generate_batch(prompt, gen_config))

    def _scored_candidates(
        self, prompt, gen_config, score_fn, predecessor, unit_idx,
    ) -> List[Tuple[Unit, CandidateScore]]:
        """Generate, deduplicate, and score one candidate wave."""
        if predecessor is None:
            predecessor = self.prompt_predecessor(prompt)
        candidates = self._unique_candidates(prompt, gen_config)
        if not candidates:
            return []
        scores = _validate_scores(
            score_fn(predecessor, candidates, unit_idx), len(candidates),
        )
        return list(zip(candidates, scores))

    @staticmethod
    def _select(scored, selection_mode: str, accept_fn) -> Tuple[Unit, dict]:
        """Select the first accepted candidate or the best region/depth pair."""
        flags = accept_fn([s for _, s in scored]) if scored else []
        unit, score, accepted = Unit(), None, False
        if selection_mode == "rejection":
            last_yellow = None
            for (u, s), flag in zip(scored, flags):
                if flag:
                    unit, score, accepted = u, s, True
                    break
                if s.region is Region.YELLOW:
                    last_yellow = u, s
            else:
                if last_yellow is not None:
                    unit, score = last_yellow
                elif scored:
                    unit, score = scored[-1]
        else:  # best-of-n
            if scored:
                best = max(
                    range(len(scored)), key=lambda i: scored[i][1].rank_key(),
                )
                unit, score = scored[best]
                accepted = bool(flags[best])
        return unit, {
            "score": score,
            "accepted": accepted,
            "n_accepted_candidates": sum(bool(f) for f in flags),
            "n_candidates": len(scored),
        }

    def generate(
        self,
        prompt: str,
        gen_config,
        score_fn: ScoreFn,
        selection_mode: str = "rejection",
        predecessor: Optional[Unit] = None,
        unit_idx: int = 0,
        margin: float = 0.0,
        accept_fn=None,
    ) -> Tuple[Unit, dict]:
        """Generate and select one unit at ``unit_idx`` after ``predecessor``."""
        _check_selection_mode(selection_mode)
        scored = self._scored_candidates(
            prompt, gen_config, score_fn, predecessor, unit_idx,
        )
        return self._select(scored, selection_mode, _acceptor(accept_fn, margin))

    def generate_with_rejects(
        self,
        prompt: str,
        gen_config,
        score_fn: ScoreFn,
        selection_mode: str = "rejection",
        num_accepted: int = 1,
        num_rejected: int = 1,
        predecessor: Optional[Unit] = None,
        unit_idx: int = 0,
        margin: float = 0.0,
        accept_fn=None,
    ) -> Tuple[List[Unit], List[Unit]]:
        """Partition one predecessor-aware wave into accepted and rejected units."""
        _check_selection_mode(selection_mode)
        scored = self._scored_candidates(
            prompt, gen_config, score_fn, predecessor, unit_idx,
        )
        if selection_mode == "rejection":
            flags = _acceptor(accept_fn, margin)([s for _, s in scored]) if scored else []
            accepted = [u for (u, _), flag in zip(scored, flags) if flag][:num_accepted]
            rejected = [u for (u, _), flag in zip(scored, flags) if not flag]
            rejected = rejected[:num_rejected][: len(accepted)]
            return accepted, rejected
        pool = sorted(scored, key=lambda x: x[1].rank_key(), reverse=True)
        n = len(pool)
        accepted = [u for u, _ in pool[: min(num_accepted, n)]]
        rejected_start = max(num_accepted, n - num_rejected)
        rejected = [u for u, _ in reversed(pool[rejected_start:])]
        return accepted, rejected

    def _resolve_max_active(self, max_active: Optional[int], n_docs: int) -> int:
        if max_active is None:
            max_active = max(1, DEFAULT_TARGET_WAVE_CANDIDATES // max(1, self.num_candidates))
        return max(1, min(max_active, n_docs))

    def _run_batched_continuation(
        self,
        prompts: Sequence[str],
        gen_config,
        score_fn: ScoreFn | Sequence[ScoreFn],
        selection_mode: str,
        accept_fn,
        max_tokens: Optional[int],
        max_active: Optional[int],
        progress: bool = False,
    ) -> List[_DocState]:
        """Run pooled unit-wise continuation over many documents."""
        if not prompts:
            return []
        max_new_tokens = max_tokens if max_tokens else (gen_config.max_new_tokens or 256)
        max_active = self._resolve_max_active(max_active, len(prompts))
        score_fns = _per_doc(score_fn, len(prompts), "score_fn")
        states = [
            _DocState(
                text=p,
                prompt_len=len(self.tokenizer.encode(p)),
                predecessor=self.prompt_predecessor(p),
            )
            for p in prompts
        ]
        pending = deque(range(len(prompts)))
        active: List[int] = []
        n_done = 0
        n_waves = 0
        total_units = 0
        # Track completed documents and live pool state.
        bar = tqdm(
            total=len(prompts),
            desc="pooled generation",
            unit="doc",
            disable=not progress,
            dynamic_ncols=True,
            mininterval=1.0,
        )
        try:
            while pending or active:
                while pending and len(active) < max_active:
                    active.append(pending.popleft())

                waves = self._generate_batches_with_metadata(
                    [states[d].text for d in active], gen_config,
                )
                still_active = []
                newly_done = 0
                for d, wave in zip(active, waves):
                    st = states[d]
                    candidates = self._dedup(item.unit for item in wave)
                    done = True
                    if candidates:
                        scores = _validate_scores(
                            score_fns[d](
                                st.predecessor, candidates, len(st.units),
                            ),
                            len(candidates),
                        )
                        unit, step_info = self._select(
                            list(zip(candidates, scores)), selection_mode, accept_fn,
                        )
                        st.text += display_with_boundary_space(st.text, unit.display)
                        st.predecessor = unit
                        st.units.append(unit)
                        st.steps.append(step_info)
                        total_units += 1
                        # The token budget is the only stopping rule: a document
                        # ends at the first unit boundary that crosses it.
                        done = (
                            len(self.tokenizer.encode(st.text)) - st.prompt_len
                            >= max_new_tokens
                        )
                    if done:
                        n_done += 1
                        newly_done += 1
                    else:
                        still_active.append(d)
                active = still_active
                n_waves += 1
                bar.set_postfix(
                    wave=n_waves,
                    active=len(active),
                    pending=len(pending),
                    units=total_units,
                    refresh=False,
                )
                if not bar.update(newly_done):
                    # Refresh waves with no completed documents.
                    bar.refresh()
        finally:
            bar.close()

        return states

    def generate_batched_continuation(
        self,
        prompts: Sequence[str],
        gen_config,
        score_fn: ScoreFn | Sequence[ScoreFn],
        selection_mode: str = "rejection",
        max_tokens: Optional[int] = None,
        max_active: Optional[int] = None,
        margin: float = 0.0,
        accept_fn=None,
        progress: bool = False,
    ) -> List[Tuple[str, dict]]:
        """Generate pooled unit-wise continuations for many prompts."""
        _check_selection_mode(selection_mode)
        states = self._run_batched_continuation(
            prompts, gen_config, score_fn, selection_mode,
            _acceptor(accept_fn, margin),
            max_tokens, max_active, progress,
        )
        results = []
        for st in states:
            accepted_flags = [s["accepted"] for s in st.steps]
            info = {
                "units": st.units,
                "steps": st.steps,
                "scores": [s["score"] for s in st.steps],
                "accepted": accepted_flags,
                "accepted_count": sum(accepted_flags),
                "unit_count": len(st.units),
                # Pool counts used for E[tries].
                "n_accepted_candidates_per_unit": [s["n_accepted_candidates"] for s in st.steps],
                "n_candidates_per_unit": [s["n_candidates"] for s in st.steps],
            }
            results.append((normalize_generated_whitespace(st.text).strip(), info))
        return results

    def generate_continuation(
        self,
        prompt: str,
        gen_config,
        score_fn: ScoreFn,
        selection_mode: str = "rejection",
        max_tokens: Optional[int] = None,
        margin: float = 0.0,
        accept_fn=None,
    ) -> Tuple[str, dict]:
        """Generate a unit-wise continuation for one prompt."""
        return self.generate_batched_continuation(
            [prompt], gen_config, score_fn, selection_mode,
            max_tokens=max_tokens, margin=margin, accept_fn=accept_fn,
        )[0]


def create_sampler(
    backend: str,
    model_path: str,
    *,
    num_candidates: int = 32,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    dtype=None,
    adapter_path: Optional[str] = None,
    adapter_strength: float = 1.0,
    segmentation_type: str = "sentence",
    segmentation_backend: str = "nltk",
    segmenter: Optional[Segmenter] = None,
    **backend_kwargs,
) -> "BaseSampler":
    """Construct an HF or vLLM sampler."""
    backend = backend.lower()
    common = dict(
        num_candidates=num_candidates,
        chunk_tokens=chunk_tokens,
        adapter_path=adapter_path,
        adapter_strength=adapter_strength,
        segmentation_type=segmentation_type,
        segmentation_backend=segmentation_backend,
        segmenter=segmenter,
    )
    if backend in ("hf", "huggingface"):
        import torch
        from sampling.hf_sampler import HFSampler
        return HFSampler(
            model_path,
            dtype=dtype if dtype is not None else torch.bfloat16,
            **common,
            **backend_kwargs,
        )
    if backend == "vllm":
        from sampling.vllm_sampler import VLLMSampler
        return VLLMSampler(
            model_path,
            dtype=dtype if dtype is not None else "bfloat16",
            **common,
            **backend_kwargs,
        )
    raise ValueError(f"Unknown sampler backend {backend!r}; expected 'hf' or 'vllm'.")


def run_contract_examples(sampler: "BaseSampler", gen_config, prompt: str) -> bool:
    """Exercise and assert the public sampler contract."""
    # Continuations stop on the token budget alone, so the examples keep a
    # short one rather than running out to the config's full generation length.
    demo_tokens = 96

    print("\n" + "=" * 60)
    print("Example 1: Accept any non-empty sentence (rejection)")
    print("=" * 60)
    print(f"Prompt: {prompt}")

    def accept_all(_predecessor, candidates, _unit_idx):
        return [CandidateScore(Region.GREEN, 1.0) for _ in candidates]

    unit, info = sampler.generate(prompt, gen_config, accept_all)
    print(f"Generated: {unit.display.strip()!r}")
    print(f"Accepted: {info['accepted']} ({info['n_accepted_candidates']}/{info['n_candidates']} in pool)")
    assert unit.normalized, "Example 1: expected a non-empty unit"
    assert info["accepted"], "Example 1: expected acceptance"

    print("\n" + "=" * 60)
    print("Example 2: Accept sentences with at least 10 words (rejection)")
    print("=" * 60)
    print(f"Prompt: {prompt}")

    def long_score(_predecessor, candidates, _unit_idx):
        return [
            CandidateScore(
                Region.GREEN if len(c.normalized.split()) >= 10 else Region.RED,
                1.0,
            )
            for c in candidates
        ]

    unit, info = sampler.generate(prompt, gen_config, long_score)
    print(f"Generated: {unit.display.strip()!r}")
    print(f"Accepted: {info['accepted']}")

    print("\n" + "=" * 60)
    print("Example 3: generate_continuation — position-aware score_fn")
    print("=" * 60)

    seen3: list = []

    def predecessor_aware_score(predecessor, candidates, unit_idx):
        seen3.append(unit_idx)
        predecessor_words = set(predecessor.normalized.split())
        threshold = max(0.1, 0.25 - 0.05 * unit_idx)
        results = []
        for c in candidates:
            words = c.normalized.split()
            if not words:
                results.append(CandidateScore(Region.RED, 1.0))
                continue
            overlap = sum(1 for w in words if w in predecessor_words) / len(words)
            results.append(CandidateScore(
                Region.GREEN if overlap < threshold else Region.RED, 1.0,
            ))
        return results

    full_text, info = sampler.generate_continuation(
        prompt, gen_config, predecessor_aware_score, max_tokens=demo_tokens,
    )
    total_count = info["unit_count"]
    print(f"Full text:\n{full_text}")
    print(f"Accepted: {info['accepted_count']}/{total_count} sentences")
    print(f"Indices seen: {sorted(set(seen3))}")
    assert sorted(set(seen3)) == list(range(total_count)), (
        f"position contract broken: expected 0..{total_count - 1}, got {sorted(set(seen3))}"
    )
    print("PASS: position-aware contract verified")

    print("\n" + "=" * 60)
    print("Example 4: generate_with_rejects — acceptance partition (rejection)")
    print("=" * 60)

    accepted_list, rejected_list = sampler.generate_with_rejects(
        prompt, gen_config, long_score, num_accepted=2, num_rejected=2,
    )
    print(f"Accepted ({len(accepted_list)}):")
    for u in accepted_list:
        print(f"  - {u.display.strip()!r}")
    print(f"Rejected ({len(rejected_list)}):")
    for u in rejected_list:
        print(f"  - {u.display.strip()!r}")
    assert len(rejected_list) <= len(accepted_list), "rejected list must not exceed accepted list"

    print("\n" + "=" * 60)
    print("Example 5: Accept sentences containing a comma (rejection)")
    print("=" * 60)

    def comma_score(_predecessor, candidates, _unit_idx):
        return [
            CandidateScore(
                Region.GREEN
                if "," in c.normalized and len(c.normalized.split()) >= 5
                else Region.RED,
                1.0,
            )
            for c in candidates
        ]

    unit, info = sampler.generate(prompt, gen_config, comma_score)
    print(f"Generated: {unit.display.strip()!r}")
    print(f"Accepted: {info['accepted']}")

    print("\n" + "=" * 60)
    print("Example 6: selection_mode='best-of-n' — highest unique-word count")
    print("=" * 60)

    def unique_word_score(_predecessor, candidates, _unit_idx):
        return [
            CandidateScore(Region.GREEN, float(len(set(c.normalized.split()))))
            for c in candidates
        ]

    best, info = sampler.generate(prompt, gen_config, unique_word_score, selection_mode="best-of-n")
    print(f"Best: {best.display.strip()!r}")
    print(f"Score: {info['score']}")
    assert best.normalized, "Example 6: expected a non-empty best unit"

    print("\n" + "=" * 60)
    print("Example 7: generate_with_rejects — rank partition (best-of-n), top 3 / bottom 3")
    print("=" * 60)

    accepted_top, rejected_bot = sampler.generate_with_rejects(
        prompt, gen_config, unique_word_score, selection_mode="best-of-n",
        num_accepted=3, num_rejected=3,
    )
    print(f"Accepted top-{len(accepted_top)} (best first):")
    for u in accepted_top:
        print(f"  [{len(set(u.normalized.split())):>3}] {u.display.strip()!r}")
    print(f"Rejected bottom-{len(rejected_bot)} (worst first):")
    for u in rejected_bot:
        print(f"  [{len(set(u.normalized.split())):>3}] {u.display.strip()!r}")
    assert set(u.normalized for u in accepted_top).isdisjoint(
        u.normalized for u in rejected_bot
    ), "accepted and rejected lists must be disjoint"

    print("\n" + "=" * 60)
    print("Example 8: best-of-n continuation — position-aware score_fn")
    print("=" * 60)

    seen8: list = []

    def positional_score(_predecessor, candidates, unit_idx):
        seen8.append(unit_idx)
        bonus = 0.5 * unit_idx
        return [
            CandidateScore(
                Region.GREEN, float(len(set(c.normalized.split()))) + bonus,
            )
            for c in candidates
        ]

    full_text, info = sampler.generate_continuation(
        prompt, gen_config, positional_score, selection_mode="best-of-n",
        max_tokens=demo_tokens,
    )
    sent_scores, sent_count = info["scores"], info["unit_count"]
    print(f"Full text:\n{full_text}")
    print(f"Per-sentence scores: {sent_scores}")
    print(f"Sentence count: {sent_count}")
    print(f"Indices seen: {sorted(set(seen8))}")
    assert sorted(set(seen8)) == list(range(sent_count)), (
        f"position contract broken: expected 0..{sent_count - 1}, got {sorted(set(seen8))}"
    )
    assert len(sent_scores) == sent_count
    print("PASS: position-aware contract verified")

    print("\n" + "=" * 60)
    print("Example 9: generate_batched_continuation — pooled documents")
    print("=" * 60)

    prompts9 = [
        prompt,
        "The deep sea remains one of the least explored places on Earth.",
    ]
    batched = sampler.generate_batched_continuation(
        prompts9, gen_config, predecessor_aware_score, max_tokens=demo_tokens,
    )
    assert len(batched) == len(prompts9), "one result per input prompt"
    for i, (text, info) in enumerate(batched):
        print(f"Doc {i}: accepted {info['accepted_count']}/{info['unit_count']}")
        print(f"  {text!r}")
        assert info["unit_count"] >= 1, f"Doc {i}: expected at least one generated unit"
        assert len(text) > len(prompts9[i]), f"Doc {i}: expected text beyond the prompt"
    print("PASS: pooled rejection continuation verified")

    print("\n" + "=" * 60)
    print("Example 10: pooled best-of-n continuation")
    print("=" * 60)

    batched = sampler.generate_batched_continuation(
        prompts9, gen_config, positional_score, selection_mode="best-of-n",
        max_tokens=demo_tokens,
    )
    assert len(batched) == len(prompts9), "one result per input prompt"
    for i, (text, info) in enumerate(batched):
        print(f"Doc {i}: {info['unit_count']} units, scores {info['scores']}")
        assert len(info["scores"]) == info["unit_count"]
    print("PASS: pooled best-of-n continuation verified")

    print("\n" + "=" * 60)
    print("Example 11: pool-dependent accept_fn — accept only above-median scores")
    print("=" * 60)

    def above_median_accept(scores):
        if not scores:
            return []
        ordered = sorted(scores, key=lambda score: score.depth)
        median = ordered[len(ordered) // 2]
        return [score.depth > median.depth for score in scores]

    unit, info = sampler.generate(
        prompt, gen_config, unique_word_score, accept_fn=above_median_accept,
    )
    print(f"Generated: {unit.display.strip()!r}")
    print(f"Accepted: {info['accepted']} "
          f"({info['n_accepted_candidates']}/{info['n_candidates']} above pool median)")
    assert info["n_accepted_candidates"] <= info["n_candidates"] // 2 + 1, (
        "above-median acceptance must flag at most half the pool (+1 for ties)"
    )
    print("PASS: pool-dependent acceptance verified")

    print("\n" + "=" * 60)
    print("Example 12: pooling extent — engine calls batch across documents")
    print("=" * 60)

    n_docs, max_active = 6, 3
    prompts12 = [f"Report {i}: {prompt}" for i in range(n_docs)]

    # Bind each scorer to one document.
    def make_doc_score(idx):
        tag = f"Report {idx}:"
        def doc_score(predecessor, candidates, unit_idx):
            if unit_idx == 0:
                assert predecessor.display.startswith(tag), (
                    f"score_fn for doc {idx} got another document's predecessor: "
                    f"{predecessor.display[:40]!r}"
                )
            return [CandidateScore(Region.GREEN, 1.0) for _ in candidates]
        return doc_score

    # Record engine batch sizes.
    call_log: list = []
    orig_generate_raw = sampler.generate_raw

    def counting_generate_raw(prompts, n, max_tokens, gen_config):
        call_log.append((len(prompts), n))
        return orig_generate_raw(prompts, n, max_tokens, gen_config)

    sampler.generate_raw = counting_generate_raw
    try:
        batched = sampler.generate_batched_continuation(
            prompts12, gen_config,
            [make_doc_score(i) for i in range(n_docs)],
            max_tokens=demo_tokens,
            max_active=max_active,
        )
    finally:
        sampler.generate_raw = orig_generate_raw

    assert len(batched) == n_docs, "one result per input prompt, in order"
    for i, (text, info) in enumerate(batched):
        print(f"Doc {i}: {info['unit_count']} unit(s)")
        assert info["unit_count"] >= 1, f"Doc {i}: expected at least one unit"
        assert text.startswith(f"Report {i}:"), f"Doc {i}: results out of order"

    # Candidate waves should batch documents and beat sequential execution,
    # which would need one wave per unit generated.
    wave_calls = [(p, n) for p, n in call_log if n > 1]
    sequential_waves = sum(info["unit_count"] for _, info in batched)
    max_docs_per_call = max(p for p, _ in wave_calls)
    print(f"Engine calls: {len(call_log)} total, {len(wave_calls)} candidate waves "
          f"(sequential would need {sequential_waves}); "
          f"max {max_docs_per_call} docs in one call")
    assert len(wave_calls) < sequential_waves, (
        f"pooling broken: {len(wave_calls)} wave calls, sequential needs {sequential_waves}"
    )
    assert max_docs_per_call >= 2, "no wave ever batched multiple documents"
    assert max_docs_per_call <= max_active, "pool exceeded max_active"
    print("PASS: pooled batching extent verified")

    print("\n" + "=" * 60)
    print("ALL EXAMPLES PASSED")
    print("=" * 60)
    return True


def _main():
    """Run CPU-only sampler helper smoke tests."""
    from types import SimpleNamespace

    cfg = SimpleNamespace(temperature=0.8, top_k=40, top_p=0.95, repetition_penalty=1.1, do_sample=True)
    params = resolve_gen_params(cfg)
    assert params["temperature"] == 0.8
    assert params["top_k"] == 40
    assert params["do_sample"] is True
    print(f"resolve_gen_params (explicit): {params}")

    sparse = SimpleNamespace(temperature=None, top_k=None, top_p=None, repetition_penalty=None, do_sample=None)
    defaults = resolve_gen_params(sparse)
    assert defaults["temperature"] == 1.0
    assert defaults["top_k"] == 50
    assert defaults["do_sample"] is True
    print(f"resolve_gen_params (defaults): {defaults}")

    greedy = SimpleNamespace(temperature=None, top_k=None, top_p=None, repetition_penalty=None, do_sample=False)
    g = resolve_gen_params(greedy)
    assert g["do_sample"] is False
    print(f"resolve_gen_params (greedy): do_sample={g['do_sample']}")

    scores = [
        CandidateScore(Region.GREEN, 0.5),
        CandidateScore(Region.YELLOW, 1.0),
        CandidateScore(Region.RED, 0.1),
        CandidateScore(Region.GREEN, 0.2),
    ]
    assert default_accept_fn(scores, margin=0.2) == [True, False, False, False]
    print("default_accept_fn: region and external margin handled correctly")

    base, adapter = resolve_adapter("meta-llama/Llama-3.1-8B")
    assert base == "meta-llama/Llama-3.1-8B" and adapter is None
    print(f"resolve_adapter (plain): base={base!r}, adapter={adapter!r}")

    print("base_sampler smoke ok")


if __name__ == "__main__":
    _main()
