"""LLM-as-judge quality metrics."""

import gc
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import yaml
from tqdm import tqdm

from config.runtime import vllm_gpu_memory_utilization

from .common import _confidence_interval


@dataclass(frozen=True)
class SamplingProfile:
    """Define local judge decoding parameters."""

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1  # vLLM: <= 0 disables top-k
    min_p: float = 0.0
    seed: int = 0


# Qwen3 thinking-mode sampling, straight from the Qwen3 model card.
_QWEN3_SAMPLING = SamplingProfile(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
_DEFAULT_SAMPLING = SamplingProfile()


@dataclass(frozen=True)
class ModelFamily:
    """Everything judge inference needs to know about one model family.

    `thinking` families reason before answering, so the rubric asks only for
    the final score.  `tagged_thinking` families additionally expose that
    reasoning inline, terminated by a literal </think> marker; hidden-reasoning
    APIs (for example GPT-5) must not set it, since their response content
    already contains only the final answer.
    """

    thinking: bool = False
    tagged_thinking: bool = False
    sampling: SamplingProfile = _DEFAULT_SAMPLING
    vllm_kwargs: dict = field(default_factory=dict)


def _qwen3_family(**vllm_extra):
    return ModelFamily(
        thinking=True,
        tagged_thinking=True,
        sampling=_QWEN3_SAMPLING,
        # Records the correct reasoning format in the engine. Offline generate
        # still returns raw text, so _final_answer_text performs the extraction.
        vllm_kwargs={"reasoning_parser": "qwen3", **vllm_extra},
    )


# Reasoning APIs decode remotely; the profile only matters if run under vLLM.
_HIDDEN_THINKING_FAMILY = ModelFamily(thinking=True, sampling=_QWEN3_SAMPLING)

# Judge behavior keyed by model-name prefix (longest match wins). Adding a
# model family means adding exactly one entry here.
_MODEL_FAMILIES = {
    "Qwen/Qwen3": _qwen3_family(),
    # Qwen3.5/3.6 checkpoints include a vision encoder even for text-only jobs.
    # vLLM can omit it and dedicate the recovered memory to the language model.
    "Qwen/Qwen3.5": _qwen3_family(language_model_only=True),
    "Qwen/Qwen3.6": _qwen3_family(language_model_only=True),
    "deepseek-ai/DeepSeek-R1": ModelFamily(
        thinking=True,
        tagged_thinking=True,
        sampling=SamplingProfile(temperature=0.6, top_p=0.95),
    ),
    "openai/gpt-oss": _HIDDEN_THINKING_FAMILY,
    "gpt-5": _HIDDEN_THINKING_FAMILY,
    "o1-": _HIDDEN_THINKING_FAMILY,
    "o3-": _HIDDEN_THINKING_FAMILY,
    "o4-": _HIDDEN_THINKING_FAMILY,
}
_DEFAULT_FAMILY = ModelFamily()


def _family_for(model_name):
    """Resolve a model's judge behavior by longest matching name prefix."""
    matches = [(p, f) for p, f in _MODEL_FAMILIES.items() if model_name.startswith(p)]
    if not matches:
        return _DEFAULT_FAMILY
    return max(matches, key=lambda kv: len(kv[0]))[1]


def _final_answer_text(text, model_name):
    """Return only the answer after a local model's completed reasoning block.

    A missing closing marker means the decode ended inside its reasoning trace.
    Returning an empty string deliberately makes that output fail parsing and
    enter the existing retry path; criterion-like JSON in the trace must never
    be accepted as the judge's final decision.
    """
    text = text or ""
    if not _family_for(model_name).tagged_thinking:
        return text
    marker = "</think>"
    if marker not in text:
        return ""
    return text.rsplit(marker, 1)[1].strip()


# Keep the document payload and judge reasoning independently bounded.  The
# rendered system/user prompt still has to share the model's 16K context with
# the output, so VLLMJudge further reduces max_tokens per request when needed.
_JUDGE_SAMPLE_MAX_TOKENS = 8192
_JUDGE_MAX_TOKENS = 8192
_JUDGE_MAX_MODEL_LEN = 16384
_JUDGE_MAX_RETRIES = 3
_OPENAI_JUDGE_MAX_TOKENS = 16384


@dataclass(frozen=True)
class JudgeOutput:
    """One judge decode plus the operational metadata needed for auditing."""

    text: str
    generated_tokens: int = -1
    finish_reason: str = "unknown"
    max_tokens: int = -1
    seed: int = -1
    error: str = ""


# Per-attempt auditing arrays, as metric-key stem -> (JudgeOutput field, fill,
# dtype).  Adding an auditing field means adding one row here.
_ATTEMPT_FIELDS = {
    "attempt_tokens": ("generated_tokens", -1, np.int64),
    "attempt_max_tokens": ("max_tokens", -1, np.int64),
    "attempt_seeds": ("seed", -1, np.int64),
    "attempt_finish_reason": ("finish_reason", "", "<U32"),
    "attempt_error": ("error", "", "<U512"),
}


def _judge_telemetry_arrays(n, repeats):
    """Allocate the per-sample judge auditing arrays, filled with sentinels.

    Single source of truth for their keys, shapes and dtypes: both the empty
    and the populated rubric schema are built from this.
    """
    attempt_shape = (n, repeats, _JUDGE_MAX_RETRIES + 1)
    arrays = {
        "input_tokens": np.full(n, -1, dtype=np.int64),
        "truncated_tokens": np.zeros(n, dtype=np.int64),
        "valid_repetitions": np.zeros(n, dtype=np.int64),
        "accepted_attempt": np.full((n, repeats), -1, dtype=np.int64),
    }
    for key, (_field, fill, dtype) in _ATTEMPT_FIELDS.items():
        arrays[key] = np.full(attempt_shape, fill, dtype=dtype)
    return arrays


@dataclass(frozen=True, kw_only=True)
class JudgeRubric:
    """Define a weighted 0-5 judging rubric and sample formatter."""

    name: str
    role: str
    dimensions: tuple
    scale_block: str
    rules_block: str
    build_user: Callable[[dict], str]
    valence: dict = None

    @property
    def dim_keys(self):
        return tuple(d[0] for d in self.dimensions)

    @property
    def weights(self):
        total = sum(d[1] for d in self.dimensions)
        return {d[0]: d[1] / total for d in self.dimensions}

    def _criteria_block(self):
        return "\n".join(f'{i + 1}. "{k}" - {desc}' for i, (k, _w, desc) in enumerate(self.dimensions))

    def system_prompt(self, model_name):
        json_example = "{" + ", ".join(f'"{k}": <0-5>' for k in self.dim_keys) + "}"
        tail = (
            "output a SINGLE line of strict JSON wrapped in <json></json> tags, with an "
            "integer 0-5 for every criterion key, and nothing after the closing tag:"
        )
        if _family_for(model_name).thinking:
            # Let reasoning models deliberate internally before emitting JSON.
            output_instr = "Think through each criterion carefully before scoring. Then " + tail
        else:
            # Force visible reasoning-before-score.
            output_instr = (
                "First, in one or two sentences, note how the text does on each "
                "criterion. Then " + tail
            )
        return (
            f"{self.role}\n\n"
            f"Evaluate the text on these criteria:\n{self._criteria_block()}\n\n"
            f"{self.scale_block}\n\n{self.rules_block}\n\n"
            f"{output_instr}\n<json>{json_example}</json>"
        )


_RUBRICS_PATH = Path(__file__).with_name("rubrics.yaml")


def _build_rubric(spec):
    """Build a rubric from one YAML entry."""
    return JudgeRubric(
        name=spec["name"],
        role=spec["role"],
        dimensions=tuple(
            (d["key"], d["weight"], d["description"]) for d in spec["dimensions"]
        ),
        scale_block=spec["scale_block"],
        rules_block=spec["rules_block"],
        valence=spec.get("valence"),
        build_user=lambda s, template=spec["user_template"]: template.format(**s),
    )


def _load_rubrics(path=_RUBRICS_PATH):
    """Load every rubric defined in rubrics.yaml, keyed by name."""
    with open(path, encoding="utf-8") as f:
        specs = yaml.safe_load(f)
    return {spec["name"]: _build_rubric(spec) for spec in specs}


_RUBRICS = _load_rubrics()
PAIRWISE_RUBRIC = _RUBRICS["llm_judge"]
QUALITY_RUBRIC = _RUBRICS["llm_quality"]


def _is_openai_model(model_name):
    return any(model_name.startswith(p) for p in ("gpt-", "o1-", "o3-", "o4-"))


def _clip05(v):
    """Coerce to a float in [0, 5]; nan if not numeric."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return min(max(f, 0.0), 5.0)


def _parse_dimension_scores(text, dim_keys, *, strict=False):
    """Extract available 0-5 criterion scores from a judge response.

    `strict` demands one complete <json></json> decision: no loose "key: n"
    fallback, and all-or-nothing on the criterion set.  A loose or malformed
    criterion-like fragment is not a completed final decision, even if every
    key happens to be visible in it.
    """
    text = text or ""
    scores = {}
    m = re.search(r"<json>\s*(\{.*?\})\s*</json>", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            for k in dim_keys:
                if k in obj:
                    s = _clip05(obj[k])
                    if not np.isnan(s):
                        scores[k] = s
        except (ValueError, TypeError):
            pass
    if strict:
        return scores if len(scores) == len(dim_keys) else {}
    for k in dim_keys:
        if k not in scores:
            mk = re.search(rf'"?{re.escape(k)}"?\s*[:=]\s*(-?\d+(?:\.\d+)?)', text)
            if mk:
                s = _clip05(mk.group(1))
                if not np.isnan(s):
                    scores[k] = s
    return scores


def _overall_from_dims(scores, weights, valence=None):
    """Return the normalized weighted mean, accounting for criterion valence."""
    if not all(k in scores for k in weights):
        return float("nan")
    valence = valence or {}
    return sum(
        weights[k] * (scores[k] if valence.get(k, 1) > 0 else 5.0 - scores[k])
        for k in weights
    ) / 5.0


def empty_rubric_metrics(rubric, n, repeats):
    """Result dict for a rubric that did not run, matching the populated schema."""
    out = {
        rubric.name: float("nan"),
        f"{rubric.name}_ci": float("nan"),
        f"{rubric.name}_median": float("nan"),
        f"{rubric.name}_per_sample": np.full(n, np.nan),
    }
    out.update(
        {
            f"{rubric.name}_{key}_per_sample": value
            for key, value in _judge_telemetry_arrays(n, repeats).items()
        }
    )
    for k in rubric.dim_keys:
        out[f"{rubric.name}_{k}_per_sample"] = np.full(n, np.nan)
        out[f"{rubric.name}_{k}_repeats_per_sample"] = np.full(
            (n, repeats), np.nan
        )
    return out


def _truncate_input_text(sample, tokenizer, max_tokens=_JUDGE_SAMPLE_MAX_TOKENS):
    """Cap only the candidate/generated text; preserve prompt and reference."""
    truncated = dict(sample)
    text = sample.get("gen")
    if not isinstance(text, str):
        return truncated, 0, 0
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    original_tokens = len(token_ids)
    if original_tokens <= max_tokens:
        return truncated, original_tokens, 0
    truncated["gen"] = tokenizer.decode(
        token_ids[:max_tokens],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return truncated, max_tokens, original_tokens - max_tokens


def _run_llm_judge_openai(
    user_contents, system_prompt, model_name, batch_size, parse_ok, retry_user,
):
    """Return OpenAI judge outputs using the established API retry path."""
    from openai import OpenAI

    client = OpenAI()

    def score_one(user_content):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        def create(request_messages):
            response = client.chat.completions.create(
                model=model_name,
                messages=request_messages,
                max_completion_tokens=_OPENAI_JUDGE_MAX_TOKENS,
            )
            return response.choices[0].message.content or ""

        raw = create(messages)
        if not parse_ok(raw):
            retry = create(
                messages
                + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": retry_user},
                ]
            )
            raw = retry if parse_ok(retry) else f"{raw}\n{retry}"
        return raw

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        return list(
            tqdm(
                executor.map(score_one, user_contents),
                total=len(user_contents),
                desc="LLM Judge (OpenAI)",
            )
        )


def _format_chat_for_generation(tokenizer, messages, model_name):
    formatted = []
    thinking_kwargs = (
        {"enable_thinking": True}
        if _family_for(model_name).tagged_thinking
        else {}
    )
    for chat in messages:
        continue_final_message = chat[-1]["role"] == "assistant"
        prompt = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=not continue_final_message,
            continue_final_message=continue_final_message,
            **thinking_kwargs,
        )
        formatted.append(prompt)
    return formatted


class VLLMJudge:
    """Reusable vLLM engine for local LLM-as-judge inference."""

    def __init__(
        self,
        model_name,
        *,
        device="cuda",
        dtype="auto",
        tensor_parallel_size=1,
        gpu_memory_utilization=None,
        max_model_len=None,
    ):
        if gpu_memory_utilization is None:
            gpu_memory_utilization = vllm_gpu_memory_utilization()
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "Local LLM judge now uses vLLM. Install vLLM in this environment "
                "before setting quality.judge_model to a non-OpenAI model."
            ) from e

        if isinstance(device, str) and device.startswith("cuda"):
            cuda_device = torch.device(device)
            if cuda_device.index is not None:
                torch.cuda.set_device(cuda_device)
        elif isinstance(device, int):
            torch.cuda.set_device(device)

        self.model_name = model_name
        self.family = _family_for(model_name)
        self.sampling_params_cls = SamplingParams
        self.max_model_len = max_model_len or _JUDGE_MAX_MODEL_LEN
        llm_kwargs = dict(
            model=model_name,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        llm_kwargs.update(self.family.vllm_kwargs)
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        self.llm = LLM(**llm_kwargs)
        if hasattr(self.llm, "get_tokenizer"):
            self.tokenizer = self.llm.get_tokenizer()
        else:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # A repetition or retry uses the same original chat. Cache the exact
        # rendered token IDs for the lifetime of this judge instead of applying
        # the chat template and tokenizing up to twelve times per sample.
        self._prompt_token_ids_cache = {}

    def _prompt_token_ids(self, chat):
        key = tuple((message["role"], message["content"]) for message in chat)
        cache = getattr(self, "_prompt_token_ids_cache", None)
        if cache is None:
            cache = self._prompt_token_ids_cache = {}
        if key not in cache:
            prompt = _format_chat_for_generation(
                self.tokenizer, [chat], self.model_name
            )[0]
            cache[key] = tuple(
                self.tokenizer.encode(prompt, add_special_tokens=False)
            )
        return cache[key]

    def generate(self, messages, *, desc, seed_offset=0, seed_offsets=None):
        if not messages:
            return []
        if seed_offsets is None:
            seed_offsets = [seed_offset] * len(messages)
        elif len(seed_offsets) != len(messages):
            raise ValueError("seed_offsets must match the number of messages")
        # Sampling avoids reasoning-model repetition. Each request gets only
        # the context space left after its fully rendered chat prompt.
        prof = self.family.sampling
        results = [None] * len(messages)
        valid_indices = []
        valid_prompts = []
        sampling_params = []
        valid_seeds = []
        for i, (chat, request_seed_offset) in enumerate(
            zip(messages, seed_offsets)
        ):
            seed = prof.seed + request_seed_offset
            prompt_token_ids = self._prompt_token_ids(chat)
            input_tokens = len(prompt_token_ids)
            max_tokens = min(_JUDGE_MAX_TOKENS, self.max_model_len - input_tokens)
            if max_tokens < 1:
                results[i] = JudgeOutput(
                    text="",
                    finish_reason="context_exhausted",
                    max_tokens=max_tokens,
                    seed=seed,
                    error=(
                        f"rendered prompt has {input_tokens} tokens for a "
                        f"{self.max_model_len}-token context"
                    ),
                )
                continue
            valid_indices.append(i)
            valid_seeds.append(seed)
            # Pass the exact IDs used for context accounting so vLLM does not
            # tokenize the rendered chat a second time.
            valid_prompts.append({"prompt_token_ids": list(prompt_token_ids)})
            sampling_params.append(
                self.sampling_params_cls(
                    temperature=prof.temperature,
                    top_p=prof.top_p,
                    top_k=prof.top_k,
                    min_p=prof.min_p,
                    seed=seed,
                    max_tokens=max_tokens,
                )
            )

        if valid_prompts:
            outputs = self.llm.generate(
                valid_prompts,
                sampling_params,
                use_tqdm=partial(tqdm, desc=desc),
            )
            if len(outputs) != len(valid_prompts):
                raise RuntimeError(
                    f"vLLM returned {len(outputs)} outputs for "
                    f"{len(valid_prompts)} prompts"
                )
            for i, out, params, seed in zip(
                valid_indices, outputs, sampling_params, valid_seeds
            ):
                if out.outputs:
                    completion = out.outputs[0]
                    results[i] = JudgeOutput(
                        text=completion.text,
                        generated_tokens=len(completion.token_ids),
                        finish_reason=completion.finish_reason or "unknown",
                        max_tokens=params.max_tokens,
                        seed=seed,
                    )
                else:
                    results[i] = JudgeOutput(
                        text="", finish_reason="empty", max_tokens=params.max_tokens,
                        seed=seed,
                    )
        return results


def build_llm_judge_pipe(
    model_name, device="cuda", gpu_memory_utilization=None,
):
    """Build a reusable local vLLM judge engine."""
    tensor_parallel_size = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1"))
    gpu_memory_utilization = vllm_gpu_memory_utilization(gpu_memory_utilization)
    max_model_len_env = os.getenv("VLLM_MAX_MODEL_LEN", str(_JUDGE_MAX_MODEL_LEN))
    max_model_len = int(max_model_len_env) if max_model_len_env else None
    return VLLMJudge(
        model_name,
        device=device,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )


def _run_llm_judge_local(
    user_contents,
    system_prompt,
    model_name,
    *,
    pipe,
    seed_offset=0,
    seed_offsets=None,
):
    """Return one JudgeOutput per input; isolate batch failures by bisection.

    Retry scheduling lives in `evaluate_llm_judge`, which drives this once per
    attempt with a fresh seed offset.
    """
    messages = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        for user_content in user_contents
    ]
    if seed_offsets is None:
        seed_offsets = [seed_offset] * len(messages)
    elif len(seed_offsets) != len(messages):
        raise ValueError("seed_offsets must match the number of user contents")

    def generate_group(group, indices, group_seed_offsets):
        try:
            values = pipe.generate(
                group,
                desc=f"LLM Judge ({model_name})",
                seed_offsets=group_seed_offsets,
            )
            if len(values) != len(group):
                raise RuntimeError(
                    f"judge returned {len(values)} outputs for {len(group)} inputs"
                )
            return {
                i: value
                for i, value in zip(indices, values)
            }
        except Exception as exc:
            if len(group) > 1:
                middle = len(group) // 2
                return {
                    **generate_group(
                        group[:middle], indices[:middle], group_seed_offsets[:middle]
                    ),
                    **generate_group(
                        group[middle:], indices[middle:], group_seed_offsets[middle:]
                    ),
                }
            return {
                indices[0]: JudgeOutput(
                    text="", finish_reason="error", seed=group_seed_offsets[0],
                    error=f"{type(exc).__name__}: {exc}",
                )
            }

    by_index = generate_group(
        messages, list(range(len(messages))), list(seed_offsets)
    )
    raw_outputs = [by_index[i] for i in range(len(messages))]

    return raw_outputs


def _evaluate_llm_judge(samples, qcfg, rubric, debug, pipe):
    """Evaluate with engine ownership already resolved by the public wrapper."""
    judge_model = qcfg.judge_model
    n = len(samples)
    if judge_model is None:
        return empty_rubric_metrics(rubric, n, qcfg.judge_repeats)

    batch_size = qcfg.judge_batch_size
    repeats = qcfg.judge_repeats
    system_prompt = rubric.system_prompt(judge_model)
    dim_keys = rubric.dim_keys
    weights = rubric.weights

    is_openai = _is_openai_model(judge_model)

    def parse_scores(raw):
        text = raw.text if isinstance(raw, JudgeOutput) else raw
        final = _final_answer_text(text, judge_model)
        # Local decodes must show one complete <json> decision; the hidden
        # reasoning APIs return only a final answer, so loose keys are fine.
        return _parse_dimension_scores(final, dim_keys, strict=not is_openai)

    def parse_ok(raw):
        return len(parse_scores(raw)) == len(dim_keys)

    retry_user = (
        "Output ONLY the <json></json> line with an integer 0-5 for each of these keys: "
        + ", ".join(dim_keys)
        + ". Do not repeat your analysis or add any other text."
    )

    telemetry = _judge_telemetry_arrays(n, repeats)
    input_tokens = telemetry["input_tokens"]
    truncated_tokens = telemetry["truncated_tokens"]

    # Input truncation is a local-engine safeguard only. OpenAI requests stay
    # on their simple API path without local token accounting.
    judged_samples = list(samples)
    if not is_openai:
        judged_samples = []
        for i, sample in enumerate(samples):
            gen = sample.get("gen")
            if gen and gen.strip():
                capped, kept, removed = _truncate_input_text(
                    sample, pipe.tokenizer
                )
            else:
                capped, kept, removed = dict(sample), 0, 0
            judged_samples.append(capped)
            input_tokens[i] = kept
            truncated_tokens[i] = removed
    n_truncated = int(np.count_nonzero(truncated_tokens))
    if n_truncated:
        print(
            f"LLM Judge [{rubric.name}]: truncated {n_truncated}/{n} generated "
            f"texts to {_JUDGE_SAMPLE_MAX_TOKENS} tokens"
        )

    # Separate empty gens (score 0) from non-empty ones to avoid wasting LLM calls
    nonempty_indices = []
    nonempty_user_contents = []
    for i, s in enumerate(judged_samples):
        gen = s.get("gen")
        if gen and gen.strip():
            nonempty_indices.append(i)
            nonempty_user_contents.append(rubric.build_user(s))

    n_empty = n - len(nonempty_user_contents)
    if n_empty:
        print(f"LLM Judge [{rubric.name}]: {n_empty}/{n} empty generations scored 0")

    # Empty gens default every criterion to 0; parsed gens fill in, failures nan.
    dim_raw = {k: [0.0] * n for k in dim_keys}  # per-criterion 0-5
    dim_repeats = {
        k: np.zeros((n, repeats), dtype=float) for k in dim_keys
    }  # per-criterion 0-5, before mean reduction
    overall = [0.0] * n  # aggregated 0-1
    # Raw decodes are only ever read back under `debug`; holding n x repeats
    # multi-KB texts otherwise costs tens of MB for nothing.
    all_raw_outputs = (
        [["[empty generation - score 0]"] * repeats for _ in range(n)] if debug else None
    )
    accepted_attempt = telemetry["accepted_attempt"]
    if nonempty_user_contents:
        repeated_scores = [[] for _ in nonempty_user_contents]
        repeated_raw = [[] for _ in nonempty_user_contents] if debug else None
        if is_openai:
            for repeat_idx in range(repeats):
                print(
                    f"LLM Judge [{rubric.name}]: repetition "
                    f"{repeat_idx + 1}/{repeats}"
                )
                raw_outputs = _run_llm_judge_openai(
                    nonempty_user_contents,
                    system_prompt,
                    judge_model,
                    batch_size,
                    parse_ok,
                    retry_user,
                )
                for sample_idx, raw in enumerate(raw_outputs):
                    if debug:
                        repeated_raw[sample_idx].append(raw)
                    repeated_scores[sample_idx].append(parse_scores(raw))
        else:
            # Queue every logical repetition together. vLLM can continuously
            # schedule the combined wave instead of draining the GPU once per
            # repetition; each request retains its original independent seed.
            accepted = [
                [({}, "") for _ in range(repeats)]
                for _ in nonempty_user_contents
            ]
            pending = [
                (sample_idx, repeat_idx)
                for sample_idx in range(len(nonempty_user_contents))
                for repeat_idx in range(repeats)
            ]
            for attempt_idx in range(_JUDGE_MAX_RETRIES + 1):
                if not pending:
                    break
                pending_contents = [
                    nonempty_user_contents[sample_idx]
                    for sample_idx, _repeat_idx in pending
                ]
                seed_offsets = [
                    repeat_idx + attempt_idx * repeats
                    for _sample_idx, repeat_idx in pending
                ]
                print(
                    f"LLM Judge [{rubric.name}]: packed repetitions "
                    f"({len(pending)} requests)"
                )
                outputs = _run_llm_judge_local(
                    pending_contents,
                    system_prompt,
                    judge_model,
                    pipe=pipe,
                    seed_offsets=seed_offsets,
                )

                still_pending = []
                for (sample_idx, repeat_idx), output in zip(pending, outputs):
                    global_idx = nonempty_indices[sample_idx]
                    for key, (attr, _fill, _dtype) in _ATTEMPT_FIELDS.items():
                        telemetry[key][global_idx, repeat_idx, attempt_idx] = getattr(
                            output, attr
                        )
                    scores = parse_scores(output)
                    if len(scores) == len(dim_keys):
                        accepted[sample_idx][repeat_idx] = (scores, output.text)
                        accepted_attempt[global_idx, repeat_idx] = attempt_idx
                    else:
                        still_pending.append((sample_idx, repeat_idx))
                pending = still_pending
                if pending and attempt_idx < _JUDGE_MAX_RETRIES:
                    print(
                        f"Retrying {len(pending)} invalid judge outputs from "
                        f"their original prompts ({attempt_idx + 1}/"
                        f"{_JUDGE_MAX_RETRIES})..."
                    )

            for sample_idx, repetitions in enumerate(accepted):
                for scores, output_text in repetitions:
                    repeated_scores[sample_idx].append(scores)
                    if debug:
                        repeated_raw[sample_idx].append(output_text)

        for sample_idx, (idx, score_group) in enumerate(
            zip(nonempty_indices, repeated_scores)
        ):
            if debug:
                all_raw_outputs[idx] = repeated_raw[sample_idx]
            n_valid = sum(len(scores) == len(dim_keys) for scores in score_group)
            telemetry["valid_repetitions"][idx] = n_valid
            sc = {}
            complete = n_valid == repeats
            for k in dim_keys:
                dim_repeats[k][idx] = [
                    scores.get(k, float("nan")) for scores in score_group
                ]
                if complete:
                    sc[k] = float(np.mean([scores[k] for scores in score_group]))
                dim_raw[k][idx] = sc.get(k, float("nan"))
            overall[idx] = _overall_from_dims(sc, weights, rubric.valence)

    overall = np.asarray(overall, dtype=float)
    valid = overall[~np.isnan(overall)]
    n_failed = int(np.isnan(overall).sum())
    if n_failed:
        print(
            f"Warning: LLM Judge [{rubric.name}] lacked all required valid "
            f"repetitions for {n_failed}/{n} samples"
        )
    if valid.size == 0:
        print(f"Warning: LLM Judge [{rubric.name}] produced no parseable scores")

    stats = {
        rubric.name: float(np.mean(valid)) if valid.size else float("nan"),
        f"{rubric.name}_ci": _confidence_interval(valid) if valid.size else float("nan"),
        f"{rubric.name}_median": float(np.median(valid)) if valid.size else float("nan"),
        # index-matched with `samples` (nan where unparsed; 0 where empty gen)
        f"{rubric.name}_per_sample": overall,
    }
    stats.update(
        {
            f"{rubric.name}_{key}_per_sample": value
            for key, value in telemetry.items()
        }
    )
    # per-criterion per-sample, normalized 0-1 to match the headline scale
    for k in dim_keys:
        stats[f"{rubric.name}_{k}_per_sample"] = np.asarray(dim_raw[k], dtype=float) / 5.0
        stats[f"{rubric.name}_{k}_repeats_per_sample"] = dim_repeats[k] / 5.0
    if debug:
        stats["raw_outputs"] = all_raw_outputs
        stats["all_scores"] = overall.tolist()
    return stats


def evaluate_llm_judge(
    samples, qcfg, rubric=PAIRWISE_RUBRIC, debug=False, pipe=None,
):
    """Score samples while safely owning a local vLLM engine when necessary."""
    judge_model = qcfg.judge_model
    if judge_model is None:
        return empty_rubric_metrics(rubric, len(samples), qcfg.judge_repeats)

    has_nonempty = any(
        isinstance(sample.get("gen"), str) and sample["gen"].strip()
        for sample in samples
    )
    needs_pipe = not _is_openai_model(judge_model) and pipe is None and has_nonempty
    if not needs_pipe:
        return _evaluate_llm_judge(samples, qcfg, rubric, debug, pipe)

    owned_pipe = build_llm_judge_pipe(judge_model)
    try:
        return _evaluate_llm_judge(samples, qcfg, rubric, debug, owned_pipe)
    finally:
        del owned_pipe
        gc.collect()
        torch.cuda.empty_cache()
