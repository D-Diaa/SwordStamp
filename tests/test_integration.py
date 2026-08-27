"""Test generation-to-detection consistency and watermark integration."""
import os
from pathlib import Path
import unittest

import torch

from watermarking.detect import detect_lsh
from segmentation import segment


def _spacy_available() -> bool:
    try:
        import spacy
        spacy.load("en_core_web_md")
        return True
    except Exception:
        return False


_spacy_skip = unittest.skipUnless(_spacy_available(), "en_core_web_md not installed")


class TestSegmentConsistency(unittest.TestCase):
    """Verify incremental and whole-document segmentation agree."""

    def _simulate_generation(self, prompt, raw_sentences):
        """Replicate incremental continuation assembly."""
        text = prompt
        produced = []
        for raw in raw_sentences:
            units = segment(raw)
            if not units:
                continue
            unit = units[0]
            if not unit.normalized:
                continue
            text += unit.display
            produced.append(unit.normalized)
        return text, produced

    def test_detected_sentences_match_generated_sentences(self):
        prompt = "Once upon a time."
        # Leading spaces simulate BPE-decoded model output
        raw_sents = [
            " The cat sat on the mat.",
            " The dog ran quickly away.",
            " Birds flew overhead in formation.",
            " The sun shone brightly today.",
            " Rain fell softly on the leaves.",
            " The wind carried seeds far away.",
            " Stars appeared at dusk.",
        ]

        full_text, gen_sents = self._simulate_generation(prompt, raw_sents)

        generated_part = full_text[len(prompt):].strip()
        detected_sents = [u.normalized for u in segment(generated_part)]

        self.assertEqual(
            detected_sents, gen_sents,
            "\nIncremental generation and post-hoc segment() disagreed:\n"
            f"  generated : {gen_sents}\n"
            f"  detected  : {detected_sents}",
        )

    def test_accepted_count_consistent_5_of_7(self):
        """Preserve accepted counts after resegmentation."""
        sentences_with_acceptance = [
            (" The cat sat on the mat.", True),
            (" The dog ran quickly away.", False),
            (" Birds flew overhead.", True),
            (" The sun was bright.", True),
            (" Rain fell softly.", False),
            (" The wind carried seeds.", True),
            (" Stars appeared at night.", True),
        ]

        gen_accepted = sum(acc for _, acc in sentences_with_acceptance)  # 5
        raw_sents = [s for s, _ in sentences_with_acceptance]

        prompt = "Here is a story."
        full_text, gen_sents = self._simulate_generation(prompt, raw_sents)

        # Build acceptance map using normalized strings (what detection sees)
        acceptance_map = {}
        for raw, acc in sentences_with_acceptance:
            units = segment(raw)
            if units:
                acceptance_map[units[0].normalized] = acc

        generated_part = full_text[len(prompt):].strip()
        detected_sents = [u.normalized for u in segment(generated_part)]

        det_accepted = sum(acceptance_map.get(s, False) for s in detected_sents)

        self.assertEqual(
            det_accepted, gen_accepted,
            f"Detection found {det_accepted} accepted sentences; generation produced "
            f"{gen_accepted}. segment() is inconsistent between the two phases.",
        )

    def test_single_long_sentence_not_split(self):
        """Keep one long sentence intact during detection."""
        prompt = "Introduction follows."
        long_sent = (
            " The researchers discovered that the new algorithm, which had been "
            "developed over several years, significantly outperformed all prior "
            "baselines on the standard benchmark datasets."
        )

        full_text, gen_sents = self._simulate_generation(prompt, [long_sent])
        generated_part = full_text[len(prompt):].strip()
        detected = [u.normalized for u in segment(generated_part)]

        self.assertEqual(
            len(detected), 1,
            f"A single long sentence was split into {len(detected)} by detection.",
        )

    def test_edge_case_sentence_without_terminal_punctuation(self):
        """Normalize missing terminal punctuation consistently."""
        prompt = "Start."
        raw = " The satellite crossed the night sky"

        full_text, gen_sents = self._simulate_generation(prompt, [raw])
        generated_part = full_text[len(prompt):].strip()
        detected = [u.normalized for u in segment(generated_part)]

        self.assertEqual(len(detected), 1)
        self.assertEqual(detected, gen_sents)

    def test_no_sentences_means_no_detection_sentences(self):
        prompt = "Preamble."
        full_text, gen_sents = self._simulate_generation(prompt, [])
        generated_part = full_text[len(prompt):].strip()
        detected = [u.normalized for u in segment(generated_part)]
        self.assertEqual(detected, [])
        self.assertEqual(gen_sents, [])



def _model_loadable(model_id: str) -> bool:
    """Return True if the model weights are cached locally (no download needed)."""
    local = Path(model_id).expanduser()
    if local.is_dir():
        return any((local / name).is_file() for name in ("config.json", "modules.json"))
    try:
        from huggingface_hub import try_to_load_from_cache
        result = try_to_load_from_cache(model_id, "config.json")
        return isinstance(result, str) and Path(result).is_file()
    except Exception:
        return False


_EMBEDDER = os.getenv("SWORDSTAMP_TEST_EMBEDDER", "AbeHou/SemStamp-c4-sbert")
_SKIP_E2E = unittest.skipUnless(
    torch.cuda.is_available()
    and _model_loadable("Qwen/Qwen2.5-3B-Instruct")
    and _model_loadable(_EMBEDDER),
    "requires CUDA plus locally cached provider and watermark-embedder checkpoints",
)

_LSH_DIM = 8
_LMBD = 0.25
_DELTA = 0.0


class _WatermarkBase(unittest.TestCase):
    """Share model setup across end-to-end watermark tests."""

    SEGMENTATION_TYPE = "sentence"

    @classmethod
    def setUpClass(cls):
        from sampling.base_sampler import create_sampler
        from watermarking.primitives import SBERTLSHModel
        from transformers import GenerationConfig

        cls.device = "cuda:0"
        cls.lsh_model = SBERTLSHModel(
            lsh_model_path=_EMBEDDER, device=cls.device,
            batch_size=1, lsh_dim=_LSH_DIM, sbert_type="base",
        )
        cls.sampler = create_sampler(
            "hf",
            "Qwen/Qwen2.5-3B-Instruct",
            num_candidates=32,
            device=cls.device,
            segmentation_type=cls.SEGMENTATION_TYPE,
        )
        cls.gen_config = GenerationConfig(
            max_new_tokens=205,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.05,
            pad_token_id=cls.sampler.tokenizer.pad_token_id,
        )


@_SKIP_E2E
class TestLSHWatermarkEndToEnd(_WatermarkBase):
    """Test LSH generation, detection, and wrong-key rejection."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from watermarking.scoring import create_lsh_score_fn
        from watermarking.primitives import hash_key
        # Prevent descriptor binding of the score function.
        cls.score_fn = staticmethod(create_lsh_score_fn(
            cls.lsh_model, _LSH_DIM, _LMBD, fixed_seed=None, key=hash_key,
        ))

    def _generate(self, prompt):
        text, info = self.sampler.generate_continuation(
            prompt, self.gen_config, self.score_fn, margin=_DELTA,
        )
        return text, info["accepted_count"], info["unit_count"]

    def test_watermark_z_score_above_null(self):
        """z-score of watermarked text must exceed the null expectation."""
        prompt = "The astronaut stepped onto the surface and looked around."
        text, accepted, total = self._generate(prompt)
        self.assertGreater(total, 0, "No sentences generated")

        units = segment(text[len(prompt):].strip())
        z = detect_lsh(units, self.lsh_model, lmbd=_LMBD, lsh_dim=_LSH_DIM)

        # Watermarked text should have z > 0 with very high probability
        self.assertGreater(z, 2.0,
            f"z-score {z:.3f} not significantly above null — watermark may be broken")

    def test_accepted_count_during_generation_matches_detection(self):
        """Match generation and detection accepted counts."""
        from watermarking.primitives import hash_key
        from watermarking.primitives import get_mask_from_seed

        # Keep the prompt to one segmentation unit.
        prompt = "Scientists announced a significant new discovery yesterday."
        text, gen_accepted, gen_total = self._generate(prompt)

        # Use the prompt unit as the initial detection seed.
        full_sents = [u.normalized for u in segment(text.strip())]
        # full_sents[0] = prompt sentence (seed); full_sents[1:] = generated
        n_generated = len(full_sents) - 1

        self.assertEqual(
            n_generated, gen_total,
            f"segment() found {n_generated} generated sentences, "
            f"but generate_continuation produced {gen_total}. "
            "segment() is inconsistent between incremental and batch mode.",
        )

        det_accepted = 0
        if len(full_sents) > 1:
            lsh_seed = self.lsh_model.get_hash([full_sents[0]])[0]
            mask = get_mask_from_seed(_LSH_DIM, _LMBD, lsh_seed, key=hash_key)
            for s in full_sents[1:]:
                h = self.lsh_model.get_hash([s])[0]
                if h in mask:
                    det_accepted += 1
                lsh_seed = h
                mask = get_mask_from_seed(_LSH_DIM, _LMBD, lsh_seed, key=hash_key)

        self.assertEqual(
            det_accepted, gen_accepted,
            f"Generation accepted {gen_accepted}/{gen_total} sentences, "
            f"but detection found {det_accepted}/{n_generated}. "
            "score_fn decisions are inconsistent with post-hoc detection.",
        )

    def test_wrong_key_gives_lower_z_score(self):
        """Require a lower z-score under the wrong key."""
        from watermarking.primitives import hash_key

        prompt = "The engineer reviewed the blueprints one final time."
        text, _, _ = self._generate(prompt)
        units = segment(text[len(prompt):].strip())

        z_correct = detect_lsh(units, self.lsh_model, lmbd=_LMBD,
                               lsh_dim=_LSH_DIM, key=hash_key)
        z_wrong = detect_lsh(units, self.lsh_model, lmbd=_LMBD,
                             lsh_dim=_LSH_DIM, key=hash_key + 1)

        self.assertGreater(z_correct, z_wrong,
            f"Correct key z={z_correct:.3f} should exceed wrong key z={z_wrong:.3f}")


@_SKIP_E2E
class TestFixedModeWatermark(_WatermarkBase):
    """Tests fixed-mode (secret-message) generation and detection."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from watermarking.scoring import create_lsh_score_fn
        from watermarking.primitives import hash_key

        cls.secret_a = "The magic words are squeamish ossifrage."
        cls.secret_b = "An entirely different secret passphrase."
        fixed_seed_a = cls.lsh_model.get_hash([cls.secret_a])[0]
        cls.score_fn_a = staticmethod(create_lsh_score_fn(
            cls.lsh_model, _LSH_DIM, _LMBD,
            fixed_seed=fixed_seed_a, key=hash_key,
        ))

    def test_correct_secret_detects_watermark(self):
        prompt = "The researcher opened the lab notebook."
        text, _info = self.sampler.generate_continuation(
            prompt, self.gen_config, self.score_fn_a, margin=_DELTA,
        )
        units = segment(text[len(prompt):].strip())
        z = detect_lsh(units, self.lsh_model, lmbd=_LMBD, lsh_dim=_LSH_DIM,
                       secret_message=self.secret_a)
        self.assertGreater(z, 2.0,
            f"Correct-secret z={z:.3f} is not above null — fixed mode may be broken")

    def test_wrong_secret_gives_lower_z_score(self):
        prompt = "The researcher opened the lab notebook."
        text, _info = self.sampler.generate_continuation(
            prompt, self.gen_config, self.score_fn_a, margin=_DELTA,
        )
        units = segment(text[len(prompt):].strip())

        z_right = detect_lsh(units, self.lsh_model, lmbd=_LMBD, lsh_dim=_LSH_DIM,
                              secret_message=self.secret_a)
        z_wrong = detect_lsh(units, self.lsh_model, lmbd=_LMBD, lsh_dim=_LSH_DIM,
                              secret_message=self.secret_b)

        self.assertGreater(z_right, z_wrong,
            f"Correct secret z={z_right:.3f} should beat wrong secret z={z_wrong:.3f}")


if __name__ == "__main__":
    unittest.main()
