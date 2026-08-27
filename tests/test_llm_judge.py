import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from config.schema import QualityConfig
from quality.evaluator import evaluate_llm_judge_metrics
from quality.metrics.llm_judge import (
    _JUDGE_MAX_RETRIES,
    _JUDGE_MAX_TOKENS,
    _JUDGE_MAX_MODEL_LEN,
    _JUDGE_SAMPLE_MAX_TOKENS,
    JudgeOutput,
    PAIRWISE_RUBRIC,
    QUALITY_RUBRIC,
    VLLMJudge,
    _family_for,
    _final_answer_text,
    _format_chat_for_generation,
    _parse_dimension_scores,
    _run_llm_judge_local,
    _truncate_input_text,
    evaluate_llm_judge,
)


class _RecordingTokenizer:
    def __init__(self):
        self.calls = []
        self.encode_calls = 0

    def apply_chat_template(self, chat, **kwargs):
        self.calls.append(kwargs)
        return "formatted"

    def encode(self, text, add_special_tokens=False):
        _ = add_special_tokens
        self.encode_calls += 1
        return text.split()

    def decode(self, token_ids, **kwargs):
        _ = kwargs
        return " ".join(token_ids)


class _FixedJudge:
    def __init__(self, output):
        self.output = output
        self.tokenizer = _RecordingTokenizer()

    def generate(self, messages, *, desc, seed_offset=0, seed_offsets=None):
        seeds = seed_offsets or [seed_offset] * len(messages)
        return [
            JudgeOutput(text=self.output, seed=seed)
            for seed in seeds
        ]


class _RepeatedJudge:
    def __init__(self, outputs):
        self.outputs = outputs
        self.seed_offsets = []
        self.batch_sizes = []
        self.tokenizer = _RecordingTokenizer()

    def generate(self, messages, *, desc, seed_offset=0, seed_offsets=None):
        seeds = seed_offsets or [seed_offset] * len(messages)
        self.seed_offsets.extend(seeds)
        self.batch_sizes.append(len(messages))
        return [
            JudgeOutput(text=self.outputs[seed], seed=seed)
            for seed in seeds
        ]


class _SeedJudge:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []
        self.tokenizer = _RecordingTokenizer()

    def generate(self, messages, *, desc, seed_offset=0, seed_offsets=None):
        _ = desc
        seeds = seed_offsets or [seed_offset] * len(messages)
        self.calls.extend((seed, [message]) for seed, message in zip(seeds, messages))
        return [
            JudgeOutput(
                text=self.outputs.get(seed, "invalid"),
                seed=seed,
            )
            for seed in seeds
        ]


class _SamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _RecordingLLM:
    def __init__(self):
        self.calls = []

    def generate(self, prompts, sampling_params, use_tqdm):
        _ = use_tqdm
        self.calls.append((prompts, sampling_params))
        return [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        text="answer", token_ids=[1, 2], finish_reason="stop"
                    )
                ]
            )
            for _ in prompts
        ]


class _NoCompletionLLM:
    """Engine that accepts a request but returns no completion for it."""

    def generate(self, prompts, sampling_params, use_tqdm):
        _ = sampling_params, use_tqdm
        return [SimpleNamespace(outputs=[]) for _ in prompts]


class _ShortVLLM:
    """Underlying vLLM engine that violates request/output cardinality."""

    def generate(self, prompts, sampling_params, use_tqdm):
        _ = sampling_params, use_tqdm
        return [] if prompts else []


class _EchoTokenizer(_RecordingTokenizer):
    """Renders each chat to its final message, so prompt length varies."""

    def apply_chat_template(self, chat, **kwargs):
        self.calls.append(kwargs)
        return chat[-1]["content"]


class _PoisonedJudge:
    """Raises for any batch containing the poisoned input, as an OOM would."""

    def __init__(self, poison):
        self.poison = poison
        self.tokenizer = _RecordingTokenizer()
        self.batch_sizes = []

    def generate(self, messages, *, desc, seed_offset=0, seed_offsets=None):
        _ = desc
        seeds = seed_offsets or [seed_offset] * len(messages)
        self.batch_sizes.append(len(messages))
        if any(self.poison in chat[1]["content"] for chat in messages):
            raise RuntimeError("CUDA out of memory")
        return [JudgeOutput(text="good output", seed=seed) for seed in seeds]


class _ShortJudge:
    def __init__(self):
        self.tokenizer = _RecordingTokenizer()

    def generate(self, messages, *, desc, seed_offset=0, seed_offsets=None):
        _ = desc
        seeds = seed_offsets or [seed_offset] * len(messages)
        if len(messages) > 1:
            return [JudgeOutput(text="only one output", seed=seeds[0])]
        user_text = messages[0][1]["content"]
        return (
            []
            if "bad" in user_text
            else [JudgeOutput(text="good output", seed=seeds[0])]
        )


class _BrokenTokenizer(_RecordingTokenizer):
    def encode(self, text, add_special_tokens=False):
        _ = text, add_special_tokens
        raise RuntimeError("tokenizer failed")


class LLMJudgeReasoningTests(unittest.TestCase):
    def test_attack_evaluation_skips_intrinsic_rubric(self):
        def fake_judge(samples, _qcfg, rubric, pipe=None):
            _ = samples, pipe
            return {rubric.name: 0.75}

        with patch(
            "quality.evaluator.evaluate_llm_judge", side_effect=fake_judge
        ) as judge:
            stats = evaluate_llm_judge_metrics(
                ["attacked"],
                ["watermarked"],
                QualityConfig(judge_model="gpt-5.6", judge_repeats=3),
                eval_intrinsic=False,
                eval_pairwise=True,
            )

        self.assertEqual(judge.call_count, 1)
        self.assertIs(judge.call_args.kwargs["rubric"], PAIRWISE_RUBRIC)
        self.assertTrue(np.isnan(stats["llm_quality"]))
        self.assertEqual(stats["llm_judge"], 0.75)

    def test_judge_model_length_defaults_to_16k(self):
        self.assertEqual(_JUDGE_MAX_MODEL_LEN, 16384)

    def test_local_judge_output_and_input_caps_are_8k(self):
        self.assertEqual(_JUDGE_MAX_TOKENS, 8192)
        self.assertEqual(_JUDGE_SAMPLE_MAX_TOKENS, 8192)
        self.assertEqual(_JUDGE_MAX_RETRIES, 3)

    def test_vllm_prompt_is_tokenized_once_and_passed_as_token_ids(self):
        tokenizer = _RecordingTokenizer()
        judge = object.__new__(VLLMJudge)
        judge.model_name = "Qwen/Qwen3-32B"
        judge.family = _family_for(judge.model_name)
        judge.sampling_params_cls = _SamplingParams
        judge.max_model_len = 10
        judge.tokenizer = tokenizer
        judge.llm = _RecordingLLM()

        outputs = judge.generate(
            [[{"role": "user", "content": "judge this"}]],
            desc="test",
        )

        prompts, params = judge.llm.calls[0]
        self.assertEqual(prompts, [{"prompt_token_ids": ["formatted"]}])
        self.assertEqual(tokenizer.encode_calls, 1)
        self.assertEqual(params[0].max_tokens, 9)
        self.assertEqual(outputs[0].generated_tokens, 2)

        # Repetitions and retries reuse the exact rendered token IDs.
        judge.generate(
            [[{"role": "user", "content": "judge this"}]],
            desc="test again",
            seed_offset=1,
        )
        self.assertEqual(tokenizer.encode_calls, 1)
        self.assertEqual(len(tokenizer.calls), 1)

        outputs = judge.generate(
            [
                [{"role": "user", "content": "judge this"}],
                [{"role": "user", "content": "judge this"}],
            ],
            desc="packed repetitions",
            seed_offsets=[1, 2],
        )
        _prompts, params = judge.llm.calls[-1]
        self.assertEqual([param.seed for param in params], [1, 2])
        self.assertEqual([output.seed for output in outputs], [1, 2])
        self.assertEqual(tokenizer.encode_calls, 1)

    def test_vllm_output_count_mismatch_raises_for_outer_isolation(self):
        judge = object.__new__(VLLMJudge)
        judge.model_name = "Qwen/Qwen3-32B"
        judge.family = _family_for(judge.model_name)
        judge.sampling_params_cls = _SamplingParams
        judge.max_model_len = 10
        judge.tokenizer = _RecordingTokenizer()
        judge.llm = _ShortVLLM()

        with self.assertRaisesRegex(
            RuntimeError, "vLLM returned 0 outputs for 1 prompts"
        ):
            judge.generate(
                [[{"role": "user", "content": "judge this"}]], desc="test"
            )

    def test_prompt_that_exceeds_context_never_reaches_the_engine(self):
        judge = object.__new__(VLLMJudge)
        judge.model_name = "Qwen/Qwen3-32B"
        judge.family = _family_for(judge.model_name)
        judge.sampling_params_cls = _SamplingParams
        judge.max_model_len = 4
        judge.tokenizer = _EchoTokenizer()
        judge.llm = _RecordingLLM()

        outputs = judge.generate(
            [
                [{"role": "user", "content": "one two"}],
                [{"role": "user", "content": "one two three four five"}],
            ],
            desc="test",
            seed_offset=2,
        )

        self.assertEqual(outputs[0].finish_reason, "stop")
        self.assertEqual(outputs[0].max_tokens, 2)
        exhausted = outputs[1]
        self.assertEqual(exhausted.finish_reason, "context_exhausted")
        self.assertEqual(exhausted.text, "")
        self.assertEqual(exhausted.max_tokens, -1)
        self.assertEqual(exhausted.seed, 2)
        self.assertIn("5 tokens", exhausted.error)
        self.assertIn("4-token context", exhausted.error)
        # Only the prompt that leaves room for an answer is submitted.
        prompts, _params = judge.llm.calls[0]
        self.assertEqual(prompts, [{"prompt_token_ids": ["one", "two"]}])

    def test_engine_returning_no_completion_is_recorded_as_empty(self):
        judge = object.__new__(VLLMJudge)
        judge.model_name = "Qwen/Qwen3-32B"
        judge.family = _family_for(judge.model_name)
        judge.sampling_params_cls = _SamplingParams
        judge.max_model_len = 10
        judge.tokenizer = _RecordingTokenizer()
        judge.llm = _NoCompletionLLM()

        outputs = judge.generate(
            [[{"role": "user", "content": "judge this"}]], desc="test"
        )

        self.assertEqual(outputs[0].finish_reason, "empty")
        self.assertEqual(outputs[0].text, "")
        self.assertEqual(outputs[0].generated_tokens, -1)
        self.assertEqual(outputs[0].max_tokens, 9)

    def test_engine_failure_is_isolated_to_the_offending_sample(self):
        pipe = _PoisonedJudge("poison")

        outputs = _run_llm_judge_local(
            ["ok-0", "ok-1", "poison", "ok-3"],
            "system",
            "Qwen/Qwen3-32B",
            pipe=pipe,
            seed_offset=4,
        )

        self.assertEqual(
            [o.text for o in outputs],
            ["good output", "good output", "", "good output"],
        )
        self.assertEqual(outputs[2].finish_reason, "error")
        self.assertIn("RuntimeError: CUDA out of memory", outputs[2].error)
        self.assertEqual(outputs[2].seed, 4)
        # A failed batch is halved until the offending input stands alone.
        self.assertEqual(pipe.batch_sizes, [4, 2, 2, 1, 1])

    def test_local_output_count_mismatch_is_isolated_per_sample(self):
        outputs = _run_llm_judge_local(
            ["good", "bad"],
            "system",
            "Qwen/Qwen3-32B",
            pipe=_ShortJudge(),
        )

        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].text, "good output")
        self.assertEqual(outputs[1].finish_reason, "error")
        self.assertIn("0 outputs for 1 inputs", outputs[1].error)

    def test_empty_local_input_does_not_build_an_engine(self):
        with patch("quality.metrics.llm_judge.build_llm_judge_pipe") as build:
            stats = evaluate_llm_judge(
                [{"ref": "reference", "gen": "   "}],
                QualityConfig(judge_model="Qwen/Qwen3-32B"),
                rubric=PAIRWISE_RUBRIC,
            )

        build.assert_not_called()
        self.assertEqual(stats["llm_judge_per_sample"].tolist(), [0.0])
        self.assertEqual(stats["llm_judge_input_tokens_per_sample"].tolist(), [0])
        self.assertEqual(stats["llm_judge_valid_repetitions_per_sample"].tolist(), [0])
        self.assertEqual(
            stats["llm_judge_accepted_attempt_per_sample"].tolist(),
            [[-1, -1, -1]],
        )

    def test_owned_engine_is_released_when_input_preparation_fails(self):
        pipe = SimpleNamespace(tokenizer=_BrokenTokenizer())
        with (
            patch(
                "quality.metrics.llm_judge.build_llm_judge_pipe",
                return_value=pipe,
            ),
            patch("quality.metrics.llm_judge.gc.collect") as collect,
            patch("quality.metrics.llm_judge.torch.cuda.empty_cache") as empty_cache,
        ):
            with self.assertRaisesRegex(RuntimeError, "tokenizer failed"):
                evaluate_llm_judge(
                    [{"ref": "reference", "gen": "candidate"}],
                    QualityConfig(judge_model="Qwen/Qwen3-32B"),
                    rubric=PAIRWISE_RUBRIC,
                )

        collect.assert_called_once_with()
        empty_cache.assert_called_once_with()

    def test_truncation_changes_only_generated_input_text(self):
        tokenizer = _RecordingTokenizer()
        sample = {
            "prompt": "prompt stays whole",
            "ref": "reference stays whole",
            "gen": "one two three four five",
        }
        truncated, kept, removed = _truncate_input_text(sample, tokenizer, 3)
        self.assertEqual(truncated["prompt"], sample["prompt"])
        self.assertEqual(truncated["ref"], sample["ref"])
        self.assertEqual(truncated["gen"], "one two three")
        self.assertEqual((kept, removed), (3, 2))
        self.assertEqual(sample["gen"], "one two three four five")

    def test_openai_path_does_not_tokenize_or_truncate_input(self):
        valid = (
            '<json>{"content_recall": 5, "detail_precision": 5, '
            '"information_injection": 0, "contradiction": 0}</json>'
        )
        seen = []

        def fake_openai(user_contents, *args, **kwargs):
            _ = args, kwargs
            seen.extend(user_contents)
            return [valid] * len(user_contents)

        gen = " ".join(f"token{i}" for i in range(9000))
        with patch(
            "quality.metrics.llm_judge._run_llm_judge_openai",
            side_effect=fake_openai,
        ):
            stats = evaluate_llm_judge(
                [{"ref": "reference", "gen": gen}],
                QualityConfig(judge_model="gpt-5.6", judge_repeats=3),
                rubric=PAIRWISE_RUBRIC,
            )

        self.assertEqual(len(seen), 3)
        self.assertIn("token8999", seen[0])
        self.assertEqual(stats["llm_judge_input_tokens_per_sample"].tolist(), [-1])
        self.assertEqual(stats["llm_judge_truncated_tokens_per_sample"].tolist(), [0])

    def test_judge_repeats_default_to_three_and_must_be_positive(self):
        self.assertEqual(QualityConfig().judge_repeats, 3)
        with self.assertRaisesRegex(ValueError, "judge_repeats must be positive"):
            QualityConfig(judge_repeats=0)

    def test_disabled_judge_keeps_configured_repeat_shape(self):
        stats = evaluate_llm_judge(
            [{"ref": "reference", "gen": "candidate"}],
            QualityConfig(judge_repeats=5),
            rubric=PAIRWISE_RUBRIC,
        )
        self.assertEqual(
            stats["llm_judge_content_recall_repeats_per_sample"].shape,
            (1, 5),
        )

    def test_qwen_parser_uses_only_final_answer(self):
        raw = (
            '<think>Provisional score: <json>{"score": 1}</json></think>'
            '<json>{"score": 5}</json>'
        )
        final = _final_answer_text(raw, "Qwen/Qwen3-32B")
        self.assertEqual(_parse_dimension_scores(final, ("score",)), {"score": 5.0})

    def test_qwen36_parser_uses_only_final_answer(self):
        raw = "analysis with score: 0</think>\n\n<json>{\"score\": 4}</json>"
        self.assertEqual(
            _final_answer_text(raw, "Qwen/Qwen3.6-27B"),
            '<json>{"score": 4}</json>',
        )

    def test_evaluator_cannot_score_provisional_trace_json(self):
        bad = (
            '{"content_recall": 0, "detail_precision": 0, '
            '"information_injection": 5, "contradiction": 5}'
        )
        good = (
            '{"content_recall": 5, "detail_precision": 5, '
            '"information_injection": 0, "contradiction": 0}'
        )
        raw = f"<think><json>{bad}</json></think><json>{good}</json>"
        stats = evaluate_llm_judge(
            [{"ref": "same fact", "gen": "same fact"}],
            QualityConfig(judge_model="Qwen/Qwen3-32B", judge_batch_size=1),
            rubric=PAIRWISE_RUBRIC,
            pipe=_FixedJudge(raw),
        )
        self.assertEqual(stats["llm_judge"], 1.0)

    def test_repetitions_use_independent_seeds_and_per_criterion_means(self):
        outputs = [
            (
                '<think>first</think><json>{"content_recall": 5, '
                '"detail_precision": 1, "information_injection": 0, '
                '"contradiction": 0}</json>'
            ),
            (
                '<think>second</think><json>{"content_recall": 1, '
                '"detail_precision": 5, "information_injection": 1, '
                '"contradiction": 0}</json>'
            ),
            (
                '<think>third</think><json>{"content_recall": 3, '
                '"detail_precision": 3, "information_injection": 5, '
                '"contradiction": 5}</json>'
            ),
        ]
        pipe = _RepeatedJudge(outputs)
        stats = evaluate_llm_judge(
            [{"ref": "reference", "gen": "candidate"}],
            QualityConfig(judge_model="Qwen/Qwen3-32B", judge_repeats=3),
            rubric=PAIRWISE_RUBRIC,
            pipe=pipe,
            debug=True,
        )

        self.assertEqual(pipe.seed_offsets, [0, 1, 2])
        self.assertEqual(pipe.batch_sizes, [3])
        expected_dims = {
            "content_recall": 3.0,
            "detail_precision": 3.0,
            "information_injection": 2.0,
            "contradiction": 5.0 / 3.0,
        }
        for key, value in expected_dims.items():
            self.assertEqual(stats[f"llm_judge_{key}_per_sample"][0], value / 5.0)
        self.assertEqual(
            stats["llm_judge_content_recall_repeats_per_sample"].tolist(),
            [[1.0, 0.2, 0.6]],
        )
        self.assertEqual(len(stats["raw_outputs"][0]), 3)
        self.assertAlmostEqual(
            stats["llm_judge"],
            sum(
                PAIRWISE_RUBRIC.weights[key]
                * (
                    value
                    if PAIRWISE_RUBRIC.valence.get(key, 1) > 0
                    else 5.0 - value
                )
                for key, value in expected_dims.items()
            )
            / 5.0,
        )

    def test_asymmetric_quality_runs_use_arithmetic_means(self):
        outputs = [
            ('<think>first</think><json>{"fluency": 5, "coherence": 5, "relevance": 5, '
             '"informativeness": 5}</json>'),
            ('<think>second</think><json>{"fluency": 1, "coherence": 1, "relevance": 1, '
             '"informativeness": 1}</json>'),
            ('<think>third</think><json>{"fluency": 1, "coherence": 1, "relevance": 1, '
             '"informativeness": 4}</json>'),
        ]
        stats = evaluate_llm_judge(
            [{"prompt": "prompt", "gen": "candidate"}],
            QualityConfig(judge_model="Qwen/Qwen3-32B", judge_repeats=3),
            rubric=QUALITY_RUBRIC,
            pipe=_RepeatedJudge(outputs),
        )

        expected_dims = {
            "fluency": 7.0 / 3.0,
            "coherence": 7.0 / 3.0,
            "relevance": 7.0 / 3.0,
            "informativeness": 10.0 / 3.0,
        }
        for key, value in expected_dims.items():
            self.assertAlmostEqual(
                stats[f"llm_quality_{key}_per_sample"][0], value / 5.0
            )
        expected = sum(
            QUALITY_RUBRIC.weights[key] * value
            for key, value in expected_dims.items()
        ) / 5.0
        self.assertAlmostEqual(stats["llm_quality"], expected)
        self.assertNotAlmostEqual(stats["llm_quality"], 1.75 / 5.0)

    def test_invalid_repetition_retries_original_prompt_with_fresh_seed(self):
        valid = (
            '<think>done</think><json>{"content_recall": 5, '
            '"detail_precision": 5, "information_injection": 0, '
            '"contradiction": 0}</json>'
        )
        pipe = _SeedJudge({0: "<think>runaway", 3: valid, 1: valid, 2: valid})
        stats = evaluate_llm_judge(
            [{"ref": "reference", "gen": "candidate"}],
            QualityConfig(judge_model="Qwen/Qwen3-32B", judge_repeats=3),
            rubric=PAIRWISE_RUBRIC,
            pipe=pipe,
        )

        self.assertEqual([seed for seed, _ in pipe.calls], [0, 1, 2, 3])
        first_messages = pipe.calls[0][1]
        retry_messages = pipe.calls[3][1]
        self.assertEqual(retry_messages, first_messages)
        self.assertEqual(len(retry_messages[0]), 2)
        self.assertEqual(stats["llm_judge"], 1.0)
        self.assertEqual(
            stats["llm_judge_accepted_attempt_per_sample"].tolist(),
            [[1, 0, 0]],
        )

    def test_missing_one_repetition_makes_entire_sample_nan(self):
        valid = (
            '<think>done</think><json>{"content_recall": 5, '
            '"detail_precision": 5, "information_injection": 0, '
            '"contradiction": 0}</json>'
        )
        # Logical repetition 1 uses seeds 1, 4, 7, and 10 and exhausts all
        # three additional attempts. The other repetitions succeed.
        pipe = _SeedJudge({0: valid, 2: valid})
        stats = evaluate_llm_judge(
            [{"ref": "reference", "gen": "candidate"}],
            QualityConfig(judge_model="Qwen/Qwen3-32B", judge_repeats=3),
            rubric=PAIRWISE_RUBRIC,
            pipe=pipe,
        )

        self.assertTrue(np.isnan(stats["llm_judge_per_sample"][0]))
        self.assertTrue(np.isnan(stats["llm_judge_content_recall_per_sample"][0]))
        self.assertEqual(
            stats["llm_judge_valid_repetitions_per_sample"].tolist(),
            [2],
        )
        self.assertEqual([seed for seed, _ in pipe.calls], [0, 1, 2, 4, 7, 10])

    def test_local_errors_are_persisted_in_attempt_telemetry(self):
        stats = evaluate_llm_judge(
            [{"ref": "reference", "gen": "poison"}],
            QualityConfig(judge_model="Qwen/Qwen3-32B", judge_repeats=1),
            rubric=PAIRWISE_RUBRIC,
            pipe=_PoisonedJudge("poison"),
        )

        errors = stats["llm_judge_attempt_error_per_sample"]
        self.assertEqual(errors.shape, (1, 1, _JUDGE_MAX_RETRIES + 1))
        self.assertTrue(
            all("CUDA out of memory" in error for error in errors[0, 0])
        )

    def test_unclosed_reasoning_is_unparseable(self):
        raw = '<think><json>{"score": 5}</json>'
        final = _final_answer_text(raw, "Qwen/Qwen3-32B")
        self.assertEqual(final, "")
        self.assertEqual(_parse_dimension_scores(final, ("score",)), {})

    def test_hidden_reasoning_api_content_is_unchanged(self):
        raw = '<json>{"score": 3}</json>'
        self.assertEqual(_final_answer_text(raw, "gpt-5.6"), raw)

    def test_thinking_is_explicitly_enabled_for_both_qwen_families(self):
        for model in ("Qwen/Qwen3-32B", "Qwen/Qwen3.6-27B"):
            with self.subTest(model=model):
                tokenizer = _RecordingTokenizer()
                _format_chat_for_generation(
                    tokenizer, [[{"role": "user", "content": "score this"}]], model
                )
                self.assertIs(tokenizer.calls[-1]["enable_thinking"], True)

    def test_vllm_features_are_family_specific(self):
        self.assertEqual(
            _family_for("Qwen/Qwen3-32B").vllm_kwargs,
            {"reasoning_parser": "qwen3"},
        )
        self.assertEqual(
            _family_for("Qwen/Qwen3.6-27B").vllm_kwargs,
            {"reasoning_parser": "qwen3", "language_model_only": True},
        )
        self.assertEqual(
            _family_for("meta-llama/Llama-3.1-8B-Instruct").vllm_kwargs,
            {},
        )


if __name__ == "__main__":
    unittest.main()
