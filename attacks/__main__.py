"""Run a configured watermark-removal attack."""

import argparse
import os

from datasets import load_from_disk
from transformers import AutoTokenizer

from config.cli import add_config_args, resolve
from config.loader import dump_config
from config.paths import attack_dir, watermark_dir
from config.runtime import vllm_gpu_memory_utilization
from attacks.base import AttackConfig
from attacks.simple.structural import CHUNK_ATOMS, PROBE_ATOMS
from segmentation import DEFAULT_BACKEND


ATTACK_MODULES = {
    "parrot": "attacks.sentence_level.parrot",
    "parrot-bigram": "attacks.sentence_level.parrot",
    "openai": "attacks.sentence_level.openai",
    "openai-bigram": "attacks.sentence_level.openai",
    "pegasus": "attacks.sentence_level.pegasus",
    "pegasus-bigram": "attacks.sentence_level.pegasus",
    "custom": "attacks.paraphrasing.custom",
    "custom_sent": "attacks.paraphrasing.custom",
    "adaptive": "attacks.paraphrasing.adaptive",
    "oracle": "attacks.paraphrasing.oracle",
    "dipper": "attacks.paraphrasing.dipper",
    "word_deletion": "attacks.simple.word_edit",
    "synonym_substitution": "attacks.simple.word_edit",
    "context_synonym": "attacks.simple.word_edit",
    "back_translation": "attacks.simple.translation",
    **{atom: "attacks.simple.structural" for atom in PROBE_ATOMS},
}

ATTACK_RUNNERS = {
    "custom_sent": "run_sentence_attack",
    "word_deletion": "run_word_deletion_attack",
    "synonym_substitution": "run_synonym_substitution_attack",
    "context_synonym": "run_context_synonym_attack",
    **{atom: f"run_{atom}_attack" for atom in PROBE_ATOMS},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Attack a watermarked dataset.")
    add_config_args(parser)
    return resolve(parser.parse_args())


def require_custom_model(cfg):
    if cfg.attack.custom_model is None:
        raise ValueError(
            f"attack.custom_model is required for attack {cfg.attack.paraphraser!r}"
        )


def load_bigram_tokenizer(cfg):
    # Bigram selection needs the shared tokenizer.
    if not cfg.attack.paraphraser.endswith("-bigram"):
        return None
    return AutoTokenizer.from_pretrained(cfg.attack.model_path)


def build_config(cfg, save_dir):
    a = cfg.attack
    # Only adaptive attacks may override sentence segmentation.
    if a.paraphraser == "adaptive":
        attacker_seg_type = cfg.segmentation.attacker_type or "sentence"
        attacker_seg_backend = cfg.segmentation.attacker_backend or DEFAULT_BACKEND
    elif a.paraphraser == "oracle":
        # The oracle attacker knows the detector, segmentation included.
        attacker_seg_type = cfg.segmentation.type
        attacker_seg_backend = cfg.segmentation.backend
    else:
        attacker_seg_type = "sentence"
        attacker_seg_backend = cfg.segmentation.backend
    return AttackConfig(
        model_path=a.model_path,
        custom_model=a.custom_model,
        prompt_style=a.prompt_style,
        batch_size=a.batch_size if a.batch_size is not None else 1,
        num_beams=a.num_beams,
        bigram=a.paraphraser.endswith("-bigram"),
        bert_threshold=a.bert_threshold,
        device="cuda",
        temperature=a.temperature,
        do_sample=a.do_sample,
        surrogate_model=a.surrogate_model,
        num_candidates=a.num_candidates,
        backend=a.backend,
        vllm_utilization=vllm_gpu_memory_utilization(cfg.runtime.vllm_utilization),
        anchor=a.anchor,
        bag_agg=a.bag_agg,
        watermark=cfg.watermark,
        segmentation_type=attacker_seg_type,
        segmentation_backend=attacker_seg_backend,
        semcut_max_words=cfg.segmentation.semcut_max_words,
        semcut_window=cfg.segmentation.semcut_window,
        semcut_batch_size=cfg.runtime.semcut_batch_size,
        output_path=save_dir,
        dipper_lex=a.dipper_lex,
        dipper_order=a.dipper_order,
        dipper_sent_interval=a.dipper_sent_interval,
        word_edit_ratio=a.word_edit_ratio,
        back_translation_lang=a.back_translation_lang,
        rechunk_words=a.rechunk_words,
        donor_corpus=a.donor_corpus,
    )


def load_attack_backend(name):
    from importlib import import_module

    return import_module(ATTACK_MODULES[name])


def run_attack(cfg):
    attack_name = cfg.attack.paraphraser
    if attack_name not in ATTACK_MODULES:
        raise ValueError(
            f"Unknown attack {attack_name!r}; choose from {sorted(ATTACK_MODULES)}"
        )
    if attack_name in {"custom", "custom_sent", "adaptive", "oracle"}:
        require_custom_model(cfg)

    # Read watermarked text and write the derived attack output.
    save_dir = cfg.io.output_dir or attack_dir(cfg)
    config = build_config(cfg, save_dir)
    tokenizer = load_bigram_tokenizer(cfg)

    wm_dir = watermark_dir(cfg)
    src_dir = wm_dir if os.path.isdir(wm_dir) else cfg.io.data_path
    texts = load_from_disk(src_dir)["text"]

    attack = load_attack_backend(attack_name)
    kwargs = {"tokenizer": tokenizer} if attack_name.endswith("-bigram") else {}
    runner_name = ATTACK_RUNNERS.get(attack_name, "run_attack")
    result = getattr(attack, runner_name)(texts, config, **kwargs)
    if os.path.isdir(save_dir):
        dump_config(cfg, os.path.join(save_dir, "resolved_config.yaml"))
    return result


def main():
    run_attack(parse_args())


if __name__ == "__main__":
    main()
