"""Watermark-removal attack package exports."""

from importlib import import_module

_EXPORTS = {
    "run_attack": ("attacks.__main__", "run_attack"),
    "AttackConfig": ("attacks.base", "AttackConfig"),
    "AttackResult": ("attacks.base", "AttackResult"),
    "batched": ("attacks.utils", "batched"),
    "well_formed_sentence": ("attacks.utils", "well_formed_sentence"),
    "split_texts_into_sentences": ("attacks.utils", "split_texts_into_sentences"),
    "join_sentences_by_document": ("attacks.utils", "join_sentences_by_document"),
    "get_bert_scorer": ("attacks.utils", "get_bert_scorer"),
    "accept_by_bigram_overlap": ("attacks.utils", "accept_by_bigram_overlap"),
    "build_custom_prompt": ("attacks.paraphrasing.custom", "build_custom_prompt"),
    "run_custom_attack": ("attacks.paraphrasing.custom", "run_attack"),
    "run_custom_sentence_attack": ("attacks.paraphrasing.custom", "run_sentence_attack"),
    "run_parrot_attack": ("attacks.sentence_level.parrot", "run_attack"),
    "run_pegasus_attack": ("attacks.sentence_level.pegasus", "run_attack"),
    "run_openai_attack": ("attacks.sentence_level.openai", "run_attack"),
    "run_adaptive_attack": ("attacks.paraphrasing.adaptive", "run_attack"),
    "run_oracle_attack": ("attacks.paraphrasing.oracle", "run_attack"),
    "OracleScorer": ("attacks.paraphrasing.oracle", "OracleScorer"),
    "SParrot": ("attacks.sentence_level.parrot", "SParrot"),
    "extract_list": ("attacks.sentence_level.openai", "extract_list"),
    "gen_prompt": ("attacks.sentence_level.openai", "gen_prompt"),
    "gen_bigram_prompt": ("attacks.sentence_level.openai", "gen_bigram_prompt"),
    "query_openai": ("attacks.sentence_level.openai", "query_openai"),
    "query_openai_bigram": ("attacks.sentence_level.openai", "query_openai_bigram"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
