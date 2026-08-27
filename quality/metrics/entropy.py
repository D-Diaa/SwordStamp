"""Token entropy adapted from original SemStamp's ``eval_quality.py``."""

from .common import text_entropy, tokenize_untokenize


def evaluate_entropy(gens, tokenizer):
    gen_sen_lis = [tokenize_untokenize(g, tokenizer) for g in gens]
    gen_entros = []
    for k in [2, 3]:
        gen_entro = text_entropy(gen_sen_lis, k=k)
        gen_entros.append(gen_entro)
    return gen_entros
