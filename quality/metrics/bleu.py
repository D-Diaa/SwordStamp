"""BLEU pairwise metric."""


def evaluate_bleu(gens, refs):
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

    smoothie = SmoothingFunction().method1
    refs_tokenized = [[ref.split()] for ref in refs]
    gens_tokenized = [gen.split() for gen in gens]
    return corpus_bleu(refs_tokenized, gens_tokenized, smoothing_function=smoothie)
