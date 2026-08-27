"""MAUVE scoring adapted from original SemStamp's ``eval_quality.py``."""


def evaluate_mauve(gens, refs):
    import mauve

    result = mauve.compute_mauve(p_text=refs, q_text=gens, device_id=0, max_text_length=512, verbose=False)
    return result.mauve
