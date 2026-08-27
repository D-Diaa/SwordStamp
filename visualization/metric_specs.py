"""Authoritative metadata for every base metric emitted by ``quality``.

Aggregate CSV fields append ``_ci`` and ``_median`` to these base keys. Metrics
with ``per_sample=True`` also have an array under the base key in
``eval_quality_per_sample.npz`` and the compiled per-sample table.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    lower_is_better: bool | None
    per_sample: bool
    group: str
    scale: float = 1.0

    @property
    def improvement_sign(self) -> float:
        """Multiplier that makes a raw delta positive when it is beneficial."""
        if self.lower_is_better is None:
            raise ValueError(f"{self.key} has no improvement direction")
        return -1.0 if self.lower_is_better else 1.0


QUALITY_METRICS = (
    # Fluency and lexical diversity.
    MetricSpec("gen_ppl", "perplexity ↓", True, True, "fluency"),
    MetricSpec("rep_2", "2-gram repetition ↓", True, True, "token_diversity"),
    MetricSpec("rep_3", "3-gram repetition ↓", True, True, "token_diversity"),
    MetricSpec("rep_4", "4-gram repetition ↓", True, True, "token_diversity"),
    MetricSpec("bi_entro", "bigram entropy ↑", False, False, "token_diversity"),
    MetricSpec("tri_entro", "trigram entropy ↑", False, False, "token_diversity"),
    MetricSpec("sem_ent", "semantic entropy ↑", False, False, "semantic_diversity"),
    MetricSpec(
        "sent_dup_pct", "sentence duplicate (%) ↓", True, True,
        "deterministic_diversity",
    ),
    MetricSpec("distinct_2", "Distinct-2 ↑", False, True, "deterministic_diversity"),
    MetricSpec(
        "4g_repeat_pct", "4-gram repeat (%) ↓", True, True,
        "deterministic_diversity",
    ),
    # Distributional and reference-based preservation.
    MetricSpec("mauve", "MAUVE vs human ↑", False, False, "distributional"),
    MetricSpec("bleu", "BLEU ↑", False, False, "pairwise"),
    MetricSpec("bert_P", "BERTScore precision ↑", False, True, "pairwise"),
    MetricSpec("bert_R", "BERTScore recall ↑", False, True, "pairwise"),
    MetricSpec("bert_F1", "BERTScore F1 ↑", False, True, "pairwise"),
    MetricSpec("emb_sim", "embedding similarity ↑", False, True, "pairwise"),
    MetricSpec("rouge1", "ROUGE-1 ↑", False, True, "pairwise"),
    MetricSpec("rouge2", "ROUGE-2 ↑", False, True, "pairwise"),
    MetricSpec("rougeL", "ROUGE-L ↑", False, True, "pairwise"),
    # Exact patience-anchor structural channels.
    MetricSpec("anchor_reword", "anchor rewording ↓", True, True, "anchor"),
    MetricSpec("anchor_reword_novel", "novel anchor rewording ↓", True, True, "anchor"),
    MetricSpec("anchor_reorder", "anchor reordering ↓", True, True, "anchor"),
    MetricSpec("anchor_merge", "anchor merging ↓", True, True, "anchor"),
    MetricSpec("anchor_split", "anchor splitting ↓", True, True, "anchor"),
    MetricSpec("anchor_reseg", "anchor resegmentation ↓", True, True, "anchor"),
    MetricSpec("anchor_coverage", "anchor coverage ↑", False, True, "anchor"),
    MetricSpec("anchor_n_anchors", "anchor count", None, True, "anchor"),
    # Intrinsic generation-quality judge. Stored in [0, 1], displayed on [0, 5].
    MetricSpec("llm_quality", "generation quality (judge) ↑", False, True, "llm_quality", 5.0),
    MetricSpec("llm_quality_fluency", "judge fluency ↑", False, True, "llm_quality", 5.0),
    MetricSpec("llm_quality_coherence", "judge coherence ↑", False, True, "llm_quality", 5.0),
    MetricSpec("llm_quality_relevance", "judge relevance ↑", False, True, "llm_quality", 5.0),
    MetricSpec(
        "llm_quality_informativeness", "judge informativeness ↑", False, True,
        "llm_quality", 5.0,
    ),
    # Pairwise preservation judge. Dimension arrays are raw property-presence
    # scores; only the aggregate applies rubric valence to bad properties.
    MetricSpec("llm_judge", "content preservation (judge) ↑", False, True, "llm_judge", 5.0),
    MetricSpec("llm_judge_content_recall", "content recall ↑", False, True, "llm_judge", 5.0),
    MetricSpec("llm_judge_detail_precision", "detail precision ↑", False, True, "llm_judge", 5.0),
    MetricSpec(
        "llm_judge_information_injection", "information injection ↓", True, True,
        "llm_judge", 5.0,
    ),
    MetricSpec("llm_judge_contradiction", "contradiction ↓", True, True, "llm_judge", 5.0),
)

QUALITY_METRICS_BY_KEY = {spec.key: spec for spec in QUALITY_METRICS}
if len(QUALITY_METRICS_BY_KEY) != len(QUALITY_METRICS):
    raise ValueError("duplicate quality metric key")


def metric_spec(key: str) -> MetricSpec:
    """Return metadata for a base quality-metric key."""
    return QUALITY_METRICS_BY_KEY[key]


def select_metrics(*, group: str | None = None,
                   per_sample: bool | None = None) -> tuple[MetricSpec, ...]:
    """Select registered metrics while preserving registry order."""
    return tuple(
        spec for spec in QUALITY_METRICS
        if (group is None or spec.group == group)
        and (per_sample is None or spec.per_sample is per_sample)
    )


def metric_labels(keys) -> list[tuple[str, str]]:
    """Return ``(key, label)`` pairs for a plot/table metric list."""
    specs = map(metric_spec, keys)
    return [(spec.key, spec.label) for spec in specs]


def metric_directions(keys) -> list[tuple[str, str, bool | None]]:
    """Return ``(key, label, lower_is_better)`` rows for a metric list."""
    specs = map(metric_spec, keys)
    return [(spec.key, spec.label, spec.lower_is_better) for spec in specs]


def quality_metric_manifest() -> list[dict]:
    """Return JSON-serializable metadata for every registered metric."""
    return [asdict(spec) for spec in QUALITY_METRICS]


PER_SAMPLE_QUALITY_KEYS = tuple(
    spec.key for spec in select_metrics(per_sample=True)
)
CORPUS_QUALITY_KEYS = tuple(
    spec.key for spec in select_metrics(per_sample=False)
)

DETERMINISTIC_DIVERSITY_METRICS = select_metrics(
    group="deterministic_diversity"
)
DETERMINISTIC_DIVERSITY_KEYS = tuple(
    spec.key for spec in DETERMINISTIC_DIVERSITY_METRICS
)
LLM_QUALITY_METRICS = select_metrics(group="llm_quality")
LLM_JUDGE_METRICS = select_metrics(group="llm_judge")
