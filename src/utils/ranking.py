from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

SIGNIFICANCE_LEVEL = 0.05


@dataclass(frozen=True, slots=True)
class PairwiseTest:
    first: str
    second: str
    p_value: float
    significant: bool


@dataclass(frozen=True, slots=True)
class RankingResult:
    models: tuple[str, ...]
    average_ranks: tuple[float, ...]
    friedman_p_value: float
    pairwise: tuple[PairwiseTest, ...]
    cliques: tuple[tuple[str, ...], ...]


def average_ranks(*, scores: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    ranks = np.vstack([rankdata(-row, method="average") for row in scores])
    return ranks.mean(axis=0)


def holm_correction(
    *,
    p_values: Sequence[float],
    alpha: float,
) -> list[bool]:
    order = np.argsort(p_values)
    significant = [False] * len(p_values)
    total = len(p_values)
    for position, index in enumerate(order):
        if p_values[index] > alpha / (total - position):
            break
        significant[index] = True
    return significant


def find_cliques(
    *,
    models: Sequence[str],
    ranks: Sequence[float],
    pairwise: Sequence[PairwiseTest],
) -> list[tuple[str, ...]]:
    order = list(np.argsort(ranks))
    not_different = {frozenset((test.first, test.second)) for test in pairwise if not test.significant}
    cliques: list[tuple[str, ...]] = []
    for start in range(len(order)):
        members = [models[order[start]]]
        for end in range(start + 1, len(order)):
            candidate = models[order[end]]
            if all(frozenset((member, candidate)) in not_different for member in members):
                members.append(candidate)
            else:
                break
        if len(members) > 1 and not any(set(members) <= set(existing) for existing in cliques):
            cliques.append(tuple(members))
    return cliques


def rank_models(
    *,
    models: Sequence[str],
    scores: np.ndarray,
    alpha: float = SIGNIFICANCE_LEVEL,
) -> RankingResult:
    from scipy.stats import friedmanchisquare, wilcoxon

    ranks = average_ranks(scores=scores)
    friedman = friedmanchisquare(*[scores[:, column] for column in range(scores.shape[1])])
    pairs = [(first, second) for first in range(len(models)) for second in range(first + 1, len(models))]
    p_values = []
    for first, second in pairs:
        difference = scores[:, first] - scores[:, second]
        p_values.append(1.0 if np.allclose(difference, 0) else float(wilcoxon(difference, zero_method="pratt").pvalue))
    significant = holm_correction(
        p_values=p_values,
        alpha=alpha,
    )
    pairwise = tuple(
        PairwiseTest(
            first=models[first],
            second=models[second],
            p_value=p_value,
            significant=flag,
        )
        for (first, second), p_value, flag in zip(pairs, p_values, significant)
    )
    cliques = find_cliques(
        models=models,
        ranks=list(ranks),
        pairwise=pairwise,
    )
    return RankingResult(
        models=tuple(models),
        average_ranks=tuple(float(rank) for rank in ranks),
        friedman_p_value=float(friedman.pvalue),
        pairwise=pairwise,
        cliques=tuple(cliques),
    )
