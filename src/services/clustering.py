from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from src.data.classes.blocks import ClusterIndex
from src.data.classes.settings import ClusteringSettings

logger = logging.getLogger(__name__)

NORM_EPSILON = 1.0e-12


def normalize_rows(*, matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, NORM_EPSILON)


class BlockClusterer:
    def __init__(
        self,
        *,
        settings: ClusteringSettings,
        seed: int,
    ) -> None:
        self._settings = settings
        self._seed = seed

    def cluster(
        self,
        *,
        fingerprints: np.ndarray,
        context_lengths: Sequence[int],
    ) -> ClusterIndex:
        from sklearn.cluster import KMeans

        requested = self._settings.num_clusters or len(set(context_lengths))
        num_clusters = min(requested, len(fingerprints))
        features = normalize_rows(matrix=fingerprints) if self._settings.normalize else fingerprints
        kmeans = KMeans(
            n_clusters=num_clusters,
            n_init=self._settings.initializations,
            random_state=self._seed,
        ).fit(features)
        labels = kmeans.labels_.astype(int)
        representatives = []
        for cluster in range(num_clusters):
            members = np.flatnonzero(labels == cluster)
            distances = np.linalg.norm(features[members] - kmeans.cluster_centers_[cluster], axis=1)
            representatives.append(int(members[np.argmin(distances)]))
        index = ClusterIndex(
            labels=tuple(int(label) for label in labels),
            representatives=tuple(representatives),
            context_lengths=tuple(int(context_lengths[member]) for member in representatives),
        )
        for cluster in range(num_clusters):
            members = index.members(cluster=cluster)
            purity = np.mean([context_lengths[member] == index.context_lengths[cluster] for member in members])
            logger.info(
                "Cluster %d: %d blocks, context length %d, context purity %.2f",
                cluster,
                len(members),
                index.context_lengths[cluster],
                purity,
            )
        return index
