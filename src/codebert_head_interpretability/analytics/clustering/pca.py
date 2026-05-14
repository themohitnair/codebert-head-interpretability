import numpy as np

from sklearn.decomposition import PCA

from codebert_head_interpretability.schemas.clustering import (
    HeadFeatureVector,
)


class PCAEmbedder:
    def fit_transform(
        self,
        vectors: list[HeadFeatureVector],
    ):
        X = np.array([v.vector for v in vectors])

        pca = PCA(n_components=2)

        embeddings = pca.fit_transform(X)

        return embeddings, pca
