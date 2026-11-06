"""Efficient Item-Based KNN collaborative filtering for recipe recommendations.

Candidate generation is neighbor-driven: retrieve neighbors of only the recipes
rated by a user, then aggregate scores for those candidates. This avoids a
nearest-neighbor query for every recipe in the catalog.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from fridge_rescue.recommenders.content_based import RECOMMENDATION_COLUMNS


class ItemKNNRecommender:
    """Predict user preferences using item-item cosine similarity."""

    def __init__(
        self,
        train_interactions: pd.DataFrame,
        recipes: pd.DataFrame,
        *,
        n_neighbors: int = 50,
        min_similarity: float = 0.0,
    ) -> None:
        required = {"user_id", "recipe_id", "rating"}
        missing = required - set(train_interactions.columns)
        if missing:
            raise ValueError(f"Missing interaction columns: {sorted(missing)}")
        if "recipe_id" not in recipes.columns:
            raise ValueError("Recipe data must contain recipe_id")
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be positive")
        if not 0.0 <= min_similarity < 1.0:
            raise ValueError("min_similarity must be in [0, 1)")

        self.recipes = recipes.drop_duplicates("recipe_id").copy()
        self.n_neighbors = int(n_neighbors)
        self.min_similarity = float(min_similarity)

        interactions = train_interactions[["user_id", "recipe_id", "rating"]].dropna().copy()
        interactions["rating"] = pd.to_numeric(interactions["rating"], errors="coerce")
        interactions = interactions.dropna(subset=["rating"])
        interactions = interactions[interactions["recipe_id"].isin(self.recipes["recipe_id"])]
        if interactions.empty:
            raise ValueError("No valid training interactions were provided")

        self.users = pd.Index(interactions["user_id"].unique())
        self.recipe_ids = pd.Index(self.recipes["recipe_id"].unique())
        self.user_to_index = {user: i for i, user in enumerate(self.users)}
        self.recipe_to_index = {recipe: i for i, recipe in enumerate(self.recipe_ids)}

        interactions["_user_idx"] = interactions["user_id"].map(self.user_to_index)
        interactions["_recipe_idx"] = interactions["recipe_id"].map(self.recipe_to_index)
        self.user_means = interactions.groupby("user_id")["rating"].mean()
        interactions["_centered"] = interactions["rating"] - interactions["user_id"].map(
            self.user_means
        )

        self.user_item_matrix = csr_matrix(
            (
                interactions["_centered"].to_numpy(dtype=np.float32),
                (interactions["_user_idx"].to_numpy(), interactions["_recipe_idx"].to_numpy()),
            ),
            shape=(len(self.users), len(self.recipe_ids)),
            dtype=np.float32,
        )
        self.item_user_matrix = self.user_item_matrix.T.tocsr()

        # Binary interaction matrix: 1 means the user rated the recipe.
        # This lets us distinguish reliable similarities from accidental
        # cosine=1.0 matches based on only one common user.
        self.item_user_binary = self.item_user_matrix.copy()
        self.item_user_binary.data = np.ones(self.item_user_binary.data.shape, dtype=np.float32)

        self.min_common_users = 2
        self.similarity_shrinkage = 10.0
        self.neighbor_model = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1)
        self.neighbor_model.fit(self.item_user_matrix)

        self.user_rated_items = {
            i: set(self.user_item_matrix[i].indices.tolist()) for i in range(len(self.users))
        }
        self.recipe_metadata = self.recipes.set_index("recipe_id")

    def _neighbors_for_item(self, item_idx: int):
        # Retrieve a wider pool because candidates with too few co-raters
        # are discarded after the nearest-neighbor search.
        pool_size = min(max(self.n_neighbors * 5, self.n_neighbors + 1), len(self.recipe_ids))
        distances, indices = self.neighbor_model.kneighbors(
            self.item_user_matrix[item_idx], n_neighbors=pool_size
        )

        candidate_rows = self.item_user_binary[indices[0]]
        target_row = self.item_user_binary[item_idx]
        common_users = (candidate_rows @ target_row.T).toarray().ravel()

        neighbors = []
        for distance, neighbor_idx, overlap in zip(
            distances[0], indices[0], common_users, strict=True
        ):
            if neighbor_idx == item_idx or overlap < self.min_common_users:
                continue

            cosine_similarity = 1.0 - float(distance)
            # Shrink similarities supported by few common raters.
            reliability = float(overlap) / (float(overlap) + self.similarity_shrinkage)
            adjusted_similarity = cosine_similarity * reliability

            if adjusted_similarity > self.min_similarity:
                neighbors.append((int(neighbor_idx), adjusted_similarity))

            if len(neighbors) >= self.n_neighbors:
                break

        return neighbors

    def predict_preference(self, user_id, recipe_id) -> float | None:
        if user_id not in self.user_to_index or recipe_id not in self.recipe_to_index:
            return None
        user_idx = self.user_to_index[user_id]
        target_idx = self.recipe_to_index[recipe_id]
        rated_items = self.user_rated_items[user_idx]
        if target_idx in rated_items:
            return None

        numerator = denominator = 0.0
        for rated_idx in rated_items:
            centered_rating = float(self.user_item_matrix[user_idx, rated_idx])
            for neighbor_idx, similarity in self._neighbors_for_item(rated_idx):
                if neighbor_idx == target_idx:
                    numerator += similarity * centered_rating
                    denominator += abs(similarity)
        if denominator == 0.0:
            return float(self.user_means.loc[user_id])
        prediction = float(self.user_means.loc[user_id]) + numerator / denominator
        return float(np.clip(prediction, 1.0, 5.0))

    def recommend(
        self, user_id, *, top_k: int = 10, max_minutes: float | None = None
    ) -> pd.DataFrame:
        if user_id not in self.user_to_index:
            raise ValueError(f"Unknown user_id: {user_id}")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        user_idx = self.user_to_index[user_id]
        rated_items = self.user_rated_items[user_idx]
        empty_columns = RECOMMENDATION_COLUMNS + ["predicted_preference"]
        if not rated_items:
            return pd.DataFrame(columns=empty_columns)

        numerators: dict[int, float] = {}
        denominators: dict[int, float] = {}
        for rated_idx in rated_items:
            centered_rating = float(self.user_item_matrix[user_idx, rated_idx])
            for candidate_idx, similarity in self._neighbors_for_item(rated_idx):
                if candidate_idx in rated_items:
                    continue
                numerators[candidate_idx] = (
                    numerators.get(candidate_idx, 0.0) + similarity * centered_rating
                )
                denominators[candidate_idx] = denominators.get(candidate_idx, 0.0) + abs(similarity)

        user_mean = float(self.user_means.loc[user_id])
        candidates = []
        for recipe_idx, numerator in numerators.items():
            recipe_id = self.recipe_ids[recipe_idx]
            metadata = self.recipe_metadata.loc[recipe_id]
            minutes = metadata.get("minutes")
            if minutes is not None and pd.notna(minutes):
                minutes = float(minutes)
                if max_minutes is not None and minutes > max_minutes:
                    continue
            else:
                minutes = None
            denominator = denominators[recipe_idx]
            prediction = user_mean if denominator == 0.0 else user_mean + numerator / denominator
            candidates.append(
                {
                    "recipe_id": recipe_id,
                    "title": metadata.get("title", str(recipe_id)),
                    "predicted_preference": float(np.clip(prediction, 1.0, 5.0)),
                    "content_score": 0.0,
                    "ingredient_coverage": 0.0,
                    "matched_ingredients": [],
                    "missing_ingredients": [],
                    "missing_count": 0,
                    "minutes": minutes,
                    "source": metadata.get("source", "Food.com"),
                }
            )

        if not candidates:
            return pd.DataFrame(columns=empty_columns)
        results = pd.DataFrame(candidates)
        return (
            results.sort_values(
                ["predicted_preference", "recipe_id"], ascending=[False, True], kind="stable"
            )
            .head(top_k)
            .reset_index(drop=True)
        )
