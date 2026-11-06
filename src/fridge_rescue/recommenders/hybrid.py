"""Hybrid recommendation combining ingredient fit and collaborative filtering."""
from __future__ import annotations

import pandas as pd

from fridge_rescue.recommenders.content_based import RECOMMENDATION_COLUMNS


class HybridRecommender:
    """Combine Jaccard ingredient fit with Item-KNN preference predictions."""

    def __init__(self, content_recommender, collaborative_recommender):
        self.content_recommender = content_recommender
        self.collaborative_recommender = collaborative_recommender

    def weights_for_user(self, user_id) -> tuple[float, float]:
        """Give content more weight for sparse users and KNN more for active users."""
        user_idx = self.collaborative_recommender.user_to_index.get(user_id)
        if user_idx is None:
            return 1.0, 0.0
        history_size = len(self.collaborative_recommender.user_rated_items[user_idx])
        if history_size < 5:
            return 0.8, 0.2
        if history_size >= 20:
            return 0.35, 0.65
        collaborative_weight = 0.2 + (history_size - 5) / 15 * 0.45
        return 1.0 - collaborative_weight, collaborative_weight

    def recommend(self, user_id, ingredients, *, top_k=10, max_minutes=None):
        if top_k < 1:
            raise ValueError("top_k must be positive")

        candidate_count = min(max(top_k * 5, top_k), self.content_recommender.max_top_k)
        content = self.content_recommender.recommend(
            ingredients, top_k=candidate_count, max_minutes=max_minutes
        ).reindex(columns=RECOMMENDATION_COLUMNS)
        collaborative = self.collaborative_recommender.recommend(
            user_id, top_k=candidate_count, max_minutes=max_minutes
        )
        collaborative = collaborative.reindex(columns=[
            "recipe_id", "title", "predicted_preference", "minutes", "source"
        ])

        if content.empty and collaborative.empty:
            return pd.DataFrame(columns=RECOMMENDATION_COLUMNS + [
                "predicted_preference", "hybrid_score"
            ])

        # Rename content metadata before merging. This avoids Pandas creating
        # minutes_x/minutes_y when the two recommenders both return minutes.
        content = content.rename(columns={
            "title": "content_title",
            "minutes": "content_minutes",
            "source": "content_source",
        })
        collaborative = collaborative.rename(columns={
            "title": "cf_title",
            "minutes": "cf_minutes",
            "source": "cf_source",
        })
        merged = content.merge(
            collaborative[["recipe_id", "cf_title", "predicted_preference",
                           "cf_minutes", "cf_source"]],
            on="recipe_id", how="outer", validate="one_to_one",
        )

        merged["title"] = merged["content_title"].fillna(merged["cf_title"])
        merged["minutes"] = merged["content_minutes"].combine_first(merged["cf_minutes"])
        merged["source"] = merged["content_source"].fillna(merged["cf_source"]).fillna("Food.com")
        merged["content_score"] = pd.to_numeric(
            merged["content_score"], errors="coerce"
        ).fillna(0.0).clip(0.0, 1.0)
        merged["ingredient_coverage"] = pd.to_numeric(
            merged["ingredient_coverage"], errors="coerce"
        ).fillna(0.0).clip(0.0, 1.0)
        preference = pd.to_numeric(merged["predicted_preference"], errors="coerce")
        merged["preference_score"] = ((preference - 1.0) / 4.0).clip(0.0, 1.0).fillna(0.0)

        alpha, beta = self.weights_for_user(user_id)
        merged["hybrid_score"] = (
            alpha * merged["content_score"] + beta * merged["preference_score"]
        )

        if max_minutes is not None:
            minutes = pd.to_numeric(merged["minutes"], errors="coerce")
            # Keep the same strict time-limit behavior as ContentBasedRecommender:
            # unknown or invalid cooking times are not eligible under a time cap.
            merged = merged[minutes.notna() & (minutes > 0) & (minutes <= max_minutes)]

        for column, default in {
            "matched_ingredients": None,
            "missing_ingredients": None,
            "missing_count": 0,
        }.items():
            if column not in merged:
                merged[column] = default
        merged["matched_ingredients"] = merged["matched_ingredients"].apply(
            lambda value: value if isinstance(value, list) else []
        )
        merged["missing_ingredients"] = merged["missing_ingredients"].apply(
            lambda value: value if isinstance(value, list) else []
        )
        merged["missing_count"] = pd.to_numeric(
            merged["missing_count"], errors="coerce"
        ).fillna(0).astype(int)

        return merged.sort_values(
            ["hybrid_score", "recipe_id"], ascending=[False, True], kind="stable"
        ).reindex(columns=RECOMMENDATION_COLUMNS + [
            "predicted_preference", "hybrid_score"
        ]).head(top_k).reset_index(drop=True)
