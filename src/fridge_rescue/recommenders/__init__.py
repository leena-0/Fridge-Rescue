"""Recommendation algorithms."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fridge_rescue.recommenders.content_based import ContentBasedRecommender

__all__ = ["ContentBasedRecommender", "load_content_recommender"]


def __getattr__(name: str) -> Any:
    """Lazily expose recommenders without importing executable CLI modules eagerly."""

    if name in __all__:
        from fridge_rescue.recommenders.content_based import (
            ContentBasedRecommender,
            load_content_recommender,
        )

        return {
            "ContentBasedRecommender": ContentBasedRecommender,
            "load_content_recommender": load_content_recommender,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
