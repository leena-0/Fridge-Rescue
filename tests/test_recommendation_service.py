import pandas as pd
import pytest

from fridge_rescue.recommenders.content_based import ContentBasedRecommender
from fridge_rescue.services.recommendation import (
    ModelUnavailableError,
    RecommendationMode,
    RecommendationRequest,
    RecommendationService,
    UserMode,
)


def _service() -> RecommendationService:
    recipes = pd.DataFrame(
        {
            "recipe_id": [1],
            "title": ["Egg rice"],
            "normalized_ingredients": [["egg", "rice"]],
            "minutes": [10],
            "dataset": ["foodcom"],
        }
    )
    return RecommendationService(ContentBasedRecommender(recipes))


def test_content_request_returns_recommendations() -> None:
    response = _service().recommend(
        RecommendationRequest(
            ingredients=["egg", "rice"],
            top_k=5,
            mode=RecommendationMode.CONTENT,
            user_mode=UserMode.EXISTING,
            user_id="demo_user",
        )
    )

    assert response.effective_mode is RecommendationMode.CONTENT
    assert response.fallback_reason is None
    assert response.results["recipe_id"].tolist() == [1]


def test_new_user_falls_back_to_content_recommendations() -> None:
    response = _service().recommend(
        RecommendationRequest(
            ingredients=["egg"],
            top_k=5,
            mode=RecommendationMode.HYBRID,
            user_mode=UserMode.NEW,
        )
    )

    assert response.requested_mode is RecommendationMode.HYBRID
    assert response.effective_mode is RecommendationMode.CONTENT
    assert response.fallback_reason is not None


def test_existing_user_gets_clear_error_for_unconnected_model() -> None:
    service = _service()

    with pytest.raises(ModelUnavailableError, match="아직 연결되지 않았습니다"):
        service.recommend(
            RecommendationRequest(
                ingredients=["egg"],
                top_k=5,
                mode=RecommendationMode.COLLABORATIVE,
                user_mode=UserMode.EXISTING,
                user_id="demo_user",
            )
        )

    assert service.available_modes == (RecommendationMode.CONTENT,)
