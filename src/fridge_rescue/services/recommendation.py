"""Recommendation service connecting UI requests to available rankers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from fridge_rescue.recommenders.content_based import ContentBasedRecommender


class RecommendationMode(StrEnum):
    CONTENT = "content"
    COLLABORATIVE = "collaborative"
    HYBRID = "hybrid"


class UserMode(StrEnum):
    NEW = "new"
    EXISTING = "existing"


class ModelUnavailableError(RuntimeError):
    """Raised when a requested recommender is not connected."""


@dataclass(frozen=True)
class RecommendationRequest:
    ingredients: list[str] | tuple[str, ...] | str
    top_k: int
    max_minutes: float | None = None
    mode: RecommendationMode = RecommendationMode.CONTENT
    user_mode: UserMode = UserMode.NEW
    user_id: str | None = None
    preferred_category: str | None = None


@dataclass(frozen=True)
class RecommendationResponse:
    results: pd.DataFrame
    requested_mode: RecommendationMode
    effective_mode: RecommendationMode
    fallback_reason: str | None = None


Ranker = Callable[[RecommendationRequest], pd.DataFrame]


class RecommendationService:
    def __init__(
        self,
        content_recommender: ContentBasedRecommender,
        *,
        collaborative_ranker: Ranker | None = None,
        hybrid_ranker: Ranker | None = None,
    ) -> None:
        self.content_recommender = content_recommender
        self._rankers = {
            RecommendationMode.COLLABORATIVE: collaborative_ranker,
            RecommendationMode.HYBRID: hybrid_ranker,
        }

    @property
    def available_modes(self):
        return (RecommendationMode.CONTENT,) + tuple(
            mode for mode, ranker in self._rankers.items() if ranker is not None
        )

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        requested_mode = RecommendationMode(request.mode)
        user_mode = UserMode(request.user_mode)
        effective_mode = requested_mode
        fallback_reason = None

        if user_mode is UserMode.NEW and requested_mode is not RecommendationMode.CONTENT:
            effective_mode = RecommendationMode.CONTENT
            fallback_reason = "신규 사용자는 평점 이력이 없어 콘텐츠 기반 추천을 사용합니다."

        if effective_mode is RecommendationMode.CONTENT:
            results = self.content_recommender.recommend(
                request.ingredients, top_k=request.top_k, max_minutes=request.max_minutes
            )
        else:
            ranker = self._rankers.get(effective_mode)
            if ranker is None:
                raise ModelUnavailableError(
                    f"{effective_mode.value} 추천 모델은 아직 연결되지 않았습니다."
                )
            if not request.user_id:
                raise ValueError("기존 사용자 모드에서는 학습 데이터의 사용자 ID가 필요합니다.")
            results = ranker(request)

        if effective_mode is not RecommendationMode.CONTENT and results.empty:
            results = self.content_recommender.recommend(
                request.ingredients, top_k=request.top_k, max_minutes=request.max_minutes
            )
            fallback_reason = "협업 필터링 근거가 부족해 콘텐츠 기반 추천으로 전환했습니다."

        return RecommendationResponse(results, requested_mode, effective_mode, fallback_reason)
