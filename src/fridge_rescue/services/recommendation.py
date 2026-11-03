"""Stable recommendation service contract shared by the UI and future rankers."""

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
    """Raised when the UI requests a recommender that has not been connected yet."""


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
    """Route requests to content, collaborative, or hybrid rankers with cold-start fallback."""

    def __init__(
        self,
        content_recommender: ContentBasedRecommender,
        *,
        collaborative_ranker: Ranker | None = None,
        hybrid_ranker: Ranker | None = None,
    ) -> None:
        self.content_recommender = content_recommender
        self._rankers: dict[RecommendationMode, Ranker | None] = {
            RecommendationMode.COLLABORATIVE: collaborative_ranker,
            RecommendationMode.HYBRID: hybrid_ranker,
        }

    @property
    def available_modes(self) -> tuple[RecommendationMode, ...]:
        modes = [RecommendationMode.CONTENT]
        modes.extend(mode for mode, ranker in self._rankers.items() if ranker is not None)
        return tuple(modes)

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        requested_mode = RecommendationMode(request.mode)
        user_mode = UserMode(request.user_mode)
        fallback_reason: str | None = None
        effective_mode = requested_mode

        if user_mode is UserMode.NEW and requested_mode is not RecommendationMode.CONTENT:
            effective_mode = RecommendationMode.CONTENT
            fallback_reason = "사용자 이력이 없어 콘텐츠 기반 추천으로 전환했습니다."

        if effective_mode is RecommendationMode.CONTENT:
            results = self.content_recommender.recommend(
                request.ingredients,
                top_k=request.top_k,
                max_minutes=request.max_minutes,
            )
        else:
            ranker = self._rankers[effective_mode]
            if ranker is None:
                raise ModelUnavailableError(
                    f"{effective_mode.value} 추천 모델은 아직 연결되지 않았습니다."
                )
            results = ranker(request)

        return RecommendationResponse(
            results=results,
            requested_mode=requested_mode,
            effective_mode=effective_mode,
            fallback_reason=fallback_reason,
        )
