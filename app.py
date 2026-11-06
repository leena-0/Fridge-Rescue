"""Streamlit entry point for Fridge Rescue."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fridge_rescue.config import load_config
from fridge_rescue.data.preprocessing import load_ingredient_synonyms
from fridge_rescue.recommenders.content_based import ContentBasedRecommender
from fridge_rescue.recommenders.hybrid import HybridRecommender
from fridge_rescue.recommenders.item_knn_optimized import ItemKNNRecommender
from fridge_rescue.services.recommendation import (
    RecommendationMode,
    RecommendationRequest,
    RecommendationService,
    UserMode,
)

MODE_LABELS = {
    "Content-based": RecommendationMode.CONTENT,
    "Collaborative (Item-KNN)": RecommendationMode.COLLABORATIVE,
    "Hybrid": RecommendationMode.HYBRID,
}
USER_MODE_LABELS = {"신규 사용자": UserMode.NEW, "기존 사용자": UserMode.EXISTING}


@st.cache_resource(show_spinner="추천 모델을 불러오고 있습니다...")
def _load_recommendation_service(
    recipe_path: str, train_path: str, synonym_path: str, max_top_k: int
):
    synonyms = load_ingredient_synonyms(synonym_path)
    content = ContentBasedRecommender.from_parquet(
        recipe_path, synonyms=synonyms, max_top_k=max_top_k
    )
    train = pd.read_parquet(train_path)
    recipes = pd.read_parquet(recipe_path)
    collaborative = ItemKNNRecommender(train, recipes, n_neighbors=50)
    hybrid = HybridRecommender(content, collaborative)

    def collaborative_ranker(request):
        try:
            user_id = int(request.user_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("사용자 ID는 Food.com 학습 데이터의 숫자 ID여야 합니다.") from exc
        return collaborative.recommend(
            user_id, top_k=request.top_k, max_minutes=request.max_minutes
        )

    def hybrid_ranker(request):
        try:
            user_id = int(request.user_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("사용자 ID는 Food.com 학습 데이터의 숫자 ID여야 합니다.") from exc
        return hybrid.recommend(
            user_id, request.ingredients, top_k=request.top_k, max_minutes=request.max_minutes
        )

    return RecommendationService(
        content, collaborative_ranker=collaborative_ranker, hybrid_ranker=hybrid_ranker
    )


def _ingredient_badges(values, empty_text="없음"):
    if isinstance(values, (list, tuple)):
        items = list(values)
    elif hasattr(values, "tolist"):
        items = values.tolist()
    else:
        items = []
    return " · ".join(f"`{item}`" for item in items) if items else empty_text


def _render_recommendations(results):
    if results.empty:
        st.warning("추천 결과가 없습니다. 재료나 사용자 ID를 확인해 주세요.")
        return
    st.subheader(f"추천 결과 {len(results)}개")
    for rank, row in enumerate(results.itertuples(index=False), start=1):
        with st.container(border=True):
            a, b, c = st.columns([5, 1.4, 1.4])
            a.markdown(f"### {rank}. {row.title}")
            a.caption(f"ID: {row.recipe_id} · 출처: {row.source}")
            score = getattr(row, "hybrid_score", None)
            if score is None or pd.isna(score):
                score = getattr(row, "content_score", 0.0)
            b.metric("추천 점수", f"{float(score):.3f}")
            minutes = row.minutes
            c.metric("조리 시간", f"{minutes:g}분" if pd.notna(minutes) else "정보 없음")
            st.progress(
                float(row.ingredient_coverage), text=f"재료 coverage {row.ingredient_coverage:.1%}"
            )
            st.markdown("**일치 재료**  " + _ingredient_badges(row.matched_ingredients))
            st.markdown(
                f"**부족 재료 {row.missing_count}개**  "
                + _ingredient_badges(row.missing_ingredients, "추가 재료 없음")
            )
            if hasattr(row, "predicted_preference") and pd.notna(row.predicted_preference):
                st.caption(f"KNN 예측 선호도: {row.predicted_preference:.2f} / 5")


config = load_config()
st.set_page_config(page_title=config.project_name, page_icon="🍳", layout="wide")
st.title("🍳 Fridge Rescue")
st.caption("냉장고 속 재료와 사용자 선호도를 반영해 레시피를 추천합니다.")

with st.sidebar:
    st.header("추천 설정")
    user_mode_label = st.radio("사용자 모드", list(USER_MODE_LABELS), horizontal=True)
    user_mode = USER_MODE_LABELS[user_mode_label]
    user_id = st.text_input(
        "Food.com 사용자 ID", value="424680", disabled=user_mode is UserMode.NEW
    )
    mode_label = st.selectbox("추천 방식", list(MODE_LABELS))
    mode = MODE_LABELS[mode_label]
    top_k = st.slider("추천 개수", 1, config.max_top_k, config.default_top_k)
    use_time_limit = st.checkbox("최대 조리 시간 사용")
    max_minutes = st.number_input(
        "최대 조리 시간(분)", min_value=1, value=30, disabled=not use_time_limit
    )

st.subheader("보유 재료")
ingredients = st.text_area("재료를 쉼표로 구분해 입력하세요.", value="egg, milk, onion", height=100)
submitted = st.button("레시피 추천받기", type="primary", use_container_width=True)

if submitted:
    if not any(value.strip() for value in ingredients.split(",")):
        st.error("한 개 이상의 재료를 입력해 주세요.")
        st.stop()
    try:
        service = _load_recommendation_service(
            str(config.processed_data_dir / "foodcom" / "recipes.parquet"),
            str(config.processed_data_dir / "foodcom" / "splits" / "train.parquet"),
            str(config.ingredient_synonyms_path),
            config.max_top_k,
        )
        response = service.recommend(
            RecommendationRequest(
                ingredients=ingredients,
                top_k=top_k,
                max_minutes=float(max_minutes) if use_time_limit else None,
                mode=mode,
                user_mode=user_mode,
                user_id=user_id.strip() or None,
            )
        )
        if response.fallback_reason:
            st.info(response.fallback_reason)
        _render_recommendations(response.results)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        st.error(str(error))

st.divider()
st.caption("추천 점수는 데이터 기반 참고값이며 식품 안전·알레르기 적합성을 보장하지 않습니다.")
