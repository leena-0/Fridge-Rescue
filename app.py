"""Streamlit entry point for Fridge Rescue."""

import pandas as pd
import streamlit as st

from fridge_rescue.config import load_config
from fridge_rescue.data.preprocessing import load_ingredient_synonyms
from fridge_rescue.recommenders.content_based import ContentBasedRecommender
from fridge_rescue.services.recommendation import (
    ModelUnavailableError,
    RecommendationMode,
    RecommendationRequest,
    RecommendationService,
    UserMode,
)

MODE_LABELS = {
    "Content-based": RecommendationMode.CONTENT,
    "Collaborative": RecommendationMode.COLLABORATIVE,
    "Hybrid": RecommendationMode.HYBRID,
}
USER_MODE_LABELS = {
    "신규 사용자": UserMode.NEW,
    "기존 사용자": UserMode.EXISTING,
}


@st.cache_resource(show_spinner="레시피 검색 인덱스를 준비하고 있습니다...")
def _load_recommendation_service(
    recipe_path: str,
    synonym_path: str,
    max_top_k: int,
) -> RecommendationService:
    synonyms = load_ingredient_synonyms(synonym_path)
    recommender = ContentBasedRecommender.from_parquet(
        recipe_path,
        synonyms=synonyms,
        max_top_k=max_top_k,
    )
    return RecommendationService(recommender)


def _ingredient_badges(values: object, *, empty_text: str) -> str:
    if isinstance(values, pd.Series):
        ingredients = values.tolist()
    elif hasattr(values, "tolist"):
        ingredients = values.tolist()
    elif isinstance(values, (list, tuple)):
        ingredients = list(values)
    else:
        ingredients = []
    if not ingredients:
        return empty_text
    return " · ".join(f"`{ingredient}`" for ingredient in ingredients)


def _render_recommendations(results: pd.DataFrame) -> None:
    if results.empty:
        st.warning("조건에 맞는 레시피가 없습니다. 재료나 조리 시간 조건을 조정해 주세요.")
        return

    st.subheader(f"추천 결과 {len(results)}개")
    for rank, row in enumerate(results.itertuples(index=False), start=1):
        with st.container(border=True):
            title_column, score_column, time_column = st.columns([5, 1.4, 1.4])
            title_column.markdown(f"### {rank}. {row.title}")
            title_column.caption(f"ID: {row.recipe_id} · 출처: {row.source}")
            score_column.metric("Jaccard", f"{row.content_score:.3f}")
            time_text = f"{row.minutes:g}분" if row.minutes is not None else "정보 없음"
            time_column.metric("조리 시간", time_text)
            st.progress(
                float(row.ingredient_coverage),
                text=f"레시피 재료 coverage {row.ingredient_coverage:.1%}",
            )
            st.markdown(
                "**일치 재료**  " + _ingredient_badges(row.matched_ingredients, empty_text="없음")
            )
            st.markdown(
                f"**부족 재료 {row.missing_count}개**  "
                + _ingredient_badges(row.missing_ingredients, empty_text="추가 재료 없음")
            )


config = load_config()

st.set_page_config(
    page_title=config.project_name,
    page_icon="🍳",
    layout="wide",
)

st.title("🍳 Fridge Rescue")
st.caption("냉장고 속 재료로 만들 수 있는 레시피를 찾아보세요.")

with st.sidebar:
    st.header("추천 설정")
    user_mode_label = st.radio(
        "사용자 모드",
        options=list(USER_MODE_LABELS),
        horizontal=True,
        help="신규 사용자는 평점 이력 없이 재료만으로 추천합니다.",
    )
    user_mode = USER_MODE_LABELS[user_mode_label]
    user_id = None
    if user_mode is UserMode.EXISTING:
        user_id = st.text_input("사용자 ID", value="demo_user_001")

    mode_label = st.selectbox("추천 방식", options=list(MODE_LABELS))
    recommendation_mode = MODE_LABELS[mode_label]
    top_k = st.slider(
        "추천 개수",
        min_value=1,
        max_value=config.max_top_k,
        value=config.default_top_k,
    )
    use_time_limit = st.checkbox("최대 조리 시간 사용")
    max_minutes_input = st.number_input(
        "최대 조리 시간(분)",
        min_value=1,
        value=30,
        disabled=not use_time_limit,
    )

st.subheader("보유 재료")
ingredients_input = st.text_area(
    "재료를 쉼표로 구분해 입력하세요.",
    value="egg, milk, onion",
    height=100,
    placeholder="예: egg, milk, green onion",
)
preferred_category = st.text_input(
    "선호 카테고리 (선택)",
    placeholder="예: breakfast",
)
st.caption("카테고리 선호는 Hybrid 모델 연결 후 soft preference로 사용됩니다.")

model_available = recommendation_mode is RecommendationMode.CONTENT
if user_mode is UserMode.NEW and not model_available:
    st.info("신규 사용자는 사용자 이력이 없어 Content-based 방식으로 자동 전환됩니다.")
elif not model_available:
    st.warning("Collaborative와 Hybrid 모델은 아직 연결되지 않았습니다.")

submitted = st.button(
    "레시피 추천받기",
    type="primary",
    use_container_width=True,
    disabled=user_mode is UserMode.EXISTING and not model_available,
)

if submitted:
    if not any(item.strip() for item in ingredients_input.split(",")):
        st.error("한 개 이상의 재료를 입력해 주세요.")
        st.stop()
    if user_mode is UserMode.EXISTING and not (user_id or "").strip():
        st.error("기존 사용자 모드에서는 사용자 ID를 입력해 주세요.")
        st.stop()
    recipe_path = config.processed_data_dir / "foodcom" / "recipes.parquet"
    try:
        service = _load_recommendation_service(
            str(recipe_path),
            str(config.ingredient_synonyms_path),
            config.max_top_k,
        )
        response = service.recommend(
            RecommendationRequest(
                ingredients=ingredients_input,
                top_k=top_k,
                max_minutes=float(max_minutes_input) if use_time_limit else None,
                mode=recommendation_mode,
                user_mode=user_mode,
                user_id=user_id,
                preferred_category=preferred_category.strip() or None,
            )
        )
    except FileNotFoundError:
        st.error("전처리된 Food.com 레시피 파일이 없습니다.")
        st.code("uv run python -m fridge_rescue.data.preprocessing --dataset foodcom --max-rows 0")
    except (ValueError, ModelUnavailableError) as error:
        st.error(str(error))
    else:
        if response.fallback_reason:
            st.info(response.fallback_reason)
        _render_recommendations(response.results)

st.divider()
st.caption(
    "추천 점수는 재료 집합의 Jaccard similarity이며, 실제 식품 안전·알레르기 적합성을 "
    "보장하지 않습니다."
)
