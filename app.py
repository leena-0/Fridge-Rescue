"""Streamlit entry point for Fridge Rescue."""

import streamlit as st

from fridge_rescue.config import load_config

config = load_config()

st.set_page_config(
    page_title=config.project_name,
    page_icon="🍳",
    layout="wide",
)

st.title("🍳 Fridge Rescue")
st.caption("냉장고 속 재료로 만들 수 있는 레시피를 추천합니다.")

st.info(
    "프로젝트 초기 설정이 완료되었습니다. "
    "데이터 파이프라인과 콘텐츠 기반 추천 모델은 다음 이슈에서 연결됩니다."
)

with st.sidebar:
    st.header("프로젝트 설정")
    st.write(f"기본 추천 개수: {config.default_top_k}")
    st.write(f"난수 시드: {config.random_seed}")
