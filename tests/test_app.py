from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path).run()

    assert not app.exception
    assert app.title[0].value == "🍳 Fridge Rescue"
    assert app.radio[0].value == "신규 사용자"
    assert app.selectbox[0].value == "Content-based"
    assert app.text_area[0].value == "egg, milk, onion"
    assert app.button[0].label == "레시피 추천받기"


def test_streamlit_app_rejects_empty_ingredients_before_loading_data() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path).run()

    app.text_area[0].set_value(" , ")
    app.button[0].click().run()

    assert not app.exception
    assert app.error[0].value == "한 개 이상의 재료를 입력해 주세요."
