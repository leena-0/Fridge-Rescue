from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path).run()

    assert not app.exception
    assert app.title[0].value == "🍳 Fridge Rescue"
    assert "프로젝트 초기 설정이 완료되었습니다" in app.info[0].value
