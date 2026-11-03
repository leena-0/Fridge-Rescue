# Fridge Rescue

사용자가 냉장고에 가지고 있는 재료를 입력하면 만들 수 있는 요리를 추천하는 머신러닝 기반 개인화 레시피 추천 시스템입니다.

Food.com 데이터를 이용해 콘텐츠 기반 추천과 협업 필터링을 개발하고, RecipeNLG 데이터로 고정된 콘텐츠 기반 추천기의 외부 검색 성능을 확인합니다.

## 개발 환경

- Python 3.11 이상 3.14 미만
- 패키지 및 가상환경 관리: [uv](https://docs.astral.sh/uv/) 권장
- 애플리케이션: Streamlit
- 테스트: pytest
- 코드 검사: Ruff

### uv로 설치

```bash
uv sync --all-extras
```

### venv와 pip로 설치

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,eda]"
```

## 실행 방법

Streamlit 애플리케이션을 실행합니다.

```bash
uv run streamlit run app.py
```

pip 환경에서는 다음 명령을 사용합니다.

```bash
streamlit run app.py
```

## 테스트와 코드 검사

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## 데이터 준비

원본 데이터는 라이선스와 용량 문제로 Git에 올리지 않습니다. 각 팀원이 아래 출처에서 직접 내려받아 `data/raw/`에 배치해야 합니다.

### Food.com Recipes and Interactions

- 출처: [Kaggle Food.com Recipes and User Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
- 기본 입력 파일
  - `data/raw/foodcom/RAW_recipes.csv`
  - `data/raw/foodcom/RAW_interactions.csv`

### RecipeNLG

- 출처: [RecipeNLG 프로젝트](https://recipenlg.cs.put.poznan.pl/)
- 내려받은 레시피 CSV 파일을 `data/raw/recipenlg/`에 배치합니다.
- 실제 파일명과 위치는 `configs/default.yaml`에서 변경할 수 있습니다.

데이터 출처의 이용 및 재배포 조건을 확인하고 원본 파일을 저장소에 커밋하지 마세요.

## 프로젝트 구조

```text
Fridge-Rescue/
├── app.py                     # Streamlit 진입점
├── configs/default.yaml       # 공통 경로와 재현성 설정
├── data/
│   ├── raw/                   # 다운로드한 원본 데이터, Git 제외
│   └── processed/             # 전처리 산출물, Git 제외
├── models/                    # 학습된 모델, Git 제외
├── notebooks/                 # 탐색 및 실험 노트북
├── outputs/
│   ├── figures/               # 보고서와 발표용 그래프
│   └── tables/                # 평가 결과표
├── src/fridge_rescue/
│   ├── data/                  # 데이터 로딩과 전처리
│   ├── recommenders/          # 추천 알고리즘
│   ├── services/              # UI와 추천 모델 연결 계층
│   └── utils/                 # 공통 유틸리티
└── tests/                     # 자동화 테스트
```

## 공통 설정

기본 난수 시드는 `42`입니다. 데이터 경로, 추천 개수와 같은 공통 값은 `configs/default.yaml`에서 관리합니다. 실험 결과를 보고할 때 사용한 설정과 시드를 함께 기록합니다.

## 데이터셋 역할

- Food.com: 데이터 탐색, 전처리, 학습, 검증 및 held-out 평가
- RecipeNLG: Food.com과 중복을 제거한 뒤 콘텐츠 기반 추천의 외부 검색 테스트에만 사용

RecipeNLG에는 Food.com 사용자 평점 이력이 없으므로 협업 필터링 정확도 평가에 사용하지 않습니다.
