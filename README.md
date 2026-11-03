# Fridge Rescue

사용자가 냉장고에 가지고 있는 재료를 입력하면 만들 수 있는 요리를 추천하는 머신러닝 기반 개인화 레시피 추천 시스템입니다.

Food.com 데이터를 이용해 콘텐츠 기반 추천과 협업 필터링을 개발하고, RecipeNLG 데이터로 고정된 콘텐츠 기반 추천기의 외부 검색 성능을 확인합니다.

## Planned System Architecture

![Fridge Rescue planned system architecture](docs/images/fridge_rescue_architecture.svg)

- Food.com의 레시피와 사용자 상호작용은 콘텐츠 기반 추천과 Item KNN 개발에 사용합니다.
- Hybrid ranker는 콘텐츠 점수와 Item KNN의 예측 선호도를 결합합니다.
- 사용자 이력이 없는 cold-start 상황에서는 콘텐츠 기반 추천 결과를 사용합니다.
- Protocol A는 Food.com에서 Content-based, Item KNN, Hybrid 세 방법을 비교합니다.
- Protocol B는 Food.com 및 RecipeNLG에서 콘텐츠 기반 레시피 검색 성능만 평가합니다.

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

원본 데이터는 라이선스와 용량 문제로 Git에 올리지 않습니다. 각 팀원이 아래 출처에서 직접 내려받은 뒤 정해진 경로에 배치해야 합니다.

먼저 저장 디렉터리를 만듭니다.

```bash
mkdir -p data/raw/foodcom data/raw/recipenlg
```

### Food.com Recipes and Interactions

1. [Kaggle Food.com Recipes and User Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)에 로그인합니다.
2. **Download** 버튼으로 데이터셋 ZIP 파일을 내려받습니다.
3. 내려받은 ZIP 파일을 `data/raw/foodcom/`에 압축 해제합니다.
4. 다음 두 파일이 정확한 경로에 있는지 확인합니다.

```text
data/raw/foodcom/RAW_recipes.csv
data/raw/foodcom/RAW_interactions.csv
```

macOS 또는 Linux에서는 저장소 루트에서 다음과 같이 압축을 해제할 수 있습니다.

```bash
unzip ~/Downloads/food-com-recipes-and-user-interactions.zip -d data/raw/foodcom
```

Kaggle CLI를 이미 설정했다면 다음 방법도 사용할 수 있습니다.

```bash
kaggle datasets download \
  -d shuyangli94/food-com-recipes-and-user-interactions \
  -p data/raw/foodcom \
  --unzip
```

### RecipeNLG

1. [RecipeNLG 공식 프로젝트](https://recipenlg.cs.put.poznan.pl/)에서 데이터 이용 조건을 확인합니다.
2. 프로젝트 사이트의 데이터셋 다운로드 안내에 따라 압축 파일을 내려받습니다.
3. 압축을 해제한 뒤 전체 레시피 CSV를 다음 경로에 배치합니다.

```text
data/raw/recipenlg/full_dataset.csv
```

내려받은 CSV의 파일명이 다르면 `full_dataset.csv`로 변경하거나 `configs/default.yaml`의 `datasets.recipenlg.recipes` 값을 실제 경로로 수정합니다.

### 배치 결과 확인

```bash
ls -lh data/raw/foodcom/RAW_recipes.csv
ls -lh data/raw/foodcom/RAW_interactions.csv
ls -lh data/raw/recipenlg/full_dataset.csv
```

세 파일이 모두 표시되면 데이터 로더를 실행할 준비가 된 것입니다. RecipeNLG 파일은 크기가 크므로 다운로드와 압축 해제를 위한 충분한 디스크 공간을 확보하세요.

데이터 출처의 이용 및 재배포 조건을 확인하고 원본 파일을 저장소에 커밋하지 마세요.

### 데이터 로더와 EDA 실행

두 데이터셋을 함께 검증하고 EDA 요약 및 그래프를 생성합니다. 기본 실행은 각 테이블의 앞 100,000행을 사용해 개발 중 메모리 사용량을 제한합니다.

```bash
uv run python -m fridge_rescue.data.eda --dataset all
```

빠른 확인을 위해 행 수를 더 줄일 수 있습니다.

```bash
uv run python -m fridge_rescue.data.eda --dataset all --max-rows 10000
```

전체 데이터를 분석하려면 `--max-rows 0`을 사용합니다. RecipeNLG는 약 2GB이므로 실행 시간이 오래 걸릴 수 있습니다.

```bash
uv run python -m fridge_rescue.data.eda --dataset all --max-rows 0
```

결과는 다음 위치에 저장됩니다.

```text
outputs/figures/
outputs/tables/eda_summary.json
outputs/tables/eda_missing_values.csv
```

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
