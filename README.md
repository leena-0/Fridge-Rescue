# Fridge Rescue

> 냉장고에 있는 재료와 사용자의 과거 선호도를 함께 활용해 레시피를 추천하는 머신러닝 기반 추천 시스템

사용자는 보유 재료와 최대 조리 시간을 입력하고, 신규 사용자는 재료 기반 추천을, 기존 사용자는 협업 필터링 또는 하이브리드 추천을 받을 수 있습니다. 추천 결과에는 일치 재료, 부족 재료, 재료 충족률과 예상 선호도를 함께 제시해 결과를 해석할 수 있도록 했습니다.

## 1. 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 문제 | 냉장고에 있는 재료를 활용하면서 개인 취향에도 맞는 레시피를 찾기 어렵다. |
| 입력 | 보유 재료, 선택적인 최대 조리 시간, 기존 사용자의 Food.com 사용자 ID |
| 출력 | 추천 레시피 순위, 추천 점수, 일치·부족 재료, 재료 충족률, 조리 시간 |
| 개발 데이터 | Food.com 레시피와 사용자-레시피 평점 상호작용 |
| 외부 테스트 데이터 | RecipeNLG 레시피 |
| 추천 모델 | Content-based, Item-KNN, Weighted Hybrid |
| 애플리케이션 | Streamlit |
| 평가 | Protocol A: Precision@10·Recall@10·NDCG@10, Protocol B: Hit@10·MRR@10 |

### 핵심 연구 질문

1. 보유 재료의 일치도만으로 유용한 레시피를 검색할 수 있는가?
2. 사용자 평점 이력을 활용한 Item-KNN이 콘텐츠 기반 추천보다 개인화 성능을 높이는가?
3. 두 신호를 결합한 Hybrid 모델이 단일 모델보다 더 나은 순위 품질을 보이는가?
4. Food.com에서 개발한 콘텐츠 기반 추천기가 RecipeNLG 외부 데이터에서도 부분 재료 검색에 작동하는가?

## 2. 시스템 아키텍처

![Fridge Rescue system architecture](docs/images/fridge_rescue_architecture.svg)

1. Food.com 레시피와 평점 상호작용을 정제하고 사용자별 train/validation/test로 분할합니다.
2. RecipeNLG는 Food.com과 겹치는 레시피를 제거한 뒤 외부 콘텐츠 검색 평가에만 사용합니다.
3. Content-based 모델은 재료 집합의 Jaccard similarity를 계산합니다.
4. Item-KNN은 사용자 평점 이력을 바탕으로 레시피 선호도를 예측합니다.
5. Hybrid 모델은 콘텐츠 점수와 정규화된 예측 선호도를 가중 결합합니다.
6. 사용자 이력이 없는 cold-start 상황에서는 Content-based 결과로 fallback합니다.

## 3. 데이터셋과 전처리

| 데이터셋 | 프로젝트 내 역할 | 사용 범위 |
|---|---|---|
| [Food.com Recipes and Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) | 모델 개발 및 내부 평가 | 레시피, 재료, 조리 시간, 사용자 평점 |
| [RecipeNLG](https://aclanthology.org/2020.inlg-1.4/) | 외부 콘텐츠 검색 평가 | 레시피 제목과 NER 재료 엔터티 |

### 전처리 원칙

- Food.com과 RecipeNLG에 동일한 재료 정규화 규칙을 적용합니다.
- 소문자화, 공백·특수문자 정리, 수량·단위 제거 후 보수적인 동의어만 치환합니다.
- Food.com 평점은 1~5만 명시적 선호도로 사용하고 0·결측·범위 밖 값은 제외합니다.
- 같은 사용자-레시피 조합이 여러 번 있으면 가장 최근 평점을 유지합니다.
- 고정 시드 42로 사용자별 80/10/10 분할을 수행합니다.
- validation과 test를 각각 한 건 이상 만들 수 없는 사용자는 train에만 포함합니다.
- 사용자·레시피 최소 활동량 기준은 train에서만 계산해 데이터 누수를 방지합니다.
- RecipeNLG에서 Food.com URL 및 정규화 제목·재료 지문과 정확히 겹치는 항목을 제거합니다.

> **RecipeNLG 출처 제한:** 최종 실험에서는 공식 다운로드 경로에서 파일을 확보하지 못해 Hugging Face의 `Mahimas/recipenlg` 미러를 사용했습니다. 공식 배포본과 완전히 동일한지는 별도로 검증하지 않았으며, 사용 전 데이터 출처와 라이선스 조건을 확인해야 합니다.

## 4. 추천 모델

### 4.1 Content-based

사용자가 입력한 재료 집합 $U$와 레시피 재료 집합 $R$ 사이의 Jaccard similarity를 사용합니다.

$$
J(U, R) = \frac{|U \cap R|}{|U \cup R|}
$$

- 재료 충족률: $|U \cap R| / |R|$
- 설명 정보: 일치 재료, 부족 재료, 부족 재료 수
- 조리 시간 제한: 랭킹 전에 hard filter로 적용
- 동점 처리: Jaccard → 충족률 → 부족 재료 수 → 제목 → 레시피 ID

### 4.2 Item-KNN

- 사용자-레시피 평점 행렬에서 사용자 평균을 제거한 값을 사용합니다.
- item-item cosine similarity로 이웃 레시피를 탐색합니다.
- 공통 평가 사용자가 2명 미만인 이웃을 제외하고 shrinkage를 적용합니다.
- 이미 평가한 레시피는 후보에서 제외하고 예측 평점을 1~5 범위로 제한합니다.

### 4.3 Weighted Hybrid

$$
score = \alpha \cdot Jaccard + \beta \cdot normalized\_KNN
$$

- 오프라인 평가에서는 validation으로 가중치를 선택하고 test에서 최종 성능을 비교합니다.
- Streamlit에서는 사용자 이력 수에 따라 콘텐츠와 협업 신호의 비중을 조절합니다.
- 사용자 이력이 없으면 Content-based만 사용합니다.

## 5. 평가 설계

### Protocol A — Food.com 선호도 순위 평가

- 동일한 적격 레시피 후보군에서 Content-based, Item-KNN, Hybrid를 비교합니다.
- train에서 평점 4~5를 준 레시피의 재료로 사용자 pantry proxy를 구성합니다.
- validation으로 Hybrid 가중치를 선택하고 test는 최종 평가에 한 번만 사용합니다.
- 사용자가 train 또는 validation에서 이미 본 레시피는 해당 평가 후보에서 제외합니다.
- 지표: Precision@10, Recall@10, NDCG@10

### Protocol B — 부분 재료 검색 평가

- 목표 레시피의 재료 중 50%, 70%, 90%만 입력해 원래 레시피를 찾는지 평가합니다.
- Food.com test와 RecipeNLG external catalog를 각각 평가합니다.
- 고정 시드 42·43·44로 반복합니다.
- 지표: Hit@10, MRR@10

## 6. 실험 결과

### Protocol A — Food.com 사용자 100명 표본

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|
| Content-based | 0.001000 | 0.000133 | 0.000851 |
| Item-KNN | 0.002000 | 0.002476 | 0.001969 |
| **Hybrid** | **0.002000** | **0.010133** | **0.003742** |

Validation에서 선택된 가중치는 Content-based 0.6, Item-KNN 0.4입니다. 100명 표본에서는 Hybrid가 가장 높은 Recall@10과 NDCG@10을 기록했습니다. 다만 절대 지표가 낮고 표본 평가이므로 전체 사용자에 대한 일반화 결과로 해석하지 않습니다.

### Protocol B — Partial-Ingredient Retrieval

| Dataset | Ingredient Retention | Hit@10 | MRR@10 |
|---|---:|---:|---:|
| Food.com Test | 50% | 0.946667 | 0.855001 |
| Food.com Test | 70% | 0.990000 | 0.962452 |
| Food.com Test | 90% | 0.990000 | 0.973286 |
| RecipeNLG External | 50% | 0.850000 | 0.664926 |
| RecipeNLG External | 70% | 0.980000 | 0.896884 |
| RecipeNLG External | 90% | 0.980000 | 0.932667 |

Food.com과 RecipeNLG 모두 입력 재료 비율이 높아질수록 검색 성능이 향상됐습니다. RecipeNLG 결과는 전체 약 220만 개가 아니라 약 100,000개 후보 표본과 목표 레시피 100개를 사용한 결과입니다.

## 7. 결과 해석과 한계

- Hybrid가 표본 평가에서 단일 모델보다 높은 순위 품질을 보였지만 전체 사용자 평가가 필요합니다.
- Protocol A의 pantry는 실제 냉장고 기록이 아니라 사용자가 좋아한 레시피의 재료로 만든 proxy입니다.
- RecipeNLG 평가는 전체 카탈로그가 아닌 후보 표본을 사용했습니다.
- RecipeNLG 미러와 공식 배포본의 완전한 동일성은 확인하지 못했습니다.
- RecipeNLG 중복 제거는 URL·제목·재료 지문의 exact match 기준이며 fuzzy near-duplicate는 남을 수 있습니다.
- 추천 결과는 데이터 기반 참고값이며 식품 안전이나 알레르기 적합성을 보장하지 않습니다.

## 8. Streamlit 데모 실행

Streamlit에 필요한 두 전처리 파일은 Git LFS로 관리합니다.

```text
data/processed/foodcom/recipes.parquet
data/processed/foodcom/splits/train.parquet
```

### 빠른 실행

```bash
# macOS에서 Git LFS가 없다면 먼저 설치
brew install git-lfs
git lfs install
git lfs pull
uv sync --all-extras
uv run streamlit run app.py
```

화면에서 다음 기능을 사용할 수 있습니다.

- 신규 사용자: Content-based 추천 및 이력 기반 방식 선택 시 cold-start fallback
- 기존 사용자: Content-based, Item-KNN, Hybrid 추천
- 입력 조건: 재료, Food.com 사용자 ID, 최대 조리 시간, 추천 개수
- 결과 설명: 추천 점수, 재료 충족률, 일치·부족 재료, 조리 시간, KNN 예측 선호도

## 9. 실험 재현

### 개발 환경

- Python 3.11 이상 3.14 미만
- 패키지 관리: [uv](https://docs.astral.sh/uv/)
- 주요 라이브러리: pandas, scikit-learn, PyArrow, Streamlit
- 테스트 및 품질 관리: pytest, Ruff, GitHub Actions

```bash
uv sync --all-extras
```

### 원본 데이터 배치

원본 데이터는 저장소에 포함하지 않습니다.

```text
data/raw/foodcom/RAW_recipes.csv
data/raw/foodcom/RAW_interactions.csv
data/raw/recipenlg/full_dataset.csv
```

Food.com은 [Kaggle 데이터셋](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)에서 내려받습니다. RecipeNLG는 데이터 이용 조건을 확인한 뒤 위 경로에 배치합니다.

### EDA와 전처리

```bash
uv run python -m fridge_rescue.data.eda --dataset all --max-rows 0
uv run python -m fridge_rescue.data.preprocessing --dataset all --max-rows 0
uv run python -m fridge_rescue.data.splitting
```

주요 산출물:

```text
data/processed/foodcom/recipes.parquet
data/processed/foodcom/splits/{train,validation,test}.parquet
data/processed/recipenlg/external_test_recipes.parquet
outputs/tables/preprocessing_report.json
outputs/tables/data_split_report.json
```

### Protocol A 실행

```bash
uv run python -m fridge_rescue.evaluation.run_evaluation \
  --users 100 \
  --k 10 \
  --neighbors 50 \
  --weight-step 0.1
```

`--users 0`을 사용하면 조건을 만족하는 전체 사용자를 평가할 수 있지만 실행 시간과 메모리 사용량이 크게 증가합니다.

```text
outputs/tables/model_comparison.csv
outputs/tables/hybrid_weight_search.csv
```

### Protocol B 실행

```bash
uv run python -m fridge_rescue.evaluation.run_protocol_b \
  --targets-per-dataset 100 \
  --recipenlg-candidates 100000
```

```text
outputs/tables/protocol_b_results.csv
outputs/tables/protocol_b_summary.csv
```

### 테스트와 코드 검사

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## 10. 프로젝트 구조

```text
Fridge-Rescue/
├── app.py                          # Streamlit 애플리케이션
├── configs/
│   ├── default.yaml                # 데이터 경로와 공통 설정
│   └── ingredient_synonyms.yaml    # 재료 동의어 사전
├── data/
│   ├── raw/                        # 원본 데이터, Git 제외
│   └── processed/                  # Streamlit용 2개 파일만 Git LFS
├── docs/images/                    # 시스템 아키텍처
├── outputs/                        # EDA·평가 결과, Git 제외
├── src/fridge_rescue/
│   ├── data/                       # 로더, EDA, 정규화, 분할
│   ├── evaluation/                 # Protocol A·B와 평가 지표
│   ├── recommenders/               # Content-based, Item-KNN, Hybrid
│   └── services/                   # UI와 추천 모델 연결 계층
└── tests/                          # 자동화 테스트
```

## 11. 구현 역할

| 역할 | 담당 내용 |
|---|---|
| 개발자 A | 프로젝트 설정·CI, 데이터 로더·EDA, 재료 정규화, 데이터 분할, Content-based, Streamlit UI |
| 개발자 B | Item-KNN, Hybrid, 가중치 탐색, Protocol A·B 평가, 결과 비교 |

## 12. 연구 참고 자료

- Majumder et al., [Generating Personalized Recipes from Historical User Preferences](https://aclanthology.org/D19-1613/) — Food.com 사용자 이력을 개인화 신호로 사용하는 배경
- Bień et al., [RecipeNLG: A Cooking Recipes Dataset for Semi-Structured Text Generation](https://aclanthology.org/2020.inlg-1.4/) — 외부 레시피 코퍼스와 NER 재료 데이터
- Çano and Morisio, [Hybrid Recommender Systems: A Systematic Literature Review](https://doi.org/10.3233/IDA-163209) — 콘텐츠·협업 신호의 weighted hybrid와 cold-start 설계 근거

본 프로젝트는 위 연구를 그대로 재현하지 않습니다. 기획 범위에 맞게 데이터 구성, 개인화 신호, 콘텐츠 기반 baseline과 weighted hybrid 설계의 근거로 활용했습니다.

## 데이터 공개 범위

- 원본 Food.com·RecipeNLG 데이터는 라이선스와 용량 문제로 Git에 올리지 않습니다.
- Streamlit 실행에 필요한 Food.com 전처리 파일 두 개만 Git LFS로 공유합니다.
- 나머지 전처리 결과, 모델 파일, 평가 산출물은 실행 명령으로 재생성합니다.
