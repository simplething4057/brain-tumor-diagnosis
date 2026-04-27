# Brain Tumor Diagnosis — Web Application

GLI(신경교종) / MEN(수막종) / MET(전이성 종양) 3종 분류 + NiiVue 3D MRI 뷰어 + RAG 방사선 소견 보고서 자동 생성

> **⚠️ 면책 조항**: 본 시스템은 연구 및 학습 목적의 AI 보조 도구입니다. 출력 결과는 공식 임상 진단을 대체하지 않으며, 실제 환자 진료에 단독으로 사용해서는 안 됩니다. 최종 진단은 반드시 자격을 갖춘 의료 전문가의 판단을 따르십시오.

---

## 애플리케이션 화면

![Brain Tumor Diagnosis 애플리케이션](https://raw.githubusercontent.com/simplething4057/brain-tumor-3class/main/docs/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202026-04-19%20181337.png)

좌: MRI 파일 업로드 패널 · 중앙: NiiVue 3D 뷰어 (Multi / 3D 뷰 + 종양 히트맵) · 우: 분류 결과 + 방사선 소견 보고서

## 데모 영상

https://github.com/simplething4057/brain-tumor-diagnosis/releases/download/v1.0.0/Brain_Tumor_Diagnosis_Final_2026-04-27_17-35-22.mp4

## 자동 생성 보고서 샘플 (PDF)

[sample-report.pdf](https://github.com/simplething4057/brain-tumor-3class/raw/main/docs/sample-report.pdf)

GLI(신경교종) 케이스의 **[촬영 정보] → [임상 증상] → [MRI 소견] → [결론]** 형식 보고서 및 3D 영상 포함 2페이지 PDF 출력물입니다.

---

## 주요 기능

- **MRI 파일 업로드**: `.nii` / `.nii.gz` 모두 지원
- **2가지 예측 모드**
  - **seg 모드**: 세그멘테이션 파일(seg.nii.gz) 업로드 → 즉시 분류 (수초)
  - **no-seg 모드**: MRI 4채널(T1C/T1N/T2F/T2W) 업로드 → BraTS Docker 자동 세그멘테이션 후 분류 (수십 분)
- **NiiVue 3D 뷰어**: Axial / Multi / 3D 뷰, 모달리티 전환, 세그멘테이션 오버레이
- **RAG 보고서 생성**: WHO CNS 2021 가이드라인 기반 ChromaDB + Ollama LLM
- **PDF 내보내기**: MRI 3D 영상(1페이지) + 방사선 소견 보고서(2페이지)
- **예측 이력**: PostgreSQL 저장 및 이력 조회

---

## 아키텍처

```
frontend/          React + Vite + Tailwind + NiiVue (WebGL 3D 뷰어)
backend/           FastAPI + SQLAlchemy(asyncpg) + ChromaDB
ml_pipeline/       ML 파이프라인 (RandomForest 분류기 + BraTS 추론)
docker-compose.yml PostgreSQL 15 + FastAPI 백엔드
```

### 예측 파이프라인

```
파일 업로드 (seg or MRI 4채널)
    ↓
[seg 모드]  seg.nii.gz → GLI/MEN/MET 폴더 복사
[no-seg]   BraTS Docker → GLI/MEN/MET 세그멘테이션 자동 생성
    ↓
feature_extractor → 21차원 피처 (부피·ET비율·부종비율·괴사비율 × 3종)
    ↓
RandomForest meta_classifier.pkl → GLI / MEN / MET 분류 + 확률
    ↓
RAG (ChromaDB WHO CNS 문서 검색) + Ollama LLM → 방사선 소견 보고서
```

### 폴링 아키텍처

예측 요청 즉시 `job_id` 반환 → 프론트에서 3초 간격 폴링 → 완료 시 결과 표시. 30분 이상 소요되는 no-seg 모드에도 HTTP 연결이 끊기지 않습니다.

### no-seg 모드: Docker-in-Docker 구조

no-seg 모드에서는 BraTS 패키지가 내부적으로 Docker CLI를 호출해 GLI / MEN / MET 세그멘테이션 컨테이너를 실행합니다. 이 때문에 다음 설정이 필수입니다.

```yaml
# docker-compose.yml
volumes:
  - /var/run/docker.sock:/var/run/docker.sock  # 호스트 Docker 데몬 접근
```

```dockerfile
# backend/Dockerfile
RUN apt-get install -y docker.io  # Docker CLI 설치
```

클라우드 배포 환경이나 rootless Docker 환경에서는 소켓 마운트가 제한될 수 있으므로 주의하십시오.

---

## 세그멘테이션 레이블 (뷰어 색상 범례)

| 색상 | 레이블 | 의미 |
|------|--------|------|
| 🔴 진빨강 `#992200` | NCR | 괴사 중심부 (Necrotic Core) |
| 🟠 주황 `#cc6600` | ED | 주변 부종 (Edema) |
| 🟡 노랑 `#ffee00` | ET | 조영 증강 종양 (Enhancing Tumor) |

BraTS 2023 표준 레이블 체계를 따릅니다 (Label 1: NCR, 2: ED, 3: ET).

---

## 빠른 시작

### 사전 요구사항

| 항목 | 버전 | 비고 |
|------|------|------|
| Docker Desktop | 최신 | WSL2 백엔드 권장 |
| Node.js | 18+ | 프론트엔드 |
| Ollama | 최신 | 보고서 생성 (선택) |
| WSL2 메모리 | 12GB 이상 | LLM 실행 시 필수 |

### 1. WSL2 메모리 설정 (LLM 사용 시)

`%USERPROFILE%\.wslconfig` 파일 생성 또는 수정:

```ini
[wsl2]
memory=12GB
swap=4GB
```

저장 후 `wsl --shutdown` 실행, Docker Desktop 재시작.

### 2. ML 모델 준비

ML 파이프라인 코드는 `ml_pipeline/`에 포함되어 있습니다. 모델 가중치(`meta_classifier.pkl`)는 git에서 제외되므로 최초 1회 재학습이 필요합니다.

```bash
pip install scikit-learn pandas requests
python retrain_classifier.py
# → ml_pipeline/models/weights/meta_classifier.pkl 생성 (약 5초 소요)
```

### 3. Ollama 모델 설치

```bash
ollama pull gemma2:2b       # 권장: 1.6GB, WSL2 8GB 이상에서 동작
# ollama pull gemma2:latest # 고품질: 5.4GB, WSL2 12GB 이상 필요 (보고서 품질 더 우수)
```

### 4. 환경 변수 설정

```bash
cp .env.example .env
```

`.env` 전체 항목:

```env
# PostgreSQL
POSTGRES_DB=brain_tumor
POSTGRES_USER=btuser
POSTGRES_PASSWORD=btpass

# LLM
LLM_BACKEND=ollama               # "ollama" | "groq"
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=gemma2:2b           # gemma2:2b (경량) | gemma2:latest (고품질)
GROQ_API_KEY=                    # Groq 사용 시 입력

# 경로 (Docker 컨테이너 내부 기준, 수정 불필요)
ML_PIPELINE_PATH=/ml_pipeline    # ./ml_pipeline 볼륨 마운트 경로
UPLOAD_DIR=/app/uploads          # 업로드 파일 저장 경로
CHROMA_DIR=/app/chroma_db        # ChromaDB 벡터 저장 경로
```

### 5. Docker 실행

```bash
docker compose up -d
```

> **첫 실행 시**: ChromaDB에 WHO CNS 2021 가이드라인 문서가 자동으로 로드됩니다 (수십 초 소요). 첫 보고서 생성이 평소보다 느릴 수 있습니다.

### 6. 프론트엔드 실행

```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

---

## 파일 업로드 형식

| 파일 | 확장자 | 모드 | 설명 |
|------|--------|------|------|
| seg | `.nii` / `.nii.gz` | seg 모드 | BraTS 형식 세그멘테이션 마스크 |
| t1c | `.nii` / `.nii.gz` | no-seg 모드 | T1 Contrast-Enhanced (필수) |
| t1n | `.nii` / `.nii.gz` | no-seg 모드 | T1 Native |
| t2f | `.nii` / `.nii.gz` | no-seg 모드 | T2 FLAIR (필수) |
| t2w | `.nii` / `.nii.gz` | no-seg 모드 | T2 Weighted |

**seg 모드**: seg 파일만 있으면 즉시 예측 가능.  
**no-seg 모드**: t1c + t2f 최소 조합 권장. 없는 채널은 인접 채널로 자동 대체.

---

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| POST | `/api/predict` | 파일 업로드 → 백그라운드 예측 시작, `job_id` 즉시 반환 |
| GET | `/api/predict/status/{job_id}` | 예측 진행 상태 폴링 |
| GET | `/api/files/{id}/{modality}` | NiiVue용 .nii/.nii.gz 서빙 |
| POST | `/api/report/{record_id}` | RAG 보고서 생성 |
| GET | `/api/history` | 예측 이력 목록 |
| GET | `/api/history/{id}` | 단건 조회 |
| GET | `/health` | 헬스체크 |

---

## LLM 전환

`.env`의 `LLM_BACKEND` 변경:

```env
# 로컬 Ollama (기본) — 인터넷 불필요, 메모리 필요
LLM_BACKEND=ollama
OLLAMA_MODEL=gemma2:2b

# 클라우드 Groq — 빠른 응답, 고품질 (llama3-70b 사용)
LLM_BACKEND=groq
GROQ_API_KEY=gsk_xxxx
```

**모델별 보고서 품질 비교**

| 모델 | 크기 | 보고서 품질 | 필요 메모리 |
|------|------|------------|------------|
| `gemma2:2b` | 1.6GB | 기본 수준, 간결 | WSL2 8GB |
| `gemma2:latest` | 5.4GB | 풍부한 서술, 임상 표현 우수 | WSL2 12GB |
| Groq `llama3-70b` | 클라우드 | 최고 품질 | 로컬 메모리 불필요 |

---

## 트러블슈팅

### `모델 파일 없음: /ml_pipeline/models/weights/meta_classifier.pkl`

모델 가중치는 git에서 제외되어 있습니다. 최초 1회 재학습이 필요합니다.

```bash
# 재학습 (features.csv는 ml_pipeline/outputs/features/ 또는 GitHub에서 자동 다운로드)
pip install scikit-learn pandas requests
python retrain_classifier.py

# 백엔드 재시작
docker compose restart backend
```

### `Can't get attribute 'TumorTypeClassifier' on <module '__mp_main__'>`

pickle 저장 시 모듈 경로가 잘못 기록된 경우입니다. `backend/pipeline/predict.py`의 `_CrossPlatformUnpickler`에서 자동 처리됩니다. 이 오류가 발생하면 백엔드 버전이 구버전일 수 있으므로 `git pull` 후 재시작하세요.

### `model requires more system memory (6.4 GiB) than is available`

WSL2 메모리가 부족합니다. `%USERPROFILE%\.wslconfig`에서 `memory=12GB`로 설정 후 `wsl --shutdown` → Docker Desktop 재시작.

또는 가벼운 모델로 전환:
```bash
ollama pull gemma2:2b
# .env: OLLAMA_MODEL=gemma2:2b
docker compose restart backend
```

### `Cannot work out file type` (nibabel 오류)

`.nii.gz` 파일을 `.nii`로 저장하면 발생합니다. 현재 버전은 원본 확장자를 자동 보존하므로, 이 오류가 발생하면 백엔드가 구버전입니다.

### no-seg 모드에서 결과가 나오지 않음

1. `docker ps`로 BraTS Docker 컨테이너가 실행 중인지 확인
2. `/var/run/docker.sock` 마운트 여부 확인 (`docker-compose.yml`)
3. CPU 전용 환경에서는 30분 이상 소요되므로 폴링 타임아웃 없이 대기

---

## 프로젝트 구조

```
Brain-tumor-diagnosis/
├── backend/
│   ├── main.py                  FastAPI 진입점
│   ├── api/routes.py            REST API 라우터 (폴링 아키텍처)
│   ├── pipeline/predict.py      RF 예측 파이프라인 (seg/no-seg)
│   ├── rag/
│   │   ├── vector_store.py      ChromaDB + WHO CNS 2021 문서
│   │   └── report_generator.py  LLM 보고서 생성 (Ollama/Groq)
│   ├── db/
│   │   ├── database.py          AsyncPG 세션
│   │   ├── models.py            SQLAlchemy 모델
│   │   └── crud.py              DB CRUD
│   ├── core/config.py           pydantic-settings 환경설정
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── App.jsx              메인 레이아웃 + 폴링 로직
│       ├── components/
│       │   ├── NiiVueViewer.jsx  WebGL 3D MRI 뷰어 (preserveDrawingBuffer 패치)
│       │   ├── FileUploader.jsx  드래그앤드롭 업로드
│       │   ├── ResultPanel.jsx   예측 결과 + RAG 보고서 + PDF 내보내기
│       │   └── HistoryPanel.jsx  예측 이력
│       └── utils/api.js         Axios + 폴링 클라이언트
├── ml_pipeline/                 ML 파이프라인 (brain-tumor-3class)
│   ├── src/
│   │   ├── classifier/
│   │   │   ├── meta_classifier.py   RandomForest 분류기 클래스
│   │   │   └── feature_extractor.py 세그멘테이션 → 21차원 피처 추출
│   │   ├── inference/
│   │   │   └── brats_infer.py       BraTS Docker 세그멘테이션 연동
│   │   └── preprocessing/
│   ├── models/weights/
│   │   └── meta_classifier.pkl      ← git 제외 (retrain_classifier.py로 생성)
│   ├── outputs/features/
│   │   └── features.csv             학습 데이터 (BraTS 2023 + UCSF)
│   ├── configs/config.yaml          학습 설정
│   ├── scripts/                     전처리·추출 유틸리티
│   └── main.py                      학습 파이프라인 진입점
├── retrain_classifier.py        meta_classifier.pkl 재학습 스크립트
├── docker-compose.yml
└── .env
```

---

## 데이터 출처

| 데이터셋 | 종양 유형 | 출처 |
|----------|-----------|------|
| BraTS 2023 GLI | 신경교종 (GLI) | [Synapse](https://www.synapse.org/#!Synapse:syn51156910) |
| BraTS 2023 MEN | 수막종 (MEN) | [Synapse](https://www.synapse.org/#!Synapse:syn51156910) |
| UCSF Brain Metastases | 전이성 종양 (MET) | [TCIA](https://www.cancerimagingarchive.net/) |
| WHO CNS Tumors 2021 | RAG 문헌 | WHO Classification of Tumours 5th Ed. |

본 프로젝트는 학술 연구 목적으로 제작되었으며 상업적 사용을 금합니다.

---

## 알려진 제약사항

- **seg 모드 분류 정확도**: 동일 seg 파일이 GLI/MEN/MET 모두에 입력되어 no-seg 모드 대비 변별력이 낮을 수 있습니다.
- **no-seg 모드 소요 시간**: CPU 전용 환경에서 30분 이상 소요됩니다.
- **LLM 메모리**: `gemma2:2b` 1.6GB / `gemma2:latest` 6.4GB 필요. WSL2 메모리 설정 필수.
- **Docker-in-Docker**: no-seg 모드는 호스트 Docker 소켓이 필요하므로 일부 클라우드 환경에서 동작하지 않을 수 있습니다.
