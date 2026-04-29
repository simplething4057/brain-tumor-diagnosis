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
  - **seg 모드**: 세그멘테이션 마스크(seg.nii.gz) 업로드 → 즉시 분류 (수초)
  - **no-seg 모드**: MRI 4채널(T1C/T1N/T2F/T2W) 업로드 → BraTS Docker 자동 세그멘테이션 → 분류 (GPU: 수분 / CPU: 30분 이상)
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
[seg 모드]  seg.nii.gz → GLI/MEN/MET 폴더에 동일 파일 복사
[no-seg]   BraTS Docker × 3 → GLI/MEN/MET 각각 다른 세그멘테이션 생성
    ↓
feature_extractor → 21차원 피처 (부피·ET비율·부종비율·괴사비율·병변수 × 3종)
    ↓
RandomForest meta_classifier.pkl → GLI / MEN / MET 분류 + 확률
    ↓
RAG (ChromaDB WHO CNS 문서 검색) + Ollama LLM → 방사선 소견 보고서
```

### seg 모드 vs no-seg 모드 상세

**seg 모드**

사용자가 BraTS 형식의 세그멘테이션 마스크(`.nii` / `.nii.gz`)를 직접 제공합니다. 시스템은 해당 파일을 GLI · MEN · MET 세 폴더에 동일하게 복사한 뒤 `feature_extractor`로 7개 수치(총 부피, ET비율, 부종비율, 괴사비율, 병변 수, 종양 유무)를 각각 추출해 21차원 피처를 구성합니다.

> **한계**: 동일한 파일이 세 종양 유형 모두에 입력되어 GLI·MEN·MET 피처가 전부 동일해집니다. RF 모델이 학습된 데이터는 세 모델이 각각 다르게 세그멘테이션한 결과이므로, seg 모드에서는 절대 수치만으로 분류하게 되어 정확도가 낮아질 수 있습니다.

**no-seg 모드**

MRI 4채널(T1C · T1N · T2F · T2W)을 업로드하면 BraTS 패키지가 Docker를 통해 종양 유형별 특화 알고리즘(BraTS23_1)을 순차 실행합니다.

| 컨테이너 | 알고리즘 | 특화 대상 |
|----------|----------|-----------|
| GLI 세그멘테이션 | BraTS23_1 | 신경교종 패턴 |
| MEN 세그멘테이션 | BraTS23_1 | 수막종 패턴 |
| MET 세그멘테이션 | BraTS23_1 | 전이암 패턴 |

세 컨테이너가 동일한 MRI를 서로 다르게 세그멘테이션하며, 그 차이값이 GLI·MEN·MET를 구분하는 핵심 신호가 됩니다. RF 모델 학습 데이터와 동일한 구조이므로 분류 정확도가 더 높습니다.

**GPU 환경 영향**

BraTS Docker 알고리즘은 CUDA GPU에 최적화되어 있습니다. GPU 없이 CPU만 사용하는 환경에서는 세그멘테이션 컨테이너 1개당 10~30분, 3개 합산 최소 30분, 길게는 1시간 이상 소요될 수 있습니다. GPU가 있는 경우 `configs/config.yaml`에서 `cuda_devices: "0"`으로 변경하면 수분 내로 완료됩니다.

```yaml
# configs/config.yaml
inference:
  cuda_devices: ""   # CPU 전용 (기본값)
  # cuda_devices: "0"  # GPU 사용 시
```

### 폴링 아키텍처

예측 요청 즉시 `job_id` 반환 → 프론트에서 3초 간격 폴링 → 완료 시 결과 표시. no-seg 모드의 장시간 처리에도 HTTP 연결이 끊기지 않으며, 서버 재시작 후에도 DB에서 결과를 복구합니다.

### no-seg 모드: Docker-in-Docker 구조

BraTS 패키지가 내부적으로 Docker CLI를 호출해 세그멘테이션 컨테이너를 실행합니다. 이 때문에 다음 설정이 필수입니다.

```yaml
# docker-compose.yml
volumes:
  - /var/run/docker.sock:/var/run/docker.sock  # 호스트 Docker 데몬 접근
```

```dockerfile
# backend/Dockerfile
RUN apt-get install -y docker.io  # Docker CLI 설치
```

클라우드 배포 환경이나 rootless Docker 환경에서는 소켓 마운트가 제한될 수 있습니다.

---

## 보고서 생성 기준

방사선 소견 보고서는 **RAG(Retrieval-Augmented Generation)** 방식으로 생성됩니다.

### 1단계 — RAG 문헌 검색

ChromaDB에 색인된 WHO CNS Tumors 2021 가이드라인 문서에서 예측된 종양 유형(GLI·MEN·MET) 관련 내용 2건을 검색해 LLM 프롬프트의 참고 문헌으로 제공합니다.

### 2단계 — 정량적 피처 임상 해석

RF 모델이 산출한 수치를 아래 기준으로 임상 단서 문장으로 변환합니다.

| 수치 | 기준 | 임상 해석 |
|------|------|-----------|
| ET ratio | > 30% | 조영증강 뚜렷, 혈뇌장벽 파괴 시사 |
| ET ratio | 10~30% | 중등도 조영증강 |
| ET ratio | < 10% | 저등급 또는 비조영증강 종양 가능성 |
| edema ratio | > 50% | 광범위 부종, 침윤성 성장 또는 전이 시사 |
| core ratio | > 20% | 괴사 중심부 뚜렷, 고등급 교종 의심 |
| 병변 수 | > 1 | 다발성 전이 또는 다소성 교종 감별 필요 |
| 종양 부피 | > 30cm³ | 수술적 접근 우선 고려 |

종양 유형별 전형적 MRI 신호 패턴(GLI: T2/FLAIR 고신호·불균일 ring enhancement / MEN: T1C 균일 강한 조영증강·dural tail sign / MET: 피질-수질 경계부 ring enhancement·주변 불균형 부종)도 단서로 추가됩니다.

### 3단계 — LLM 보고서 작성

위 참고 문헌과 임상 단서를 바탕으로 LLM이 4섹션 형식의 한국어 보고서를 생성합니다.

| 섹션 | 내용 |
|------|------|
| [촬영 정보] | 촬영 시퀀스 및 세그멘테이션 방법 명시 |
| [임상 증상] | 예측 종양 유형의 전형적 임상 증상 서술 |
| [MRI 소견] | 종양 크기·형태·조영증강·부종·괴사를 수치 근거로 상세 서술 |
| [결론] | 최종 진단, 감별 진단, 추가 검사 및 치료 권고 |

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
pip install scikit-learn pandas requests
python retrain_classifier.py
docker compose restart backend
```

### `Can't get attribute 'TumorTypeClassifier' on <module '__mp_main__'>`

pickle 저장 시 모듈 경로가 잘못 기록된 경우입니다. `backend/pipeline/predict.py`의 `_CrossPlatformUnpickler`에서 자동으로 remapping하므로, 이 오류가 발생하면 백엔드가 구버전입니다. `git pull` 후 재시작하세요.

### `model requires more system memory (6.4 GiB) than is available` (Ollama)

WSL2 메모리가 부족합니다. `%USERPROFILE%\.wslconfig`에서 `memory=12GB`로 설정 후 `wsl --shutdown` → Docker Desktop 재시작.

즉시 해결이 필요하면 가벼운 모델로 전환:
```bash
ollama pull gemma2:2b
# .env: OLLAMA_MODEL=gemma2:2b
docker compose restart backend
```

### `Cannot work out file type` (nibabel 오류)

`.nii.gz` 파일이 `.nii`로 저장될 때 발생합니다. 현재 버전은 원본 확장자를 자동 보존하므로, 이 오류가 발생하면 백엔드가 구버전입니다. `git pull` 후 재시작하세요.

### no-seg 모드에서 결과가 나오지 않음 / 매우 오래 걸림

**BraTS Docker 이미지가 없는 경우** — 첫 실행 시 BraTS 알고리즘 이미지를 자동으로 pull합니다. 이미지 크기가 수 GB이므로 초기 다운로드에 수십 분이 소요될 수 있습니다. `docker ps`로 다운로드 진행 여부를 확인하세요.

**CPU 전용 환경** — GPU(CUDA) 없이 실행 시 세그멘테이션 컨테이너 1개당 10~30분, 3종 합산 최대 1시간 이상 소요됩니다. 폴링은 타임아웃 없이 계속 대기하므로 브라우저를 닫지 않고 기다리면 됩니다.

**GPU 환경으로 전환** — `ml_pipeline/configs/config.yaml`에서 설정 변경:
```yaml
inference:
  cuda_devices: "0"  # GPU ID (nvidia-smi로 확인)
```

**Docker 소켓 마운트 누락** — `docker-compose.yml`에 `/var/run/docker.sock` 마운트가 있는지 확인하세요. 없으면 BraTS Docker 컨테이너를 실행할 수 없습니다.

### 보고서 생성 실패 (LLM 500 오류)

Ollama가 실행 중인지, 모델이 설치되어 있는지 확인:
```bash
ollama list          # 설치된 모델 확인
ollama pull gemma2:2b  # 없으면 설치
```

Docker 컨테이너에서 호스트 Ollama로 접근이 안 되는 경우 `.env`의 `OLLAMA_BASE_URL`이 `http://host.docker.internal:11434`인지 확인하세요 (`localhost` 사용 시 컨테이너 내부로 연결되어 실패).

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
- **no-seg 모드 소요 시간**: GPU 없는 CPU 전용 환경에서 세그멘테이션 3종 합산 30분~1시간 이상 소요됩니다. GPU 환경(CUDA)에서는 수분 내 완료됩니다.
- **BraTS 초기 실행**: no-seg 첫 실행 시 BraTS Docker 이미지(수 GB)를 자동 다운로드하므로 초기 대기 시간이 발생합니다.
- **LLM 메모리**: `gemma2:2b` 1.6GB / `gemma2:latest` 6.4GB 필요. WSL2 메모리 12GB 이상 권장.
- **Docker-in-Docker**: no-seg 모드는 호스트 Docker 소켓이 필요하므로 일부 클라우드 환경에서 동작하지 않을 수 있습니다.
- **보고서 품질**: 사용 LLM 크기에 따라 보고서 서술 품질이 달라집니다. 임상 표현의 완성도를 높이려면 `gemma2:latest` 또는 Groq API 사용을 권장합니다.
