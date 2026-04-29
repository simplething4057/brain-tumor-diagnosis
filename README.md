# Brain Tumor Diagnosis — Web Application

GLI(신경교종) / MEN(수막종) / MET(전이성 종양) 3종 분류 + NiiVue 3D MRI 뷰어 + RAG 방사선 소견 보고서 자동 생성

> ⚠️ **면책 조항**: 본 시스템은 연구 및 학습 목적의 AI 보조 도구입니다. 출력 결과는 공식 임상 진단을 대체하지 않으며, 실제 환자 진료에 단독으로 사용해서는 안 됩니다. 최종 진단은 반드시 자격을 갖춘 의료 전문가의 판단을 따르십시오.

---

## 목차

- [애플리케이션 화면](#애플리케이션-화면)
- [데모 영상](#데모-영상)
- [자동 생성 보고서 샘플 (PDF)](#자동-생성-보고서-샘플-pdf)
- [주요 기능](#주요-기능)
- [실행 방법](#실행-방법)
- [파일 업로드 형식](#파일-업로드-형식)
- [트러블슈팅](#트러블슈팅)
- [아키텍처](#아키텍처)
- [보고서 생성 기준](#보고서-생성-기준)
- [LLM 설정](#llm-설정)
- [API 엔드포인트](#api-엔드포인트)
- [세그멘테이션 레이블](#세그멘테이션-레이블-뷰어-색상-범례)
- [프로젝트 구조](#프로젝트-구조)
- [알려진 제약사항](#알려진-제약사항)
- [데이터 출처](#데이터-출처)
- [라이선스](#라이선스)

---

## 애플리케이션 화면

![Brain Tumor Diagnosis 애플리케이션](https://raw.githubusercontent.com/simplething4057/brain-tumor-3class/main/docs/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202026-04-19%20181337.png)

좌: MRI 파일 업로드 패널 · 중앙: NiiVue 3D 뷰어 (Multi / 3D 뷰 + 종양 히트맵) · 우: 분류 결과 + 방사선 소견 보고서

## 데모 영상

[▶ 영상 다운로드 (mp4)](https://github.com/simplething4057/brain-tumor-diagnosis/releases/download/v1.0.0/Brain_Tumor_Diagnosis_Final_2026-04-27_17-35-22.mp4)

> GitHub에서 인라인 재생이 지원되지 않습니다. 링크를 클릭하여 다운로드 후 재생해 주세요.

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

## 실행 방법

### 본인 PC (재실행 시)

이미 세팅이 완료된 상태라면 두 가지만 실행하면 됩니다.

```bash
# 터미널 1 — DB + 백엔드
docker compose up -d

# 터미널 2 — 프론트엔드
cd frontend
npm run dev
```

브라우저에서 `http://localhost:5173` 접속. Ollama는 Windows 백그라운드 서비스로 자동 실행됩니다.

| 서비스 | URL | 비고 |
|--------|-----|------|
| 프론트엔드 | `http://localhost:5173` | React 개발 서버 |
| 백엔드 API | `http://localhost:8000` | FastAPI |
| API 문서 (Swagger) | `http://localhost:8000/docs` | 자동 생성 |
| PostgreSQL | `localhost:5432` | DB 직접 접속 시 |

---

### 새 PC (최초 설치)

#### 사전 설치 항목

| 항목 | 버전 | 비고 |
|------|------|------|
| Docker Desktop | 최신 | WSL2 백엔드 권장 |
| Node.js | 18+ | 프론트엔드 빌드 |
| Ollama | 최신 | 보고서 생성용 LLM |

#### 1단계. 저장소 클론

```bash
git clone https://github.com/simplething4057/brain-tumor-diagnosis.git
cd brain-tumor-diagnosis
```

#### 2단계. WSL2 메모리 설정

Ollama LLM 실행을 위해 WSL2 메모리를 확보합니다.

`%USERPROFILE%\.wslconfig` 파일을 열거나 생성:

```ini
[wsl2]
memory=12GB
swap=4GB
```

저장 후 PowerShell에서:

```powershell
wsl --shutdown
# Docker Desktop 재시작
```

#### 3단계. 환경 변수 설정

```bash
cp .env.example .env
```

기본값 그대로 사용 가능합니다. Groq API를 쓰려면 `GROQ_API_KEY`에 키를 입력하고 `LLM_BACKEND=groq`로 변경합니다.

#### 4단계. Ollama 모델 설치

```bash
ollama pull gemma2:2b        # 권장: 1.6GB, WSL2 8GB 이상에서 동작
# ollama pull gemma2:latest  # 고품질: 5.4GB, WSL2 12GB 이상 필요
```

> 다운로드 중 TLS 오류가 발생하면 네트워크 문제입니다. 잠시 후 재시도하거나 다른 네트워크(핫스팟 등)로 전환 후 다시 실행하세요.

#### 5단계. Docker 실행

```bash
docker compose up -d
```

최초 실행 시 백엔드 이미지를 빌드합니다 (약 3~5분). ChromaDB에 WHO CNS 2021 문서가 자동으로 초기화되므로 첫 보고서 생성은 평소보다 느릴 수 있습니다.

#### 6단계. 프론트엔드 실행

```bash
cd frontend
npm install    # 최초 1회만 실행 (node_modules/ 생성)
npm run dev
```

> `node_modules`는 Git에 포함되지 않습니다. `npm install`이 `package.json`을 읽어 의존성을 자동으로 설치하므로, 클론 후 반드시 한 번 실행해야 합니다.

브라우저에서 `http://localhost:5173` 접속합니다.

---

## 파일 업로드 형식

| 파일 | 확장자 | 모드 | 설명 |
|------|--------|------|------|
| seg | `.nii` / `.nii.gz` | seg 모드 | BraTS 형식 세그멘테이션 마스크 |
| t1c | `.nii` / `.nii.gz` | no-seg 모드 | T1 Contrast-Enhanced (필수) |
| t1n | `.nii` / `.nii.gz` | no-seg 모드 | T1 Native |
| t2f | `.nii` / `.nii.gz` | no-seg 모드 | T2 FLAIR (필수) |
| t2w | `.nii` / `.nii.gz` | no-seg 모드 | T2 Weighted |

**seg 모드**: seg 파일 하나만으로 즉시 예측 가능.  
**no-seg 모드**: t1c + t2f 최소 조합 권장. 없는 채널은 인접 채널로 자동 대체.

> ⚠️ **no-seg 모드 소요 시간 경고**: 업로드한 MRI 4채널로부터 세그멘테이션 마스크를 자동 생성하는 과정(BraTS Docker × 3)이 포함되어 있습니다. GPU(CUDA) 환경에서는 수분 내 완료되지만, **GPU 없는 CPU 전용 환경에서는 30분~1시간 이상 소요될 수 있습니다.** 세그멘테이션 마스크를 이미 보유하고 있다면 seg 모드 사용을 권장합니다.

---

## 트러블슈팅

### Docker / 백엔드

**`docker compose up` 실행 후 백엔드가 계속 재시작됨**

백엔드가 DB 연결 전에 먼저 뜨는 경우입니다. `docker compose logs backend`로 오류를 확인하세요. DB `healthcheck`가 통과되면 자동으로 정상화됩니다. 수 초 내로 안정되지 않으면 다음을 시도하세요.

```bash
docker compose down
docker compose up -d
```

**`port 5432 is already in use` 오류**

호스트에 PostgreSQL이 별도로 설치되어 있는 경우입니다.

```bash
# 사용 중인 프로세스 확인
netstat -ano | findstr :5432
# docker-compose.yml의 ports를 "5433:5432" 로 변경
```

**`port 8000 is already in use` 오류**

다른 서비스가 8000번 포트를 점유하고 있습니다. `docker-compose.yml`에서 `"8001:8000"`으로 변경하고, `frontend/src/utils/api.js`의 baseURL도 동일하게 수정하세요.

**`.env` 변경 후 반영이 안 됨**

`docker compose restart`는 환경 변수를 다시 읽지 않습니다. 반드시 다음 명령을 사용해야 합니다.

```bash
docker compose up -d --force-recreate backend
```

---

### Ollama / LLM

**`ollama pull` 시 TLS handshake timeout**

네트워크 또는 방화벽 문제입니다.

```
Error: pull model manifest: TLS handshake timeout
```

잠시 후 재시도하거나, 다른 네트워크(모바일 핫스팟 등)로 전환 후 다시 실행하세요. VPN을 사용 중이라면 끄고 시도해보세요.

**`model requires more system memory than is available` (Ollama)**

WSL2 메모리 부족입니다.

```powershell
# %USERPROFILE%\.wslconfig 확인
Get-Content "$env:USERPROFILE\.wslconfig"
```

`memory=12GB`로 설정 후 `wsl --shutdown` → Docker Desktop 재시작. 즉시 해결이 필요하면 더 가벼운 모델로 전환:

```bash
ollama pull gemma2:2b
# .env: OLLAMA_MODEL=gemma2:2b
docker compose up -d --force-recreate backend
```

**보고서 생성 실패: `500 Internal Server Error` (Ollama)**

Ollama가 실행 중인지, 모델이 설치되어 있는지 확인합니다.

```bash
ollama list              # 설치된 모델 확인
ollama pull gemma2:2b    # 없으면 설치
```

Docker 컨테이너에서 호스트 Ollama에 접근하지 못하는 경우, `.env`의 `OLLAMA_BASE_URL`이 `http://host.docker.internal:11434`인지 확인하세요. `localhost`를 사용하면 컨테이너 내부로만 연결되어 항상 실패합니다.

**`[wsl2]` 섹션이 `.wslconfig`에 중복 생성됨**

`Add-Content`를 여러 번 실행한 경우 섹션이 중복될 수 있습니다. `Set-Content`로 전체 덮어쓰기 하세요.

```powershell
Set-Content "$env:USERPROFILE\.wslconfig" "[wsl2]`nmemory=12GB`nswap=4GB"
```

---

### ML 모델

**`모델 파일 없음: /ml_pipeline/models/weights/meta_classifier.pkl`**

`meta_classifier.pkl`은 GitHub에 포함되어 있습니다. 이 오류가 발생하면 `git pull`이 제대로 되지 않은 경우입니다.

```bash
git pull
docker compose restart backend
```

git에서 정상적으로 받아왔는데도 파일이 없다면 수동으로 재생성할 수 있습니다.

```bash
pip install scikit-learn pandas loguru requests
python retrain_classifier.py   # features.csv는 ml_pipeline/outputs/features/에 포함되어 있음
docker compose restart backend
```

**`Can't get attribute 'TumorTypeClassifier' on <module '__mp_main__'>`**

pickle에 잘못된 모듈 경로가 기록된 경우입니다. `retrain_classifier.py`로 재생성하면 해결됩니다.

```bash
pip install scikit-learn pandas loguru requests
python retrain_classifier.py
docker compose restart backend
```

**`Cannot work out file type` (nibabel 오류)**

업로드된 `.nii.gz` 파일 처리 중 확장자 인식 오류입니다. 최신 버전에서 수정되어 있습니다. `git pull` 후 이미지를 재빌드하세요.

```bash
git pull
docker compose up -d --build backend
```

---

### no-seg 모드

**결과가 매우 오래 걸리거나 나오지 않음 (CPU 환경)**

GPU(CUDA) 없이 실행 시 BraTS 세그멘테이션 컨테이너 1개당 10~30분, 3종 합산 최대 1시간 이상 소요됩니다. 폴링은 타임아웃 없이 계속 대기하므로 브라우저를 닫지 않고 기다리면 됩니다.

GPU 환경으로 전환하려면 `ml_pipeline/configs/config.yaml`을 수정하세요.

```yaml
inference:
  cuda_devices: "0"   # GPU ID (nvidia-smi로 확인)
```

**BraTS 첫 실행이 오래 걸림**

BraTS 알고리즘 Docker 이미지(수 GB)를 자동으로 다운로드합니다. `docker ps`로 진행 여부를 확인하세요.

**`docker.sock` 권한 오류**

no-seg 모드는 컨테이너 내에서 호스트 Docker 데몬에 접근합니다. `docker-compose.yml`에 소켓 마운트가 있는지 확인하세요.

```yaml
volumes:
  - /var/run/docker.sock:/var/run/docker.sock
```

클라우드 환경이나 rootless Docker 환경에서는 소켓 마운트가 제한되어 no-seg 모드가 동작하지 않을 수 있습니다.

---

### 프론트엔드

**`npm run dev` 실행 후 화면이 빈 화면이거나 API 오류**

백엔드가 아직 기동 중인 경우입니다. `docker compose logs backend`로 상태를 확인하고, 백엔드 로그에 `Application startup complete`가 뜬 후 새로고침하세요.

**`CORS` 오류 (브라우저 콘솔)**

`frontend/src/utils/api.js`의 baseURL이 백엔드 주소와 일치하는지 확인하세요 (기본값: `http://localhost:8000`).

**`npm install` 오류 또는 패키지 충돌**

Node.js 버전이 18 미만인 경우 발생할 수 있습니다.

```bash
node -v    # 18.x 이상인지 확인
npm cache clean --force
npm install
```

---

### Git

**`Pathspec is in submodule 'ml_pipeline'` 오류**

`ml_pipeline/`이 git 서브모듈로 등록된 경우입니다. 일반 폴더로 전환해야 합니다.

```bash
git rm --cached ml_pipeline
git add ml_pipeline
git commit -m "fix: convert ml_pipeline from submodule to regular directory"
git push
```

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

### seg 모드 vs no-seg 모드

**seg 모드**

사용자가 BraTS 형식의 세그멘테이션 마스크를 직접 제공합니다. 해당 파일을 GLI · MEN · MET 세 폴더에 동일하게 복사한 뒤 피처를 추출합니다.

> **한계**: 동일한 파일이 세 종양 유형 모두에 입력되어 GLI·MEN·MET 피처가 전부 동일해집니다. RF 모델이 학습된 데이터는 세 모델이 각각 다르게 세그멘테이션한 결과이므로, seg 모드에서는 절대 수치만으로 분류하게 되어 정확도가 낮아질 수 있습니다.

**no-seg 모드**

MRI 4채널을 업로드하면 BraTS 패키지가 Docker를 통해 종양 유형별 특화 알고리즘을 순차 실행합니다.

| 컨테이너 | 특화 대상 |
|----------|-----------|
| GLI 세그멘테이션 | 신경교종 패턴 |
| MEN 세그멘테이션 | 수막종 패턴 |
| MET 세그멘테이션 | 전이암 패턴 |

세 컨테이너가 동일한 MRI를 서로 다르게 세그멘테이션하며, 그 차이값이 GLI·MEN·MET를 구분하는 핵심 신호가 됩니다.

### 폴링 아키텍처

예측 요청 즉시 `job_id` 반환 → 프론트에서 3초 간격 폴링 → 완료 시 결과 표시. no-seg 모드의 장시간 처리에도 HTTP 연결이 끊기지 않으며, 서버 재시작 후에도 DB에서 결과를 복구합니다.

---

## 보고서 생성 기준

방사선 소견 보고서는 **RAG(Retrieval-Augmented Generation)** 방식으로 생성됩니다.

**1단계 — RAG 문헌 검색**: ChromaDB에 색인된 WHO CNS Tumors 2021 가이드라인에서 예측 종양 유형 관련 내용을 검색해 LLM 프롬프트의 참고 문헌으로 제공합니다.

**2단계 — 정량적 피처 해석**: RF 모델이 산출한 수치를 아래 기준으로 임상 단서 문장으로 변환합니다.

| 수치 | 기준 | 임상 해석 |
|------|------|-----------|
| ET ratio | > 30% | 조영증강 뚜렷, 혈뇌장벽 파괴 시사 |
| ET ratio | 10~30% | 중등도 조영증강 |
| ET ratio | < 10% | 저등급 또는 비조영증강 종양 가능성 |
| edema ratio | > 50% | 광범위 부종, 침윤성 성장 또는 전이 시사 |
| core ratio | > 20% | 괴사 중심부 뚜렷, 고등급 교종 의심 |
| 병변 수 | > 1 | 다발성 전이 또는 다소성 교종 감별 필요 |
| 종양 부피 | > 30cm³ | 수술적 접근 우선 고려 |

**3단계 — LLM 보고서 작성**: 위 내용을 바탕으로 LLM이 4섹션 형식의 한국어 보고서를 생성합니다.

| 섹션 | 내용 |
|------|------|
| [촬영 정보] | 촬영 시퀀스 및 세그멘테이션 방법 명시 |
| [임상 증상] | 예측 종양 유형의 전형적 임상 증상 서술 |
| [MRI 소견] | 종양 크기·형태·조영증강·부종·괴사를 수치 근거로 상세 서술 |
| [결론] | 최종 진단, 감별 진단, 추가 검사 및 치료 권고 |

---

## LLM 설정

`.env`의 `LLM_BACKEND` 변경으로 전환합니다.

```env
# 로컬 Ollama (기본) — 인터넷 불필요, 메모리 필요
LLM_BACKEND=ollama
OLLAMA_MODEL=gemma2:2b

# 클라우드 Groq — 빠른 응답, 고품질 (API 키 필요)
LLM_BACKEND=groq
GROQ_API_KEY=gsk_xxxx
```

| 모델 | 크기 | 보고서 품질 | 필요 메모리 |
|------|------|------------|------------|
| `gemma2:2b` | 1.6GB | 기본 수준, 간결 | WSL2 8GB |
| `gemma2:latest` | 5.4GB | 풍부한 서술, 임상 표현 우수 | WSL2 12GB |
| Groq `llama3-70b` | 클라우드 | 최고 품질 | 로컬 메모리 불필요 |

---

## API 엔드포인트

> **Base URL**: `http://localhost:8000`

| Method | Path | 설명 |
|--------|------|------|
| POST | `/api/predict` | 파일 업로드 → 백그라운드 예측 시작, `job_id` 반환 |
| GET | `/api/predict/status/{job_id}` | 예측 진행 상태 폴링 |
| GET | `/api/files/{id}/{modality}` | NiiVue용 `.nii/.nii.gz` 파일 서빙 |
| POST | `/api/report/{record_id}` | RAG 보고서 생성 |
| GET | `/api/history` | 예측 이력 목록 |
| GET | `/api/history/{id}` | 단건 조회 |
| GET | `/health` | 헬스체크 |

---

## 세그멘테이션 레이블 (뷰어 색상 범례)

| 색상 | 레이블 | 의미 |
|------|--------|------|
| 🔴 진빨강 `#992200` | NCR | 괴사 중심부 (Necrotic Core) |
| 🟠 주황 `#cc6600` | ED | 주변 부종 (Edema) |
| 🟡 노랑 `#ffee00` | ET | 조영 증강 종양 (Enhancing Tumor) |

BraTS 2023 표준 레이블 체계를 따릅니다 (Label 1: NCR, 2: ED, 3: ET).

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
│       │   ├── NiiVueViewer.jsx  WebGL 3D MRI 뷰어
│       │   ├── FileUploader.jsx  드래그앤드롭 업로드
│       │   ├── ResultPanel.jsx   예측 결과 + RAG 보고서 + PDF 내보내기
│       │   └── HistoryPanel.jsx  예측 이력
│       └── utils/api.js         Axios + 폴링 클라이언트
├── ml_pipeline/                 ML 파이프라인
│   ├── src/
│   │   ├── classifier/
│   │   │   ├── meta_classifier.py   RandomForest 분류기 클래스
│   │   │   └── feature_extractor.py 세그멘테이션 → 21차원 피처 추출
│   │   └── inference/
│   │       └── brats_infer.py       BraTS Docker 세그멘테이션 연동
│   ├── models/weights/
│   │   └── meta_classifier.pkl      RF 모델 가중치
│   ├── outputs/features/
│   │   └── features.csv             학습 데이터 (BraTS 2023 + UCSF)
│   ├── configs/config.yaml          GPU/CPU 설정
│   └── main.py                      학습 파이프라인 진입점
├── retrain_classifier.py        meta_classifier.pkl 재학습 스크립트
├── docker-compose.yml
├── .env.example                 환경 변수 템플릿
└── .env                         실제 환경 변수 (git 제외)
```

---

## 알려진 제약사항

**seg 모드 분류 정확도**: 동일 seg 파일이 GLI/MEN/MET 모두에 입력되어 no-seg 모드 대비 변별력이 낮을 수 있습니다.

**no-seg 모드 소요 시간**: GPU 없는 CPU 전용 환경에서 세그멘테이션 3종 합산 30분~1시간 이상 소요됩니다.

**BraTS 초기 실행**: no-seg 첫 실행 시 BraTS Docker 이미지(수 GB)를 자동 다운로드합니다.

**LLM 메모리**: `gemma2:2b` 1.6GB / `gemma2:latest` 6.4GB 필요. WSL2 메모리 12GB 이상 권장.

**Docker-in-Docker**: no-seg 모드는 호스트 Docker 소켓이 필요하므로 일부 클라우드 환경에서 동작하지 않을 수 있습니다.

**보고서 품질**: 사용 LLM 크기에 따라 서술 품질이 달라집니다. 임상 표현의 완성도를 높이려면 `gemma2:latest` 또는 Groq API 사용을 권장합니다.

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

## 라이선스

본 프로젝트는 **[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)** (Creative Commons Attribution-NonCommercial 4.0 International) 라이선스를 따릅니다.

- 출처 표기 조건으로 학술·교육 목적의 자유로운 사용 허용
- **상업적 사용 금지**
- 동일 라이선스 조건 하에 개작 및 재배포 허용
