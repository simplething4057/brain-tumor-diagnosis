# 뇌종양 변별 AI 파이프라인
Brain Tumor Classification Pipeline

## 프로젝트 개요
BraTS 2023 데이터셋 기반 3종 뇌종양(Glioma / Meningioma / Metastases) 자동 변별 및 RAG 기반 소견문 생성

## 기술 스택
- **모델**: nnDetection / MONAI 3D U-Net / SegMamba (SSM)
- **LLM**: Ollama (개발) / Groq API (배포)
- **백엔드**: FastAPI + PostgreSQL
- **프론트엔드**: React + Vite + Tailwind CSS
- **인프라**: Docker Compose

---

## 브랜치 전략

| 브랜치 | 용도 |
|--------|------|
| `main` | 최종 통합본 (항상 동작하는 상태 유지) |
| `dev` | 통합 개발 브랜치 |
| `feature/nndetection` | nnDetection 실험 |
| `feature/monai` | MONAI 3D U-Net 실험 |
| `feature/segmamba` | SegMamba 실험 |
| `feature/backend` | FastAPI 백엔드 개발 |
| `feature/frontend` | React 프론트엔드 개발 |

### 작업 규칙
- `feature/*` → `dev` PR 후 머지
- `dev` → `main` 은 최종 완성 시에만
- 커밋 메시지 형식: `[모델명] 작업내용`
  - 예: `[MONAI] 100케이스 50에폭 학습 완료`
  - 예: `[Backend] /predict 엔드포인트 구현`

---

## 모델 가중치 관리

모델 가중치 파일(`.pth`, `.pt`, `.pkl`)은 용량 문제로 **GitHub에 포함하지 않습니다.**
가중치는 Hugging Face Hub에 업로드하여 관리하며, 서버 시작 시 자동으로 다운로드됩니다.

### 학습 완료 후 업로드 방법 (Kaggle → HuggingFace)
```python
from huggingface_hub import upload_file

# MONAI 가중치 업로드
upload_file(
    path_or_fileobj="monai_best.pth",
    path_in_repo="monai_best.pth",
    repo_id="your-username/brain-tumor-models",
    repo_type="model"
)

# SegMamba 가중치 업로드
upload_file(
    path_or_fileobj="segmamba_best.pth",
    path_in_repo="segmamba_best.pth",
    repo_id="your-username/brain-tumor-models",
    repo_type="model"
)
```

### 로컬 개발 시 수동 다운로드
```bash
# HuggingFace Hub에서 직접 다운로드
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='your-username/brain-tumor-models', filename='monai_best.pth', local_dir='models/monai')
hf_hub_download(repo_id='your-username/brain-tumor-models', filename='segmamba_best.pth', local_dir='models/segmamba')
"
```

### 배포 시 자동 다운로드
서버 시작 시 `startup.py`가 자동으로 가중치를 다운로드합니다.
`models/` 폴더에 이미 존재하면 다운로드를 건너뜁니다.

---

## 실행 방법

### 사전 준비
```bash
# 저장소 클론
git clone https://github.com/simplething4057/brain-tumor-diagnosis.git
cd brain-tumor-diagnosis

# 환경변수 설정
cp .env.dev .env   # 개발 환경
```

### 개발 환경 (Ollama 로컬 LLM)
```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### 배포 환경 (Groq API)
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up
```

---

## 데이터셋
- BraTS 2023 (Kaggle 재배포본, GLI + MEN + MET 트랙)
- 데이터 형식: `.nii.gz` (4채널: T1c, T1n, T2f, T2w)
- 학습 환경: Kaggle Notebook (T4 GPU, FP16 Mixed Precision)
- 표본 수: 각 종양별 50 / 100 / 150케이스 순차 실험

---

## 평가 지표
- **주 지표**: Dice Score (ET / TC / WT) — BraTS 공식 방식
- **보조 지표**: IoU, F1 Score (macro), Confusion Matrix
- **베이스라인**: nnU-Net (ET 0.855 / TC 0.901 / WT 0.926)
