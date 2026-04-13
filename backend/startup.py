import os
from huggingface_hub import hf_hub_download

HF_REPO_ID = os.getenv("HF_REPO_ID", "your-username/brain-tumor-models")

MODELS = [
    {"filename": "monai_best.pth",     "local_dir": "models/monai"},
    {"filename": "segmamba_best.pth",  "local_dir": "models/segmamba"},
    {"filename": "normalization_stats.pkl", "local_dir": "models"},
]

def download_models():
    """
    서버 시작 시 HuggingFace Hub에서 모델 가중치 자동 다운로드
    이미 존재하는 파일은 건너뜁니다
    """
    for model in MODELS:
        local_path = os.path.join(model["local_dir"], model["filename"])
        os.makedirs(model["local_dir"], exist_ok=True)

        if os.path.exists(local_path):
            print(f"[startup] 이미 존재함, 건너뜀: {local_path}")
            continue

        print(f"[startup] 다운로드 중: {model['filename']} ...")
        try:
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=model["filename"],
                local_dir=model["local_dir"]
            )
            print(f"[startup] 완료: {local_path}")
        except Exception as e:
            print(f"[startup] 오류 발생 ({model['filename']}): {e}")
            raise RuntimeError(f"모델 가중치 다운로드 실패: {model['filename']}")
