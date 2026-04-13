from fastapi import APIRouter, UploadFile, File
from pipeline.preprocessing import preprocess_nii
from pipeline.infer import run_inference
from pipeline.gradcam import generate_gradcam
from pipeline.text_encoder import encode_results
from rag.report_generator import generate_report
from db.crud import save_result, get_history

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    .nii.gz 파일 업로드 → 종양 변별 + 소견문 생성
    """
    # 1. 전처리
    volume = preprocess_nii(await file.read())
    # 2. 모델 추론 (분할 + 분류)
    mask, label, confidence = run_inference(volume)
    # 3. Grad-CAM
    heatmap = generate_gradcam(volume, mask)
    # 4. 텍스트 변환
    metadata = encode_results(mask, label, confidence, heatmap)
    # 5. 소견문 생성
    report = generate_report(metadata)
    # 6. DB 저장
    save_result(label, confidence, report, metadata)

    return {
        "label": label,
        "confidence": confidence,
        "metadata": metadata,
        "report": report,
    }

@router.get("/history")
def history():
    return get_history()
