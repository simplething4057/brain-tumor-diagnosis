import numpy as np

def encode_results(mask, label, confidence, heatmap) -> dict:
    """
    모델 출력 → LLM 입력용 텍스트 메타데이터 변환
    """
    total_voxels = mask.size
    tumor_voxels = np.sum(mask > 0)
    area_ratio = (tumor_voxels / total_voxels) * 100

    # 종양 중심 좌표
    coords = np.argwhere(mask > 0)
    center = coords.mean(axis=0) if len(coords) > 0 else [0, 0, 0]

    # 경계 불규칙성 (간단 추정)
    boundary = "불규칙" if area_ratio > 5 else "비교적 명확"

    # Grad-CAM 고활성 영역
    gradcam_max = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    gradcam_region = f"좌표 {gradcam_max}"

    return {
        "label": label,
        "confidence": confidence,
        "area_ratio": round(area_ratio, 2),
        "location": f"중심 좌표 {[round(c, 1) for c in center]}",
        "boundary": boundary,
        "gradcam_region": gradcam_region,
    }
