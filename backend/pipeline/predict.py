"""
예측 파이프라인:
1. 업로드된 .nii 파일을 임시 디렉터리에 저장
2. [seg 모드]  seg 파일을 GLI/MEN/MET 폴더에 복사 → feature_extractor → RF
3. [no-seg 모드] MRI만 업로드 시 → Docker segmenter 연동 예정 (현재 테스트용 mock 반환)
"""
import sys
import shutil
import pickle
import pathlib
import uuid
from pathlib import Path


class _CrossPlatformUnpickler(pickle.Unpickler):
    """
    Windows → Linux pickle 호환 처리:
    - WindowsPath → PurePosixPath
    - TumorTypeClassifier: 저장 당시 모듈(__main__, __mp_main__ 등)에 관계없이
      항상 src.classifier.meta_classifier 에서 로드
    """
    def find_class(self, module, name):
        if module == "pathlib" and name == "WindowsPath":
            return pathlib.PurePosixPath
        if name == "TumorTypeClassifier":
            from src.classifier.meta_classifier import TumorTypeClassifier
            return TumorTypeClassifier
        return super().find_class(module, name)

import numpy as np
import pandas as pd
from loguru import logger

from core.config import settings

# ML 파이프라인 경로를 모듈 로드 시점에 sys.path에 추가
_ml_root = str(settings.ml_path)
if _ml_root not in sys.path:
    sys.path.insert(0, _ml_root)
    logger.info(f"ML 파이프라인 경로 등록: {_ml_root}")

CLASSES = ["GLI", "MEN", "MET"]
FEAT_COLS = [
    f"{c.lower()}_{f}"
    for c in CLASSES
    for f in [
        "total_voxels", "total_volume_mm3", "et_ratio", "edema_ratio",
        "core_ratio", "lesion_count", "has_tumor"
    ]
]


def load_classifier():
    """RF 모델 로드 (싱글턴 패턴)"""
    if not hasattr(load_classifier, "_model"):
        path = settings.model_path
        if not path.exists():
            raise FileNotFoundError(f"모델 파일 없음: {path}")
        with open(path, "rb") as f:
            load_classifier._model = _CrossPlatformUnpickler(f).load()
        logger.info(f"RF 모델 로드 완료: {path}")
    return load_classifier._model


def run_prediction(
    subject_id: str,
    seg_nii_path: Path,
) -> dict:
    """
    Args:
        subject_id: 고유 식별자 (UUID)
        seg_nii_path: 업로드된 seg .nii 파일 경로

    Returns:
        {
            subject_id, prediction, confidence,
            gli_prob, men_prob, met_prob, features
        }
    """
    from src.classifier.feature_extractor import build_feature_vector  # noqa: E402

    clf = load_classifier()

    # seg를 GLI/MEN/MET 폴더에 복사 (feature_extractor 입력 구조 맞춤)
    # 원본 확장자 보존: .nii.gz 파일을 .nii로 저장하면 nibabel이 타입 감지 실패
    seg_nii_path = Path(seg_nii_path)
    src_name = seg_nii_path.name
    if src_name.endswith(".nii.gz"):
        seg_ext = ".nii.gz"
    else:
        seg_ext = ".nii"

    seg_paths = {}
    for cls in CLASSES:
        out_dir = settings.seg_output_path / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / f"{subject_id}{seg_ext}"
        shutil.copy2(seg_nii_path, dst)
        seg_paths[cls] = dst

    # 특징 추출
    feats = build_feature_vector(seg_paths)
    X = pd.DataFrame([feats])[FEAT_COLS]

    # 예측
    y_pred_enc = clf.model.predict(X.values)[0]
    proba = clf.model.predict_proba(X.values)[0]

    label_map = {0: "GLI", 1: "MEN", 2: "MET"}
    prediction = label_map[y_pred_enc]
    confidence = float(proba[y_pred_enc])

    result = {
        "subject_id": subject_id,
        "prediction": prediction,
        "confidence": confidence,
        "gli_prob": float(proba[0]),
        "men_prob": float(proba[1]),
        "met_prob": float(proba[2]),
        "features": feats,
    }
    logger.info(f"[{subject_id}] → {prediction} (conf={confidence:.3f})")
    return result


def run_prediction_no_seg(subject_id: str, mri_paths: dict) -> dict:
    """
    MRI 파일(t1c/t1n/t2f/t2w)만으로 예측.

    파이프라인:
        1. brats 패키지로 GLI / MEN / MET 세그멘테이션
           (내부적으로 BraTS Docker 알고리즘 자동 실행 — CPU 모드)
        2. 3개 seg.nii.gz → build_feature_vector() → 21차원 피처
        3. RF 모델로 GLI/MEN/MET 분류

    Args:
        subject_id: UUID 문자열
        mri_paths:  {"t1c": Path, "t1n": Path, "t2f": Path, "t2w": Path}
                    없는 모달리티는 있는 것으로 자동 대체.
    """
    from src.classifier.feature_extractor import build_feature_vector  # noqa: E402
    from src.inference.brats_infer import InferenceInput, run_all_types  # noqa: E402

    clf = load_classifier()
    out_dir = settings.upload_path / subject_id

    # ── 모달리티 대체 처리 ────────────────────────────────────────────────────
    # brats는 4채널 모두 요구 → 없는 채널은 인접 채널로 대체
    t1c = mri_paths.get("t1c")
    t1n = mri_paths.get("t1n")
    t2f = mri_paths.get("t2f")
    t2w = mri_paths.get("t2w")

    if t1c is None and t1n is None:
        raise ValueError("t1c 또는 t1n 중 최소 하나가 필요합니다.")
    if t2f is None and t2w is None:
        raise ValueError("t2f 또는 t2w 중 최소 하나가 필요합니다.")

    t1c = t1c or t1n   # t1c 없으면 t1n으로 대체
    t1n = t1n or t1c
    t2f = t2f or t2w   # t2f 없으면 t2w로 대체
    t2w = t2w or t2f

    inputs = InferenceInput(t1c=t1c, t1n=t1n, t2f=t2f, t2w=t2w)

    # ── 세그멘테이션 config (CPU 모드) ────────────────────────────────────────
    # cuda_devices="" → CPU 사용 (GPU 사용 시 "0"으로 변경)
    infer_config = {
        "inference": {
            "cuda_devices": "",
            "GLI": {"algorithm": "BraTS23_1"},
            "MEN": {"algorithm": "BraTS23_1"},
            "MET": {"algorithm": "BraTS23_1"},
        }
    }

    # ── BraTS 세그멘테이션 실행 ───────────────────────────────────────────────
    # 출력: out_dir/GLI/{subject_id}.nii.gz
    #       out_dir/MEN/{subject_id}.nii.gz
    #       out_dir/MET/{subject_id}.nii.gz
    logger.info(f"[{subject_id}] brats 세그멘테이션 시작 (CPU 모드) — {list(mri_paths.keys())}")
    seg_paths = run_all_types(
        subject_id=subject_id,
        inputs=inputs,
        config=infer_config,
        output_base=out_dir,
    )

    failed = [cls for cls, path in seg_paths.items() if path is None]
    if failed:
        raise RuntimeError(f"세그멘테이션 실패: {failed}")

    # ── 피처 추출 → RF 분류 (기존 코드 재사용) ──────────────────────────────
    feats = build_feature_vector(seg_paths)
    X = pd.DataFrame([feats])[FEAT_COLS]

    y_pred_enc = clf.model.predict(X.values)[0]
    proba = clf.model.predict_proba(X.values)[0]

    label_map = {0: "GLI", 1: "MEN", 2: "MET"}
    prediction = label_map[y_pred_enc]
    confidence = float(proba[y_pred_enc])

    logger.info(f"[{subject_id}] no-seg → {prediction} (conf={confidence:.3f})")
    return {
        "subject_id": subject_id,
        "prediction": prediction,
        "confidence": confidence,
        "gli_prob": float(proba[0]),
        "men_prob": float(proba[1]),
        "met_prob": float(proba[2]),
        "features": feats,
        "mode": "no_seg",
    }
