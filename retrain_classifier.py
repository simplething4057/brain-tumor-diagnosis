"""
meta_classifier.pkl 재학습 스크립트
───────────────────────────────────
사전 조건:
  pip install scikit-learn pandas loguru requests

실행 방법:
  cd C:\Users\USER\Documents\Claude\Projects\Brain-tumor-diagnosis
  python retrain_classifier.py

완료 후 생성 파일:
  ml_pipeline/models/weights/meta_classifier.pkl
"""
import sys
import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── brain-tumor-3class 경로 결정 ──────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent          # Brain-tumor-diagnosis 폴더
BT3CLASS_DIR = SCRIPT_DIR / "ml_pipeline"

if not BT3CLASS_DIR.exists():
    print(f"[오류] 폴더를 찾을 수 없음: {BT3CLASS_DIR}")
    print("      ml_pipeline/ 폴더가 Brain-tumor-diagnosis 루트에 있어야 합니다.")
    sys.exit(1)

# ★ 핵심: src.classifier.meta_classifier 를 임포트할 수 있도록 sys.path 등록
if str(BT3CLASS_DIR) not in sys.path:
    sys.path.insert(0, str(BT3CLASS_DIR))

# brain-tumor-3class 의 실제 클래스를 임포트 → pickle 에 올바른 모듈 경로 기록
from src.classifier.meta_classifier import TumorTypeClassifier  # noqa: E402

SAVE_PATH    = BT3CLASS_DIR / "models" / "weights" / "meta_classifier.pkl"
FEATURES_CSV = BT3CLASS_DIR / "outputs" / "features" / "features.csv"

FEATURES_CSV_URL = (
    "https://raw.githubusercontent.com/simplething4057/brain-tumor-3class/main"
    "/outputs/features/features.csv"
)

# ── features.csv 로드 (없으면 GitHub에서 다운로드) ───────────────────────────
def load_features() -> pd.DataFrame:
    if FEATURES_CSV.exists():
        print(f"로컬 features.csv 사용: {FEATURES_CSV}")
        return pd.read_csv(FEATURES_CSV)

    print("features.csv 없음 → GitHub에서 다운로드 중...")
    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    try:
        import requests
        r = requests.get(FEATURES_CSV_URL, timeout=30)
        r.raise_for_status()
        FEATURES_CSV.write_bytes(r.content)
    except ImportError:
        import urllib.request
        urllib.request.urlretrieve(FEATURES_CSV_URL, FEATURES_CSV)
    print(f"  다운로드 완료: {FEATURES_CSV}")
    return pd.read_csv(FEATURES_CSV)


# ── 메인 학습 로직 ────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  meta_classifier.pkl 재학습")
    print(f"  ml_pipeline: {BT3CLASS_DIR}")
    print(f"  저장 경로: {SAVE_PATH}")
    print("=" * 60)

    # 1. 데이터 로드
    df = load_features()
    print(f"\n[1] 데이터 로드: {len(df)}행 × {len(df.columns)}열")
    print(f"    클래스 분포: {df['true_label'].value_counts().to_dict()}")

    # 2. 피처/레이블 분리
    feature_cols = [c for c in df.columns if c not in ("subject_id", "true_label")]
    X = df[feature_cols]
    y = df["true_label"].tolist()

    # 3. Train/Test 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    print(f"\n[2] Train: {len(X_train)}  Test: {len(X_test)}")

    # 4. config 구성 (meta_classifier.py 의 TumorTypeClassifier.__init__ 에 필요)
    config = {
        "classifier": {
            "type": "random_forest",
            "model_save_path": str(SAVE_PATH),
            "random_forest": {
                "n_estimators": 200,
                "max_depth": 10,
                "class_weight": "balanced",
            },
        }
    }

    # 5. 학습
    print("\n[3] RandomForest 학습 중...")
    clf = TumorTypeClassifier(config)
    clf.fit(X_train, y_train.tolist() if hasattr(y_train, "tolist") else y_train)

    # 6. 평가
    y_pred = clf.predict(X_test)
    print("\n[4] 평가 결과:")
    print(classification_report(y_test, y_pred, target_names=["GLI", "MEN", "MET"]))

    # 7. 저장 (TumorTypeClassifier.save() 사용 → 올바른 모듈 경로로 pickle 저장)
    print("[5] 모델 저장 중...")
    clf.save()
    print(f"    → {SAVE_PATH}")

    print("\n완료! Docker 백엔드를 재시작하세요:")
    print("  docker compose restart backend")
    print("=" * 60)


if __name__ == "__main__":
    main()
