"""
FastAPI 라우터:
POST /api/predict             — 파일 업로드 → 백그라운드 예측 시작, job_id 즉시 반환
GET  /api/predict/status/{id} — 예측 진행 상태 폴링
GET  /api/files/{id}/{mod}    — .nii / .nii.gz 서빙 (NiiVue용)
POST /api/report/{id}         — RAG 보고서 생성
GET  /api/history             — 예측 이력 조회
GET  /api/history/{id}        — 단건 조회
"""
import asyncio
import uuid
import shutil
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from db.database import get_db
from db import crud
from pipeline.predict import run_prediction, run_prediction_no_seg
from rag.report_generator import generate_report

router = APIRouter(prefix="/api")

ALLOWED_MODALITIES = {"t1n", "t1c", "t2f", "t2w", "seg"}

# ─── 인메모리 Job 저장소 ─────────────────────────────────────────────────────
# { job_id: { status, result, error, subject_id, file_exts } }
_jobs: dict[str, dict] = {}


def _get_nii_ext(filename: str) -> str:
    if filename and filename.endswith(".nii.gz"):
        return ".nii.gz"
    return ".nii"


def _find_nii_file(directory: Path, name: str) -> Path | None:
    for ext in (".nii.gz", ".nii"):
        p = directory / f"{name}{ext}"
        if p.exists():
            return p
    return None


async def _run_job(job_id: str, subject_id: str, seg_path, mri_paths: dict):
    """백그라운드 예측 태스크 — 완료 즉시 DB 저장 (서버 재시작 대비)"""
    try:
        if seg_path is not None:
            result = await asyncio.to_thread(run_prediction, subject_id, seg_path)
        else:
            result = await asyncio.to_thread(run_prediction_no_seg, subject_id, mri_paths)

        # DB에 즉시 저장 (메모리 손실 대비)
        record = await asyncio.to_thread(
            crud._create_prediction_sync,
            {
                "subject_id": subject_id,
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "gli_prob": result["gli_prob"],
                "men_prob": result["men_prob"],
                "met_prob": result["met_prob"],
                "features": result["features"],
            },
        )
        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["result"] = result
        _jobs[job_id]["record_id"] = record.id
    except Exception as e:
        from loguru import logger
        logger.error(f"[{subject_id}] 백그라운드 예측 실패: {e}")
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)


# ─── 예측 시작 ────────────────────────────────────────────────────────────────

@router.post("/predict")
async def predict(
    seg: UploadFile = File(None),
    t1c: UploadFile = File(None),
    t1n: UploadFile = File(None),
    t2f: UploadFile = File(None),
    t2w: UploadFile = File(None),
):
    mri_uploads = {
        name: upload
        for name, upload in [("t1c", t1c), ("t1n", t1n), ("t2f", t2f), ("t2w", t2w)]
        if upload is not None
    }
    if seg is None and not mri_uploads:
        raise HTTPException(
            status_code=400,
            detail="seg 또는 MRI 파일(t1c/t1n/t2f/t2w) 중 최소 1개가 필요합니다.",
        )

    subject_id = str(uuid.uuid4())
    subj_dir = settings.upload_path / subject_id
    subj_dir.mkdir(parents=True, exist_ok=True)

    # ── 파일 저장 ────────────────────────────────────────────────────────────
    seg_path = None
    file_exts: dict[str, str] = {}

    if seg is not None:
        ext = _get_nii_ext(seg.filename or "")
        seg_path = subj_dir / f"seg{ext}"
        with open(seg_path, "wb") as f:
            shutil.copyfileobj(seg.file, f)
        file_exts["seg"] = ext

    mri_paths = {}
    for name, upload in mri_uploads.items():
        ext = _get_nii_ext(upload.filename or "")
        out = subj_dir / f"{name}{ext}"
        with open(out, "wb") as f:
            shutil.copyfileobj(upload.file, f)
        mri_paths[name] = out
        file_exts[name] = ext

    # ── 백그라운드 예측 시작 ─────────────────────────────────────────────────
    job_id = subject_id
    _jobs[job_id] = {
        "status": "running",
        "result": None,
        "error": None,
        "subject_id": subject_id,
        "file_exts": file_exts,
        "mode": "seg" if seg_path else "no_seg",
    }
    asyncio.create_task(_run_job(job_id, subject_id, seg_path, mri_paths))

    return {
        "job_id": job_id,
        "subject_id": subject_id,
        "status": "running",
        "file_exts": file_exts,
        "mode": "seg" if seg_path else "no_seg",
    }


# ─── 예측 상태 폴링 ───────────────────────────────────────────────────────────

@router.get("/predict/status/{job_id}")
async def predict_status(job_id: str, db: AsyncSession = Depends(get_db)):
    job = _jobs.get(job_id)

    # ── 메모리에 없으면 DB에서 복구 (서버 재시작 후 폴링 재개용) ──────────────
    if not job:
        record = await crud.get_prediction_by_subject_id(job_id)
        if record:
            return {
                "status": "done",
                "record_id": record.id,
                "subject_id": record.subject_id,
                "prediction": record.prediction,
                "confidence": record.confidence,
                "probabilities": {
                    "GLI": record.gli_prob,
                    "MEN": record.men_prob,
                    "MET": record.met_prob,
                },
                "mode": "unknown",
                "file_exts": {},
            }
        # 아직 진행 중이거나 존재하지 않는 job
        return {"status": "running", "subject_id": job_id}

    if job["status"] == "running":
        return {"status": "running", "subject_id": job["subject_id"]}

    if job["status"] == "failed":
        err = job.pop("error", "알 수 없는 오류")
        _jobs.pop(job_id, None)
        raise HTTPException(status_code=500, detail=f"예측 실패: {err}")

    # done → 백그라운드 태스크가 이미 DB에 저장함, record_id만 꺼내 반환
    result = job["result"]
    subject_id = job["subject_id"]
    file_exts = job["file_exts"]
    mode = job.get("mode", "seg")
    record_id = job.get("record_id")
    _jobs.pop(job_id, None)

    # record_id가 없으면 (예외 상황) DB에서 조회
    if not record_id:
        rec = await crud.get_prediction_by_subject_id(subject_id)
        record_id = rec.id if rec else None

    return {
        "status": "done",
        "record_id": record_id,
        "subject_id": subject_id,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "probabilities": {
            "GLI": result["gli_prob"],
            "MEN": result["men_prob"],
            "MET": result["met_prob"],
        },
        "mode": mode,
        "file_exts": file_exts,
    }


# ─── 파일 서빙 (NiiVue용) ────────────────────────────────────────────────────

@router.get("/files/{subject_id}/{modality}")
async def get_file(subject_id: str, modality: str):
    if modality not in ALLOWED_MODALITIES:
        raise HTTPException(status_code=400, detail=f"허용 모달리티: {ALLOWED_MODALITIES}")

    nii_path = _find_nii_file(settings.upload_path / subject_id, modality)
    if nii_path is None:
        raise HTTPException(status_code=404, detail=f"파일 없음: {subject_id}/{modality}")

    filename = f"{subject_id}-{modality}.nii.gz" if nii_path.name.endswith(".nii.gz") else f"{subject_id}-{modality}.nii"
    media_type = "application/gzip" if nii_path.name.endswith(".nii.gz") else "application/octet-stream"

    return FileResponse(
        path=str(nii_path),
        media_type=media_type,
        filename=filename,
        headers={"Access-Control-Allow-Origin": "*"},
    )


# ─── RAG 보고서 생성 ─────────────────────────────────────────────────────────

@router.post("/report/{record_id}")
async def create_report(record_id: int, db: AsyncSession = Depends(get_db)):
    record = await crud.get_prediction_by_id(db, record_id)
    if not record:
        raise HTTPException(status_code=404, detail="예측 기록 없음")

    if record.report:
        return {"record_id": record_id, "report": record.report}

    report = await generate_report(
        prediction=record.prediction,
        confidence=record.confidence,
        gli_prob=record.gli_prob,
        men_prob=record.men_prob,
        met_prob=record.met_prob,
        features=record.features,
    )

    updated = await crud.update_report(db, record_id, report)
    return {"record_id": record_id, "report": updated.report}


# ─── 이력 조회 ───────────────────────────────────────────────────────────────

@router.get("/history")
async def get_history(skip: int = 0, limit: int = 50, db: AsyncSession = Depends(get_db)):
    records = await crud.get_history(db, skip=skip, limit=limit)
    return [r.to_dict() for r in records]


@router.get("/history/{record_id}")
async def get_history_item(record_id: int, db: AsyncSession = Depends(get_db)):
    record = await crud.get_prediction_by_id(db, record_id)
    if not record:
        raise HTTPException(status_code=404, detail="기록 없음")
    return record.to_dict()
