from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from startup import download_models

app = FastAPI(title="Brain Tumor Classification API")

@app.on_event("startup")
async def startup_event():
    download_models()   # 서버 시작 시 HuggingFace에서 가중치 자동 다운로드

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

@app.get("/health")
def health():
    return {"status": "ok"}
