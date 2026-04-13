from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class InferenceResult(Base):
    __tablename__ = "inference_results"

    id           = Column(Integer, primary_key=True, index=True)
    label        = Column(String, nullable=False)        # Glioma / Meningioma / Metastases
    confidence   = Column(Float, nullable=False)
    area_ratio   = Column(Float)
    location     = Column(String)
    boundary     = Column(String)
    report       = Column(Text)                          # LLM 소견문
    created_at   = Column(DateTime, default=datetime.utcnow)
