"""
RAG + LLM 기반 방사선 소견 보고서 생성
LLM_BACKEND=ollama → Ollama (로컬)
LLM_BACKEND=groq   → Groq API (클라우드)
"""
import httpx
from loguru import logger

from core.config import settings
from rag.vector_store import vector_store

REPORT_PROMPT_TEMPLATE = """당신은 대한민국 상급종합병원의 신경방사선과 전문의입니다.
아래 AI 분석 결과와 참고 문헌을 바탕으로 실제 임상에서 사용하는 형식의 한국어 방사선 판독 보고서를 작성하십시오.

## 참고 문헌
{context}

## AI 예측 결과
- 예측 종양 유형: {prediction}
- 신뢰도: {confidence:.1%}  (GLI {gli_prob:.1%} / MEN {men_prob:.1%} / MET {met_prob:.1%})

## 정량적 영상 특징
{feature_summary}

## 임상적 해석 단서
{clinical_hints}

## 작성 규칙
- 한국어(한글)로만 작성. 일본어 사용 금지.
- ET, ED, NCR, T1C, FLAIR 등 의학 약어는 영문 그대로 사용 가능.
- 각 항목에 위 수치 근거를 반영한 구체적 문장을 작성할 것.
- 괄호 안 설명문은 절대 그대로 출력하지 말고 실제 내용으로 대체할 것.
- 환자 이름, 날짜, MR 번호는 기재 금지.

## 출력 형식 (4개 항목 모두 반드시 작성)

**[촬영 정보]**
뇌 MRI 다중 시퀀스(T1C, T1N, T2/FLAIR) 영상 분석. BraTS 기반 자동 세그멘테이션 적용.

**[임상 증상]**
{prediction} 종양에서 흔히 나타나는 두통, 신경학적 결손, 인지기능 저하 등의 증상이 의심되며, 병변의 위치 및 크기에 따른 국소 신경학적 증상을 동반할 수 있다.

**[MRI 소견]**
(여기에 종양 크기·형태, 조영증강 패턴, 주변 부종, 괴사 여부를 위 수치 근거로 3~5문장 상세 서술)

**[결론]**
(여기에 최종 진단, 감별 진단, 추가 검사 및 치료 권고 사항을 2~3문장 서술)
"""


def _build_clinical_hints(feats: dict, prediction: str) -> str:
    """수치 기반 임상 해석 단서 생성"""
    prefix = prediction.lower()  # gli / men / met
    hints = []

    et_ratio   = float(feats.get(f"{prefix}_et_ratio", 0) or 0)
    ed_ratio   = float(feats.get(f"{prefix}_edema_ratio", 0) or 0)
    core_ratio = float(feats.get(f"{prefix}_core_ratio", 0) or 0)
    volume     = float(feats.get(f"{prefix}_total_volume_mm3", 0) or 0)
    lesions    = int(feats.get(f"{prefix}_lesion_count", 1) or 1)

    # 조영 증강 패턴
    if et_ratio > 0.3:
        hints.append(f"조영 증강 분율 높음 (ET ratio {et_ratio:.1%}) → 조영증강 패턴 뚜렷, 혈뇌장벽 파괴 시사")
    elif et_ratio > 0.1:
        hints.append(f"중등도 조영 증강 (ET ratio {et_ratio:.1%})")
    else:
        hints.append(f"조영 증강 분율 낮음 (ET ratio {et_ratio:.1%}) → 저등급 또는 비조영증강 종양 가능성")

    # 주변 부종
    if ed_ratio > 0.5:
        hints.append(f"광범위 주변 부종 (edema ratio {ed_ratio:.1%}) → 침윤성 성장 또는 전이성 병변 가능성")
    elif ed_ratio > 0.2:
        hints.append(f"중등도 주변 부종 (edema ratio {ed_ratio:.1%})")
    else:
        hints.append(f"경미한 주변 부종 (edema ratio {ed_ratio:.1%})")

    # 괴사
    if core_ratio > 0.2:
        hints.append(f"괴사 중심부 분율 높음 (core ratio {core_ratio:.1%}) → 고등급 교종 의심")
    elif core_ratio > 0.05:
        hints.append(f"부분적 괴사 소견 (core ratio {core_ratio:.1%})")

    # 종양 크기
    vol_cm3 = volume / 1000
    if vol_cm3 > 30:
        hints.append(f"대형 종양 ({vol_cm3:.1f} cm³) → 수술적 접근 우선 고려")
    elif vol_cm3 > 10:
        hints.append(f"중등도 크기 종양 ({vol_cm3:.1f} cm³)")
    else:
        hints.append(f"소형 종양 ({vol_cm3:.1f} cm³)")

    # 다발성
    if lesions > 1:
        hints.append(f"병변 {lesions}개 감지 → 다발성 전이 또는 다소성 교종 감별 필요")

    # MRI 신호 특성 (종양 유형별 전형적 패턴)
    signal_map = {
        "gli": [
            "T2/FLAIR 고신호강도 병변 (침윤성 부종 포함)",
            "T1C 불균일 ring enhancement (고등급 시)" if et_ratio > 0.15 else "T1C 조영증강 미미 (저등급 가능성)",
            "ADC 감소 동반 가능 (세포 밀도 높은 경우)",
        ],
        "men": [
            "T1C 균일한 강한 조영증강 (dural tail sign 동반 가능)",
            "T2 동신호 ~ 약간 고신호강도",
            "경막 부착부 확인 필요",
        ],
        "met": [
            "T1C ring 또는 결절형 조영증강",
            "T2/FLAIR 병변 주변 불균형 광범위 부종",
            "피질-수질 경계부 호발, 다발성 병변 확인 필요",
        ],
    }
    for sig in signal_map.get(prefix, []):
        hints.append(f"[MRI 신호] {sig}")

    return "\n".join(f"- {h}" for h in hints) if hints else "- 특이 소견 없음"


def _format_features(feats: dict, prediction: str) -> str:
    """예측 클래스의 주요 특징 수치 요약"""
    if not feats:
        return "특징 데이터 없음"

    prefix = prediction.lower()  # gli / men / met (소문자)
    key_map = {
        "total_volume_mm3": "종양 부피",
        "et_ratio":         "조영 증강 분율 (ET ratio)",
        "edema_ratio":      "주변 부종 분율 (edema ratio)",
        "core_ratio":       "괴사 중심 분율 (core ratio)",
        "lesion_count":     "병변 개수",
        "has_tumor":        "종양 존재 여부",
    }
    lines = []
    for key, label in key_map.items():
        val = feats.get(f"{prefix}_{key}")
        if val is None:
            continue
        if "ratio" in key:
            lines.append(f"- {label}: {float(val):.1%}")
        elif "volume" in key:
            lines.append(f"- {label}: {float(val):.1f} mm³  ({float(val)/1000:.2f} cm³)")
        else:
            lines.append(f"- {label}: {val}")
    return "\n".join(lines) if lines else "특징 데이터 없음"


async def _call_ollama(prompt: str) -> str:
    logger.debug(f"Ollama 프롬프트 길이: {len(prompt)} 문자")
    async with httpx.AsyncClient(timeout=540.0) as client:  # 9분
        try:
            resp = await client.post(
                f"{settings.ollama_base_url}/api/generate",
                json={
                    "model": settings.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 1200,
                        "num_ctx": 4096,
                    },
                },
            )
            if resp.status_code != 200:
                logger.error(f"Ollama 오류 {resp.status_code}: {resp.text[:500]}")
            resp.raise_for_status()
            return resp.json()["response"].strip()
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP 오류: {e.response.status_code} — {e.response.text[:500]}")
            raise


async def _call_groq(prompt: str) -> str:
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage

    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model_name="llama3-70b-8192",
        temperature=0.3,
        max_tokens=800,
    )
    msg = await llm.ainvoke([HumanMessage(content=prompt)])
    return msg.content.strip()


async def generate_report(
    prediction: str,
    confidence: float,
    gli_prob: float,
    men_prob: float,
    met_prob: float,
    features: dict | None,
) -> str:
    """방사선 소견 보고서 생성 (RAG + LLM)"""
    query = f"{prediction} brain tumor MRI findings radiology report Korean"
    context_docs = vector_store.query(query, n_results=2, label_filter=prediction)
    context = "\n\n---\n\n".join(context_docs) if context_docs else "참고 문헌 없음."

    feats = features or {}
    feature_summary = _format_features(feats, prediction)
    clinical_hints  = _build_clinical_hints(feats, prediction)

    prompt = REPORT_PROMPT_TEMPLATE.format(
        context=context,
        prediction=prediction,
        confidence=confidence,
        gli_prob=gli_prob,
        men_prob=men_prob,
        met_prob=met_prob,
        feature_summary=feature_summary,
        clinical_hints=clinical_hints,
    )

    try:
        if settings.llm_backend == "groq" and settings.groq_api_key:
            logger.info("Groq API 사용")
            report = await _call_groq(prompt)
        else:
            logger.info(f"Ollama 사용 (모델: {settings.ollama_model})")
            report = await _call_ollama(prompt)
        return report
    except Exception as e:
        logger.error(f"LLM 호출 실패: {e}")
        return (
            f"**[촬영 정보]**\n"
            f"뇌 MRI 다중 시퀀스 분석. BraTS 기반 자동 세그멘테이션 적용.\n\n"
            f"**[임상 증상]**\n"
            f"AI 예측 결과: {prediction} (신뢰도: {confidence:.1%})\n\n"
            f"**[MRI 소견]**\n"
            f"{feature_summary}\n\n"
            f"**[결론]**\n"
            f"자동 보고서 생성 실패. LLM 서비스를 확인하세요.\n"
            f"오류: {str(e)}"
        )
