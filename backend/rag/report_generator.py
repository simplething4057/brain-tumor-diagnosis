import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_llm():
    mode = os.getenv("LLM_MODE", "local")
    if mode == "local":
        from langchain_community.llms import Ollama
        return Ollama(
            base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "llama3:8b")
        )
    else:
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=os.getenv("GROQ_MODEL", "llama3-70b-8192")
        )

def generate_report(metadata: dict) -> str:
    """
    Grad-CAM 결과 + 탐지 수치 → RAG 기반 소견문 생성
    """
    llm = get_llm()
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma(
        persist_directory="./rag_docs/chroma_db",
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = f"""
다음 뇌 MRI 분석 결과를 바탕으로 방사선과 소견문을 작성하세요.

[분석 결과]
종양 종류: {metadata['label']} (신뢰도 {metadata['confidence']:.1f}%)
위치: {metadata['location']}
면적 비율: {metadata['area_ratio']:.1f}%
경계 형태: {metadata['boundary']}
Grad-CAM 고활성 영역: {metadata['gradcam_region']}

[참고 문서]
{retriever.get_relevant_documents(metadata['label'])}
"""
    return llm.invoke(prompt)
