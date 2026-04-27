import axios from "axios";

const api = axios.create({
  baseURL: "/api",
  timeout: 60000, // 기본 1분 (파일 업로드 / 상태 폴링용)
});

/**
 * MRI 파일 업로드 → 백그라운드 예측 시작
 * 즉시 { job_id, subject_id, status: "running", file_exts } 반환
 */
export async function predict(files) {
  const formData = new FormData();
  if (files.seg) formData.append("seg", files.seg);
  if (files.t1c) formData.append("t1c", files.t1c);
  if (files.t1n) formData.append("t1n", files.t1n);
  if (files.t2f) formData.append("t2f", files.t2f);
  if (files.t2w) formData.append("t2w", files.t2w);

  const { data } = await api.post("/predict", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 60000, // 파일 업로드: 1분
  });
  return data;
}

/**
 * 예측 상태 폴링 — 완료 또는 실패까지 반복 (타임아웃 없음)
 * @param {string} jobId
 * @param {function} onTick — 매 폴링마다 { elapsedSec } 전달
 * @param {number} intervalMs — 폴링 간격 (기본 3초)
 */
export async function pollPredictStatus(jobId, onTick, intervalMs = 3000) {
  const startTime = Date.now();

  while (true) {
    await new Promise((r) => setTimeout(r, intervalMs));

    const elapsedSec = Math.floor((Date.now() - startTime) / 1000);
    if (onTick) onTick({ elapsedSec });

    try {
      const { data } = await api.get(`/predict/status/${jobId}`, {
        timeout: 15000,
      });

      if (data.status === "done") return data;
      if (data.status === "failed") {
        throw new Error(data.detail || "예측 실패");
      }
      // status === "running" → 계속 폴링
    } catch (e) {
      // 500 오류(예측 실패)만 전파, 나머지(404 포함)는 재시도
      if (e.response?.status === 500) throw e;
      console.warn("폴링 오류 (재시도):", e.response?.status ?? e.message);
    }
  }
}

/**
 * NiiVue에서 사용할 파일 URL 반환
 */
export function getFileUrl(subjectId, modality) {
  return `/api/files/${subjectId}/${modality}`;
}

/**
 * RAG 보고서 생성 요청
 */
export async function generateReport(recordId) {
  const { data } = await api.post(`/report/${recordId}`, null, {
    timeout: 10 * 60 * 1000, // 10분
  });
  return data.report;
}

/**
 * 예측 이력 조회
 */
export async function fetchHistory(skip = 0, limit = 50) {
  const { data } = await api.get("/history", { params: { skip, limit } });
  return data;
}

/**
 * 단건 이력 조회
 */
export async function fetchHistoryItem(recordId) {
  const { data } = await api.get(`/history/${recordId}`);
  return data;
}
