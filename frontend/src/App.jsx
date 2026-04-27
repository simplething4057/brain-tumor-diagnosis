import { useState, useCallback, useRef } from "react";
import FileUploader from "./components/FileUploader";
import NiiVueViewer from "./components/NiiVueViewer";
import ResultPanel from "./components/ResultPanel";
import HistoryPanel from "./components/HistoryPanel";
import { predict, pollPredictStatus } from "./utils/api";

function Logo() {
  return (
    <div className="flex items-center gap-3">
      <div className="w-8 h-8 rounded-lg bg-brand-500 flex items-center justify-center text-white font-bold text-sm">
        BT
      </div>
      <div>
        <h1 className="text-sm font-bold text-white leading-none">
          Brain Tumor Diagnosis
        </h1>
        <p className="text-xs text-gray-500">GLI · MEN · MET 3종 분류</p>
      </div>
    </div>
  );
}

function formatElapsed(sec) {
  if (sec < 60) return `${sec}초`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}분 ${s}초`;
}

export default function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState(""); // 경과 시간 등 상태 메시지
  const [result, setResult] = useState(null);
  const [recordId, setRecordId] = useState(null);
  const [subjectId, setSubjectId] = useState(null);
  const [availableModalities, setAvailableModalities] = useState([]);
  const [fileExts, setFileExts] = useState({});
  const [activeTab, setActiveTab] = useState("upload");
  const [error, setError] = useState(null);
  const viewerRef = useRef(null);

  const handlePredict = useCallback(async (files) => {
    setIsLoading(true);
    setLoadingMsg("파일 업로드 중...");
    setError(null);
    setResult(null);

    try {
      // 1) 파일 업로드 → job_id 즉시 반환
      const jobData = await predict(files);
      const { job_id, subject_id, file_exts: exts, mode } = jobData;

      // 업로드된 모달리티 목록
      const mods = [];
      if (files.seg) mods.push("seg");
      if (files.t1c) mods.push("t1c");
      if (files.t1n) mods.push("t1n");
      if (files.t2f) mods.push("t2f");
      if (files.t2w) mods.push("t2w");

      setSubjectId(subject_id);
      setAvailableModalities(mods);
      setFileExts(exts || {});

      const modeLabel = mode === "no_seg"
        ? "BraTS 세그멘테이션 중 (CPU, 수십 분 소요)"
        : "예측 중...";
      setLoadingMsg(modeLabel);

      // 2) 폴링 — 완료까지 3초마다 상태 확인 (타임아웃 없음)
      const finalData = await pollPredictStatus(
        job_id,
        ({ elapsedSec }) => {
          setLoadingMsg(`${modeLabel} — 경과 ${formatElapsed(elapsedSec)}`);
        },
        3000
      );

      setResult({
        prediction: finalData.prediction,
        confidence: finalData.confidence,
        probabilities: finalData.probabilities,
      });
      setRecordId(finalData.record_id);
      if (finalData.file_exts) setFileExts(finalData.file_exts);
    } catch (e) {
      const msg = e.response?.data?.detail || e.message || "예측 요청 실패";
      setError(msg);
    } finally {
      setIsLoading(false);
      setLoadingMsg("");
    }
  }, []);

  const handleHistorySelect = useCallback((rec) => {
    setResult({
      prediction: rec.prediction,
      confidence: rec.confidence,
      probabilities: {
        GLI: rec.gli_prob,
        MEN: rec.men_prob,
        MET: rec.met_prob,
      },
    });
    setRecordId(rec.id);
    setSubjectId(rec.subject_id);
    setAvailableModalities(["seg"]);
    setActiveTab("upload");
  }, []);

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      {/* 헤더 */}
      <header className="flex items-center justify-between px-4 py-3 bg-gray-900 border-b border-gray-800 flex-shrink-0">
        <Logo />
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <span className="w-2 h-2 rounded-full bg-green-500" />
          RF Classifier · NiiVue · RAG+LLM · CPU 테스트 모드
        </div>
      </header>

      {/* 메인 레이아웃 */}
      <div className="flex flex-1 overflow-hidden">
        {/* 사이드바 (좌) */}
        <aside className="w-72 flex-shrink-0 bg-gray-900 border-r border-gray-800 flex flex-col overflow-hidden">
          <div className="flex border-b border-gray-800">
            {["upload", "history"].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`flex-1 py-2 text-xs font-medium uppercase transition ${
                  activeTab === tab
                    ? "text-white border-b-2 border-brand-500"
                    : "text-gray-500 hover:text-gray-300"
                }`}
              >
                {tab === "upload" ? "업로드" : "이력"}
              </button>
            ))}
          </div>

          <div className="flex-1 overflow-y-auto p-4">
            {activeTab === "upload" ? (
              <div className="space-y-4">
                <FileUploader onPredict={handlePredict} isLoading={isLoading} />

                {/* 로딩 진행 상태 */}
                {isLoading && loadingMsg && (
                  <div className="rounded-lg bg-indigo-900 bg-opacity-30 border border-indigo-700 p-3">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="animate-spin w-3 h-3 border-2 border-indigo-400 border-t-transparent rounded-full flex-shrink-0" />
                      <p className="text-xs text-indigo-300 font-medium">처리 중</p>
                    </div>
                    <p className="text-xs text-indigo-400">{loadingMsg}</p>
                  </div>
                )}

                {error && (
                  <div className="rounded-lg bg-red-900 bg-opacity-30 border border-red-700 p-3">
                    <p className="text-xs text-red-400">{error}</p>
                  </div>
                )}
              </div>
            ) : (
              <HistoryPanel onSelect={handleHistorySelect} currentId={recordId} />
            )}
          </div>
        </aside>

        {/* 3D 뷰어 (중앙) */}
        <main className="flex-1 p-3 bg-gray-950 overflow-hidden">
          {subjectId && availableModalities.length > 0 ? (
            <NiiVueViewer
              ref={viewerRef}
              subjectId={subjectId}
              availableModalities={availableModalities}
              fileExts={fileExts}
            />
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="w-16 h-16 rounded-full bg-gray-800 flex items-center justify-center mb-4">
                <svg className="w-8 h-8 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                    d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
              </div>
              <p className="text-gray-600 text-sm">MRI 또는 seg.nii 업로드 후 3D 뷰어가 표시됩니다</p>
              <p className="text-gray-700 text-xs mt-1">T1C/T1N/T2F/T2W 파일만으로도 업로드 가능합니다</p>
            </div>
          )}
        </main>

        {/* 결과 패널 (우) */}
        {result && (
          <aside className="w-80 flex-shrink-0 bg-gray-900 border-l border-gray-800 overflow-y-auto p-4">
            <ResultPanel
              result={result}
              recordId={recordId}
              captureViewerImage={() => viewerRef.current?.captureImage() ?? ""}
            />
          </aside>
        )}
      </div>
    </div>
  );
}
