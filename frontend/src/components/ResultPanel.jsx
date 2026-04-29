import { useState } from "react";
import { generateReport } from "../utils/api";

const SECTION_KEYS = ["촬영 정보", "임상 증상", "MRI 소견", "결론"];

const SECTION_STYLE = {
  "촬영 정보": { icon: "🔬", color: "text-sky-400",    border: "border-sky-700"    },
  "임상 증상": { icon: "🩺", color: "text-yellow-400", border: "border-yellow-700" },
  "MRI 소견": { icon: "🧠", color: "text-indigo-400",  border: "border-indigo-700" },
  "결론":     { icon: "📌", color: "text-green-400",   border: "border-green-700"  },
};

function parseReport(text) {
  // Split on **[섹션명]** headers (with optional surrounding whitespace/newlines)
  const headerRe = /\*{0,2}\[?(촬영 정보|임상 증상|MRI 소견|결론)\]?\*{0,2}/g;
  const sections = [];
  let lastKey = null;
  let lastIndex = 0;

  for (const m of text.matchAll(headerRe)) {
    if (lastKey !== null) {
      sections.push({ key: lastKey, body: text.slice(lastIndex, m.index).trim() });
    }
    lastKey = m[1];
    lastIndex = m.index + m[0].length;
  }
  if (lastKey !== null) {
    sections.push({ key: lastKey, body: text.slice(lastIndex).trim() });
  }

  // Fallback: return raw text if no sections found
  if (sections.length === 0) {
    return [{ key: null, body: text.trim() }];
  }
  return sections;
}

function ReportSection({ sectionKey, body }) {
  const style = SECTION_STYLE[sectionKey];

  const lines = body.split("\n").filter(l => l.trim() !== "");
  const isBullet = lines.some(l => /^[\*\-]\s/.test(l.trim()));

  return (
    <div className={`border-l-2 pl-3 py-0.5 ${style?.border ?? "border-gray-600"}`}>
      <p className={`text-[10px] font-bold uppercase tracking-widest mb-1 ${style?.color ?? "text-gray-400"}`}>
        {style?.icon ?? ""} {sectionKey}
      </p>
      {isBullet ? (
        <ul className="space-y-0.5">
          {lines.map((l, i) => {
            const cleaned = l.replace(/^[\*\-]\s+/, "").replace(/\*\*(.*?)\*\*/g, "$1");
            return (
              <li key={i} className="flex gap-1.5 text-xs text-gray-300 leading-relaxed">
                <span className={`mt-0.5 shrink-0 ${style?.color ?? "text-gray-400"}`}>•</span>
                <span>{cleaned}</span>
              </li>
            );
          })}
        </ul>
      ) : (
        <p className="text-xs text-gray-300 leading-relaxed">
          {body.replace(/\*\*(.*?)\*\*/g, "$1")}
        </p>
      )}
    </div>
  );
}

const LABEL_INFO = {
  GLI: {
    name: "Glioma",
    name_ko: "신경교종",
    color: "text-yellow-400",
    bg: "bg-yellow-400",
    desc: "IDH 변이 여부 및 분자 마커 확인 권고",
  },
  MEN: {
    name: "Meningioma",
    name_ko: "수막종",
    color: "text-green-400",
    bg: "bg-green-400",
    desc: "경막 기반 외부 종양 — 수술적 접근 고려",
  },
  MET: {
    name: "Metastasis",
    name_ko: "전이성 종양",
    color: "text-red-400",
    bg: "bg-red-400",
    desc: "원발성 악성종양 병력 확인 및 전신 검사 권고",
  },
};

function ProbBar({ label, prob, isMax }) {
  const info = LABEL_INFO[label];
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className={isMax ? info.color + " font-bold" : "text-gray-400"}>
          {label} — {info.name_ko}
        </span>
        <span className={isMax ? info.color + " font-bold" : "text-gray-400"}>
          {(prob * 100).toFixed(1)}%
        </span>
      </div>
      <div className="h-2 rounded-full bg-gray-700">
        <div
          className={`h-2 rounded-full transition-all duration-500 ${info.bg}`}
          style={{ width: `${prob * 100}%` }}
        />
      </div>
    </div>
  );
}

export default function ResultPanel({ result, recordId, captureViewerImage }) {
  const [report, setReport] = useState(null);
  const [loadingReport, setLoadingReport] = useState(false);
  const [reportError, setReportError] = useState(null);

  if (!result) return null;

  const { prediction, confidence, probabilities } = result;

  if (prediction === "PENDING") {
    return (
      <div className="space-y-4">
        <div className="rounded-lg p-4 border border-yellow-600 border-opacity-50 bg-yellow-900 bg-opacity-20">
          <p className="text-xs text-yellow-400 uppercase tracking-wider mb-1">상태</p>
          <h3 className="text-lg font-bold text-yellow-300">영상 업로드 완료</h3>
          <p className="text-sm text-gray-400 mt-1">MRI 파일이 서버에 저장되었습니다.</p>
        </div>
        <div className="rounded-lg p-4 bg-gray-800 border border-gray-700 text-xs text-gray-400 space-y-2">
          <p className="font-semibold text-gray-300">다음 단계 (구현 예정)</p>
          <ol className="list-decimal list-inside space-y-1">
            <li>GLI / MEN / MET Docker segmenter 실행</li>
            <li>seg_GLI.nii · seg_MEN.nii · seg_MET.nii 생성</li>
            <li>3개 seg → 피처 추출 → RF 분류</li>
          </ol>
          <p className="text-yellow-600 mt-2">
            ⚠ 현재 테스트 단계 — Docker segmenter 연동 후 분류 결과가 표시됩니다.
          </p>
        </div>
      </div>
    );
  }

  const info = LABEL_INFO[prediction];

  const handleExportPDF = () => {
    // NiiVueViewer ref에서 drawScene() → toDataURL() 순서로 캡처
    const mriDataUrl = captureViewerImage ? captureViewerImage() : "";

    const reportHtml = (report || "")
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/g, "<em>$1</em>")
      .replace(/\n/g, "<br/>");

    const printContent = `<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8"/>
  <title>뇌종양 방사선 판독 보고서</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif; color: #111; font-size: 13px; line-height: 1.7; }

    .page { width: 100%; min-height: 100vh; page-break-after: always; }
    .page:last-child { page-break-after: auto; }

    /* 1페이지: MRI 영상 */
    .page-mri {
      background: #0c0c14;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 40px;
      gap: 20px;
    }
    .page-mri .title { color: #e2e8f0; font-size: 17px; font-weight: bold; }
    .page-mri img { max-width: 100%; max-height: 72vh; object-fit: contain; border: 1px solid #2d3748; border-radius: 8px; }
    .page-mri .no-img { color: #4a5568; font-size: 14px; border: 2px dashed #2d3748; padding: 60px 80px; border-radius: 8px; text-align: center; }
    .page-mri .badge-row { display: flex; align-items: center; gap: 12px; color: #a0aec0; font-size: 12px; flex-wrap: wrap; justify-content: center; }

    /* 2페이지: 보고서 */
    .page-report { padding: 48px; background: #fff; }
    .page-report h1 { font-size: 18px; border-bottom: 2px solid #2d3748; padding-bottom: 8px; margin-bottom: 16px; }
    .page-report .meta { color: #555; font-size: 12px; margin-bottom: 24px; padding-bottom: 12px; border-bottom: 1px solid #e2e8f0; }
    .report-body { color: #1a202c; font-size: 13px; }

    .badge { display: inline-block; padding: 3px 12px; border-radius: 4px; background: #1a1a2e; color: #fff; font-weight: bold; font-size: 13px; margin-right: 6px; }
    strong { font-weight: 700; }
    em { font-style: italic; }

    @media print { .page { min-height: 100vh; } }
  </style>
</head>
<body>

  <!-- 1페이지: MRI 영상 -->
  <div class="page page-mri">
    <div class="title">🧠 뇌 MRI 영상</div>
    ${mriDataUrl
      ? `<img src="${mriDataUrl}" alt="MRI 영상" />`
      : `<div class="no-img">MRI 영상을 캡처할 수 없습니다.<br/>뷰어에서 직접 확인하세요.</div>`
    }
    <div class="badge-row">
      <span class="badge">${prediction}</span>
      <span>${info.name} — ${info.name_ko}</span>
      <span>신뢰도 ${(confidence * 100).toFixed(1)}%</span>
      <span>GLI ${(probabilities.GLI * 100).toFixed(1)}% / MEN ${(probabilities.MEN * 100).toFixed(1)}% / MET ${(probabilities.MET * 100).toFixed(1)}%</span>
    </div>
  </div>

  <!-- 2페이지: 판독 보고서 -->
  <div class="page page-report">
    <h1>🧠 뇌종양 방사선 판독 보고서</h1>
    <div class="meta">
      <span class="badge">${prediction}</span>
      신뢰도: ${(confidence * 100).toFixed(1)}%
      &nbsp;|&nbsp; GLI ${(probabilities.GLI * 100).toFixed(1)}% / MEN ${(probabilities.MEN * 100).toFixed(1)}% / MET ${(probabilities.MET * 100).toFixed(1)}%
    </div>
    <div class="report-body">${reportHtml}</div>
  </div>

  <script>window.onload = function(){ window.print(); }<\/script>
</body>
</html>`;

    const blob = new Blob([printContent], { type: "text/html;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const win = window.open(url, "_blank");
    if (win) win.focus();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
  };

  const handleGenerateReport = async () => {
    setLoadingReport(true);
    setReportError(null);
    try {
      const text = await generateReport(recordId);
      setReport(text);
    } catch (e) {
      setReportError("보고서 생성 실패: " + e.message);
    } finally {
      setLoadingReport(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* 예측 결과 헤더 */}
      <div className={`rounded-lg p-4 border border-opacity-30 bg-opacity-10 ${info.bg} border-current`}>
        <div className="flex items-start justify-between">
          <div>
            <p className="text-xs text-gray-400 uppercase tracking-wider mb-1">예측 결과</p>
            <h3 className={`text-2xl font-bold ${info.color}`}>{prediction}</h3>
            <p className="text-sm text-gray-300">{info.name} — {info.name_ko}</p>
          </div>
          <div className="text-right">
            <p className="text-xs text-gray-400">신뢰도</p>
            <p className={`text-xl font-bold ${info.color}`}>
              {(confidence * 100).toFixed(1)}%
            </p>
          </div>
        </div>
        <p className="text-xs text-gray-400 mt-2 border-t border-gray-700 pt-2">
          {info.desc}
        </p>
      </div>

      {/* 확률 바 */}
      <div className="space-y-2">
        <p className="text-xs text-gray-400 uppercase tracking-wider">분류 확률</p>
        {Object.entries(probabilities)
          .sort(([, a], [, b]) => b - a)
          .map(([label, prob]) => (
            <ProbBar
              key={label}
              label={label}
              prob={prob}
              isMax={label === prediction}
            />
          ))}
      </div>

      {/* RAG 보고서 섹션 */}
      <div>
        {!report ? (
          <button
            onClick={handleGenerateReport}
            disabled={loadingReport}
            className={`w-full py-2 rounded-lg text-sm font-medium transition border ${
              loadingReport
                ? "border-gray-600 text-gray-500 cursor-not-allowed"
                : "border-indigo-500 text-indigo-400 hover:bg-indigo-500 hover:text-white"
            }`}
          >
            {loadingReport ? (
              <span className="flex items-center justify-center gap-2">
                <span className="animate-spin w-4 h-4 border-2 border-indigo-400 border-t-transparent rounded-full" />
                보고서 생성 중 (LLM)...
              </span>
            ) : (
              "📋 방사선 소견 보고서 생성 (RAG + LLM)"
            )}
          </button>
        ) : (
          <div className="rounded-lg bg-gray-800 border border-gray-600 p-4">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-sm font-semibold text-indigo-300">
                📋 방사선 소견 보고서
              </h4>
              <div className="flex items-center gap-2">
                <button
                  onClick={handleExportPDF}
                  className="text-xs text-indigo-400 hover:text-indigo-200 border border-indigo-700 rounded px-2 py-0.5 transition"
                >
                  PDF 저장
                </button>
                <button
                  onClick={() => setReport(null)}
                  className="text-xs text-gray-500 hover:text-gray-300"
                >
                  닫기
                </button>
              </div>
            </div>
            <div className="space-y-3">
              {parseReport(report).map((sec, i) =>
                sec.key ? (
                  <ReportSection key={i} sectionKey={sec.key} body={sec.body} />
                ) : (
                  <p key={i} className="text-xs text-gray-300 leading-relaxed whitespace-pre-wrap">
                    {sec.body}
                  </p>
                )
              )}
            </div>
          </div>
        )}
        {reportError && (
          <p className="text-xs text-red-400 mt-2">{reportError}</p>
        )}
      </div>
    </div>
  );
}
