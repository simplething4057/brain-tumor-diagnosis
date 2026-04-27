import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";

// seg: 있으면 기존 파이프라인(seg → RF), 없으면 MRI → Docker segmenter 경로
const MODALITIES = [
  { key: "t1c", label: "T1C", required: false, color: "blue" },
  { key: "t1n", label: "T1N", required: false, color: "blue" },
  { key: "t2f", label: "T2F / FLAIR", required: false, color: "blue" },
  { key: "t2w", label: "T2W", required: false, color: "blue" },
  { key: "seg", label: "Segmentation", required: false, color: "orange" },
];

function ModalityDropzone({ modality, file, onFile }) {
  const { required, label, color } = modality;
  const onDrop = useCallback(
    (accepted) => {
      if (accepted.length > 0) onFile(modality.key, accepted[0]);
    },
    [modality.key, onFile]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    // .nii 파일은 브라우저마다 MIME 타입이 달라 accept 제한 없이 확장자로만 검증
    multiple: false,
  });

  return (
    <div
      {...getRootProps()}
      className={`border-2 border-dashed rounded-lg p-3 cursor-pointer transition text-center ${
        isDragActive
          ? "border-brand-500 bg-brand-500 bg-opacity-10"
          : file
          ? "border-green-500 bg-green-500 bg-opacity-10"
          : required
          ? "border-orange-500 border-opacity-60 hover:border-orange-400"
          : "border-gray-600 hover:border-gray-500"
      }`}
    >
      <input {...getInputProps()} />
      <div className="flex items-center justify-center gap-2">
        <span
          className={`text-xs font-semibold uppercase ${
            required ? "text-orange-400" : "text-blue-400"
          }`}
        >
          {label}
          {required && <span className="text-red-400 ml-0.5">*</span>}
        </span>
      </div>
      {file ? (
        <p className="text-xs text-green-400 mt-1 truncate">{file.name}</p>
      ) : (
        <p className="text-xs text-gray-500 mt-1">
          {isDragActive ? "놓으세요" : ".nii / .nii.gz"}
        </p>
      )}
    </div>
  );
}

export default function FileUploader({ onPredict, isLoading }) {
  const [files, setFiles] = useState({});

  const handleFile = useCallback((key, file) => {
    setFiles((prev) => ({ ...prev, [key]: file }));
  }, []);

  const hasMRI = ["t1c", "t1n", "t2f", "t2w"].some((k) => !!files[k]);

  const handleSubmit = () => {
    if (!files.seg && !hasMRI) {
      alert("seg.nii 또는 MRI 파일(T1C/T1N/T2F/T2W) 중 최소 1개를 업로드해 주세요.");
      return;
    }
    onPredict(files);
  };

  const canSubmit = (!!files.seg || hasMRI) && !isLoading;

  return (
    <div className="space-y-3">
      <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">
        MRI 파일 업로드
      </h2>

      <div className="grid grid-cols-1 gap-2">
        {MODALITIES.map((mod) => (
          <ModalityDropzone
            key={mod.key}
            modality={mod}
            file={files[mod.key]}
            onFile={handleFile}
          />
        ))}
      </div>

      <button
        onClick={handleSubmit}
        disabled={!canSubmit}
        className={`w-full py-2.5 rounded-lg text-sm font-semibold transition ${
          canSubmit
            ? "bg-brand-500 hover:bg-brand-600 text-white"
            : "bg-gray-700 text-gray-500 cursor-not-allowed"
        }`}
      >
        {isLoading ? (
          <span className="flex items-center justify-center gap-2">
            <span className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full" />
            분석 중...
          </span>
        ) : (
          "🔬 분류 시작"
        )}
      </button>

      <p className="text-xs text-gray-500 text-center">
        MRI(t1c/t1n/t2f/t2w) 또는 seg — 최소 1개 필요 (.nii / .nii.gz)
      </p>
      {!files.seg && hasMRI && (
        <p className="text-xs text-yellow-500 text-center">
          ⚠ seg 없음 — BraTS 자동 분할 후 분류 (CPU 모드, 수십 분 소요)
        </p>
      )}
    </div>
  );
}
