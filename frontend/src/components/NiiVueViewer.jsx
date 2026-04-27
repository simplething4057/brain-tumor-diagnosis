import { useEffect, useRef, useState, forwardRef, useImperativeHandle } from "react";
import { Niivue } from "@niivue/niivue";
import { getFileUrl } from "../utils/api";

const NiiVueViewer = forwardRef(function NiiVueViewer(
  { subjectId, availableModalities, fileExts = {} },
  ref
) {
  const canvasRef = useRef(null);
  const nvRef = useRef(null);
  const [activeModality, setActiveModality] = useState(
    availableModalities.includes("t1c") ? "t1c" : availableModalities[0]
  );
  const [showSeg, setShowSeg] = useState(true);
  const [sliceType, setSliceType] = useState(2);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // 부모(App)에서 captureImage()를 호출할 수 있도록 노출
  useImperativeHandle(ref, () => ({
    captureImage() {
      const nv = nvRef.current;
      const canvas = canvasRef.current;
      if (!nv || !canvas) return "";
      try {
        // drawScene()으로 프레임버퍼를 채운 직후 toDataURL() 호출
        nv.drawScene();
        return canvas.toDataURL("image/png");
      } catch (e) {
        console.warn("MRI 캡처 실패:", e);
        return "";
      }
    },
  }));

  useEffect(() => {
    if (!canvasRef.current || !subjectId) return;

    // NiiVue는 preserveDrawingBuffer: false로 WebGL 컨텍스트를 생성해
    // toDataURL()이 빈 이미지를 반환합니다.
    // NiiVue 초기화 전에 이 캔버스의 getContext만 패치해 true로 강제합니다.
    const canvas = canvasRef.current;
    const _origGetContext = canvas.getContext.bind(canvas);
    canvas.getContext = (type, attrs) =>
      _origGetContext(type, { ...attrs, preserveDrawingBuffer: true });

    const nv = new Niivue({
      show3Dcrosshair: true,
      backColor: [0.07, 0.07, 0.1, 1],
      crosshairColor: [1, 0, 0, 0.8],
      selectionBoxColor: [1, 1, 1, 0.5],
      isColorbar: false,
    });

    // 패치 복원 (다른 컨텍스트에 영향 없도록)
    canvas.getContext = _origGetContext;

    nv.attachToCanvas(canvasRef.current);
    nvRef.current = nv;

    loadVolumes(nv, subjectId, activeModality, showSeg, fileExts);

    return () => {};
  }, [subjectId]);

  const loadVolumes = async (nv, sid, modality, withSeg, exts = {}) => {
    setIsLoading(true);
    setError(null);
    try {
      const volumes = [];
      const isSegOnly = modality === "seg";

      if (!isSegOnly) {
        const modExt = exts[modality] || ".nii";
        volumes.push({
          url: getFileUrl(sid, modality),
          name: `${modality}${modExt}`,
          colormap: "gray",
          opacity: 1.0,
          visible: true,
        });
      }

      if (withSeg && availableModalities.includes("seg")) {
        const segExt = exts["seg"] || ".nii";
        volumes.push({
          url: getFileUrl(sid, "seg"),
          name: `seg${segExt}`,
          colormap: "warm",
          opacity: isSegOnly ? 1.0 : 0.8,
          visible: true,
        });
      }

      await nv.loadVolumes(volumes);

      const segVolIdx = nv.volumes.length - 1;
      if (nv.volumes.length > 0 && availableModalities.includes("seg")) {
        const segVol = nv.volumes[segVolIdx];
        segVol.cal_min = 0.5;
        segVol.cal_max = 4.0;
        nv.updateGLVolume();
      }

      nv.setSliceType(sliceType);
    } catch (e) {
      setError(`NiiVue 로드 실패: ${e.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleModalityChange = async (mod) => {
    setActiveModality(mod);
    if (nvRef.current && subjectId) {
      await loadVolumes(nvRef.current, subjectId, mod, showSeg, fileExts);
    }
  };

  const handleToggleSeg = async () => {
    const next = !showSeg;
    setShowSeg(next);
    if (nvRef.current && subjectId) {
      await loadVolumes(nvRef.current, subjectId, activeModality, next, fileExts);
    }
  };

  const handleSliceType = (type) => {
    setSliceType(type);
    if (nvRef.current) nvRef.current.setSliceType(type);
  };

  return (
    <div className="flex flex-col h-full bg-gray-900 rounded-lg overflow-hidden border border-gray-700">
      {/* 뷰어 툴바 */}
      <div className="flex items-center gap-2 px-3 py-2 bg-gray-800 border-b border-gray-700 flex-wrap">
        <span className="text-xs text-gray-400 mr-1">모달리티:</span>
        {availableModalities
          .filter((m) => m !== "seg")
          .map((mod) => (
            <button
              key={mod}
              onClick={() => handleModalityChange(mod)}
              className={`px-2 py-1 rounded text-xs font-mono uppercase transition ${
                activeModality === mod
                  ? "bg-brand-500 text-white"
                  : "bg-gray-700 text-gray-300 hover:bg-gray-600"
              }`}
            >
              {mod}
            </button>
          ))}

        <div className="w-px h-4 bg-gray-600 mx-1" />

        <button
          onClick={handleToggleSeg}
          className={`px-2 py-1 rounded text-xs transition ${
            showSeg
              ? "bg-orange-600 text-white"
              : "bg-gray-700 text-gray-400 hover:bg-gray-600"
          }`}
        >
          {showSeg ? "SEG ON" : "SEG OFF"}
        </button>

        <div className="w-px h-4 bg-gray-600 mx-1" />

        <span className="text-xs text-gray-400">뷰:</span>
        {[
          { label: "Axial", val: 0 },
          { label: "Multi", val: 2 },
          { label: "3D", val: 3 },
        ].map(({ label, val }) => (
          <button
            key={val}
            onClick={() => handleSliceType(val)}
            className={`px-2 py-1 rounded text-xs transition ${
              sliceType === val
                ? "bg-indigo-600 text-white"
                : "bg-gray-700 text-gray-300 hover:bg-gray-600"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* 색상 범례 */}
      {showSeg && availableModalities.includes("seg") && (
        <div className="flex items-center gap-4 px-3 py-1.5 bg-gray-800 border-b border-gray-700 text-xs text-gray-400">
          <span className="font-medium text-gray-300">SEG 범례:</span>
          {[
            { label: "NCR (괴사)", color: "#992200" },
            { label: "ED (부종)",  color: "#cc6600" },
            { label: "ET (조영증강)", color: "#ffee00" },
          ].map(({ label, color }) => (
            <span key={label} className="flex items-center gap-1">
              <span
                className="inline-block w-3 h-3 rounded-sm border border-gray-600"
                style={{ background: color }}
              />
              {label}
            </span>
          ))}
        </div>
      )}

      {/* 캔버스 */}
      <div className="relative flex-1 min-h-0">
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-80 z-10">
            <div className="text-center">
              <div className="animate-spin w-8 h-8 border-4 border-brand-500 border-t-transparent rounded-full mx-auto mb-2" />
              <p className="text-sm text-gray-300">MRI 로딩 중...</p>
            </div>
          </div>
        )}
        {error && (
          <div className="absolute inset-0 flex items-center justify-center">
            <p className="text-red-400 text-sm">{error}</p>
          </div>
        )}
        <canvas
          ref={canvasRef}
          className="w-full h-full"
          style={{ touchAction: "none" }}
        />
      </div>
    </div>
  );
});

export default NiiVueViewer;
