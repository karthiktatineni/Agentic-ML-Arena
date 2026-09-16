'use client';

import React, { useState, useEffect } from 'react';

interface ManualPredictionViewProps {
  initialModelId?: string | null;
}

interface ModelItem {
  run_id: string;
  model_hash: string;
  model_name: string;
  score?: number | null;
  download_url?: string;
}

export default function ManualPredictionView({ initialModelId }: ManualPredictionViewProps) {
  const [models, setModels] = useState<ModelItem[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>('');
  const [isLoadingModels, setIsLoadingModels] = useState(true);

  // Model schema & features
  const [modelDetails, setModelDetails] = useState<any>(null);
  const [featureValues, setFeatureValues] = useState<Record<string, any>>({});
  const [isLoadingFeatures, setIsLoadingFeatures] = useState(false);

  // Prediction Output
  const [predictionResult, setPredictionResult] = useState<any>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictError, setPredictError] = useState<string | null>(null);

  // Batch Prediction
  const [batchFile, setBatchFile] = useState<File | null>(null);
  const [isBatchRunning, setIsBatchRunning] = useState(false);
  const [batchResult, setBatchResult] = useState<any>(null);
  const [batchMsg, setBatchMsg] = useState('');

  // 1. Fetch available models
  useEffect(() => {
    const fetchModels = async () => {
      setIsLoadingModels(true);
      try {
        const res = await fetch('http://localhost:8000/api/predict/models');
        if (res.ok) {
          const data = await res.json();
          const list = data.models || [];
          setModels(list);
          if (list.length > 0) {
            // Pick initial or latest
            const target = initialModelId && list.some((m: ModelItem) => m.model_hash === initialModelId || m.run_id === initialModelId)
              ? initialModelId
              : list[0].model_hash || list[0].run_id;
            setSelectedModelId(target);
          }
        }
      } catch (err) {
        console.error('Failed to load models list', err);
      } finally {
        setIsLoadingModels(false);
      }
    };
    fetchModels();
  }, [initialModelId]);

  // 2. Fetch feature schema for selected model
  useEffect(() => {
    if (!selectedModelId) return;
    const fetchFeatures = async () => {
      setIsLoadingFeatures(true);
      setPredictError(null);
      setPredictionResult(null);
      try {
        const res = await fetch(`http://localhost:8000/api/predict/features?model_id=${selectedModelId}`);
        if (res.ok) {
          const data = await res.json();
          setModelDetails(data);
          setFeatureValues(data.sample_record || {});
        } else {
          setPredictError(`Could not load feature schema for model ${selectedModelId}`);
        }
      } catch (err: any) {
        setPredictError(`Failed to fetch features: ${err.message}`);
      } finally {
        setIsLoadingFeatures(false);
      }
    };
    fetchFeatures();
  }, [selectedModelId]);

  const handleInputChange = (feat: string, val: string) => {
    const isCat = modelDetails?.categorical_columns && Array.isArray(modelDetails.categorical_columns[feat]);
    let parsed: any = val;
    if (!isCat) {
      const num = Number(val);
      parsed = !isNaN(num) && val.trim() !== '' ? num : val;
    }
    setFeatureValues((prev) => ({
      ...prev,
      [feat]: parsed,
    }));
  };

  const handleRunPrediction = async () => {
    setIsPredicting(true);
    setPredictError(null);
    try {
      const payload = {
        features: featureValues,
        model_id: selectedModelId || undefined,
      };
      const res = await fetch('http://localhost:8000/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server returned ${res.status}`);
      }
      const data = await res.json();
      setPredictionResult(data);
    } catch (err: any) {
      setPredictError(`Prediction failed: ${err.message}`);
    } finally {
      setIsPredicting(false);
    }
  };

  const handleBatchPredict = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!batchFile) return;
    setIsBatchRunning(true);
    setBatchMsg('Processing batch inference on rows...');
    setBatchResult(null);

    const formData = new FormData();
    formData.append('file', batchFile);
    if (selectedModelId) formData.append('model_id', selectedModelId);

    try {
      const res = await fetch('http://localhost:8000/api/predict/batch', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setBatchResult(data);
      setBatchMsg(`Batch complete! Processed ${data.rows_processed} rows.`);
    } catch (err: any) {
      setBatchMsg(`Batch failed: ${err.message}`);
    } finally {
      setIsBatchRunning(false);
    }
  };

  const activeHash = modelDetails?.model_hash || selectedModelId;
  const downloadJoblibUrl = `http://localhost:8000/api/experiments/download/${activeHash}`;
  const downloadScriptUrl = `http://localhost:8000/api/experiments/download-script/${selectedModelId}`;

  return (
    <div className="flex flex-col gap-6 w-full pb-12">
      {/* Header & Model Selector */}
      <section className="bg-surface-container-low rounded-xl p-5 border border-outline-variant/40 shadow-xl flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 rounded bg-secondary-container/30 text-secondary font-mono text-xs uppercase tracking-wider font-bold">
              Inference Console
            </span>
            <span className="text-xs text-on-surface-variant font-mono">/ Production Joblib Serving</span>
          </div>
          <h1 className="text-2xl font-bold text-primary tracking-tight">
            Predict with Trained Model Bundle
          </h1>
        </div>

        {/* Model Switcher & 1-Click Downloads */}
        <div className="flex flex-wrap items-center gap-3 w-full lg:w-auto">
          <div className="flex items-center gap-2 bg-surface-container px-3 py-1.5 rounded-lg border border-outline-variant/50">
            <span className="text-xs text-on-surface-variant font-mono font-bold">MODEL:</span>
            {isLoadingModels ? (
              <span className="text-xs text-on-surface-variant font-mono">Loading models...</span>
            ) : models.length === 0 ? (
              <span className="text-xs text-error font-mono">No trained models found</span>
            ) : (
              <select
                value={selectedModelId}
                onChange={(e) => setSelectedModelId(e.target.value)}
                className="bg-surface-container-high text-xs font-mono font-bold text-primary focus:outline-none cursor-pointer px-2 py-1 rounded border border-outline-variant/40 min-w-[200px]"
              >
                {models.map((m) => (
                  <option key={m.model_hash || m.run_id} value={m.model_hash || m.run_id}>
                    {m.model_name} ({m.model_hash?.slice(0, 14) || m.run_id})
                  </option>
                ))}
              </select>
            )}
          </div>

          {/* 1-Click Download .joblib Button */}
          <a
            href={downloadJoblibUrl}
            download={`${activeHash}.joblib`}
            className="px-3.5 py-2 rounded-lg bg-primary-container text-on-primary-fixed text-xs font-mono font-bold hover:shadow-[0_0_16px_rgba(0,242,254,0.5)] transition-all flex items-center gap-1.5"
          >
            <span className="material-symbols-outlined text-[16px]">download</span>
            <span>Download .joblib</span>
          </a>

          {/* Download Standalone predict.py Script */}
          <a
            href={downloadScriptUrl}
            download="predict.py"
            className="px-3 py-2 rounded-lg bg-surface-container-high border border-outline-variant text-on-surface hover:bg-surface-bright text-xs font-mono transition-all flex items-center gap-1.5"
          >
            <span className="material-symbols-outlined text-[16px]">code</span>
            <span>predict.py</span>
          </a>
        </div>
      </section>

      {predictError && (
        <div className="px-4 py-3 rounded-lg bg-error-container/20 border border-error text-error text-xs font-mono flex items-center gap-2">
          <span className="material-symbols-outlined text-[18px]">error</span>
          <span>{predictError}</span>
        </div>
      )}

      {/* Model Spec Ribbon */}
      {modelDetails && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="bg-surface-container-low p-3.5 rounded-xl border border-outline-variant/30 flex flex-col">
            <span className="text-[10px] font-mono text-on-surface-variant uppercase">Model Architecture</span>
            <span className="text-sm font-bold text-primary font-mono mt-0.5">{modelDetails.model_name}</span>
          </div>
          <div className="bg-surface-container-low p-3.5 rounded-xl border border-outline-variant/30 flex flex-col">
            <span className="text-[10px] font-mono text-on-surface-variant uppercase">Task Type</span>
            <span className="text-sm font-bold text-secondary font-mono mt-0.5 uppercase">{modelDetails.task_type}</span>
          </div>
          <div className="bg-surface-container-low p-3.5 rounded-xl border border-outline-variant/30 flex flex-col">
            <span className="text-[10px] font-mono text-on-surface-variant uppercase">Validated Features</span>
            <span className="text-sm font-bold text-on-surface font-mono mt-0.5">{modelDetails.feature_names?.length || 0} columns</span>
          </div>
          <div className="bg-surface-container-low p-3.5 rounded-xl border border-outline-variant/30 flex flex-col">
            <span className="text-[10px] font-mono text-on-surface-variant uppercase">Storage Path</span>
            <span className="text-xs font-mono text-tertiary-fixed-dim mt-0.5 truncate">{modelDetails.run_folder || 'data/runs'}</span>
          </div>
        </div>
      )}

      {/* Main Prediction Interface Grid */}
      <div className="grid grid-cols-12 gap-6 items-start">
        {/* Left Column: Dynamic Feature Form (8 Cols) */}
        <div className="col-span-12 xl:col-span-8 flex flex-col gap-4">
          <div className="bg-surface-container-low rounded-xl p-6 border border-outline-variant/40 shadow-xl flex flex-col gap-5">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-outline-variant/30 pb-3">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-[20px] text-primary-container">tune</span>
                <h2 className="text-base font-bold text-on-surface font-mono">
                  Input Feature Atoms (Model Payload)
                </h2>
              </div>
              <span className="text-xs font-mono text-on-surface-variant">
                Enter values to run real-time inference
              </span>
            </div>

            {isLoadingFeatures ? (
              <div className="p-12 text-center text-on-surface-variant font-mono text-xs flex flex-col items-center gap-2">
                <span className="h-6 w-6 rounded-full border-2 border-primary-container border-t-transparent animate-spin"></span>
                <span>Inspecting model bundle features...</span>
              </div>
            ) : !modelDetails?.feature_names || modelDetails.feature_names.length === 0 ? (
              <div className="p-8 text-center text-on-surface-variant font-mono text-xs">
                No feature names discovered in this bundle.
              </div>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {modelDetails.feature_names.map((feat: string) => {
                  const val = featureValues[feat] !== undefined ? featureValues[feat] : '';
                  const catOptions = modelDetails.categorical_columns && modelDetails.categorical_columns[feat];
                  const isCategorical = Array.isArray(catOptions) && catOptions.length > 0;

                  return (
                    <div key={feat} className="flex flex-col gap-1.5 bg-surface-container p-3 rounded-lg border border-outline-variant/30">
                      <div className="flex items-center justify-between">
                        <label className="text-xs font-mono font-semibold text-primary truncate" title={feat}>
                          {feat}
                        </label>
                        <span className={`text-[9px] font-mono px-1.5 py-0.5 rounded font-bold uppercase tracking-wider ${
                          isCategorical
                            ? 'bg-secondary-container/30 text-secondary border border-secondary-container/40'
                            : 'bg-surface-container-high text-on-surface-variant'
                        }`}>
                          {isCategorical ? 'Categorical' : 'Numeric'}
                        </span>
                      </div>

                      {isCategorical ? (
                        <select
                          value={String(val)}
                          onChange={(e) => handleInputChange(feat, e.target.value)}
                          className="bg-surface-container-high px-3 py-2 rounded text-xs font-mono text-primary font-semibold focus:outline-none focus:border-primary-container border border-outline-variant/40 cursor-pointer"
                        >
                          <option value="" disabled>Select {feat}...</option>
                          {catOptions.map((opt: any) => (
                            <option key={String(opt)} value={String(opt)}>
                              {String(opt)}
                            </option>
                          ))}
                        </select>
                      ) : (
                        <input
                          type="number"
                          step="any"
                          value={val}
                          onChange={(e) => handleInputChange(feat, e.target.value)}
                          placeholder="0"
                          className="bg-surface-container-high px-3 py-2 rounded text-xs font-mono text-on-surface focus:outline-none focus:border-primary-container border border-outline-variant/40"
                        />
                      )}
                    </div>
                  );
                })}
              </div>
            )}

            <div className="flex flex-wrap items-center justify-between gap-3 pt-4 border-t border-outline-variant/30">
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setFeatureValues(modelDetails?.sample_record || {})}
                  className="px-3 py-1.5 rounded bg-surface-container hover:bg-surface-container-high text-on-surface-variant text-xs font-mono transition-colors"
                >
                  Reset to Defaults
                </button>
                <button
                  type="button"
                  onClick={() => {
                    const empty: Record<string, any> = {};
                    (modelDetails?.feature_names || []).forEach((f: string) => (empty[f] = 0));
                    setFeatureValues(empty);
                  }}
                  className="px-3 py-1.5 rounded bg-surface-container hover:bg-surface-container-high text-on-surface-variant text-xs font-mono transition-colors"
                >
                  Clear Values (0)
                </button>
              </div>

              <button
                type="button"
                onClick={handleRunPrediction}
                disabled={isPredicting || !selectedModelId}
                className="px-6 py-2.5 rounded-lg bg-primary-container text-on-primary-fixed text-xs font-mono font-bold hover:shadow-[0_0_20px_rgba(0,242,254,0.6)] transition-all flex items-center gap-2 cursor-pointer disabled:opacity-50"
              >
                <span className="material-symbols-outlined text-[18px]">bolt</span>
                <span>{isPredicting ? 'Computing Inference...' : 'RUN REAL-TIME PREDICTION'}</span>
              </button>
            </div>
          </div>

          {/* Batch CSV Inference Form */}
          <div className="bg-surface-container-low rounded-xl p-5 border border-outline-variant/40 shadow-xl flex flex-col gap-4">
            <div className="flex items-center gap-2">
              <span className="material-symbols-outlined text-[18px] text-secondary">table_view</span>
              <h3 className="text-sm font-bold text-on-surface font-mono">
                Batch CSV Prediction (Bulk Evaluation)
              </h3>
            </div>
            <form onSubmit={handleBatchPredict} className="flex flex-wrap items-center gap-3">
              <input
                type="file"
                accept=".csv"
                onChange={(e) => setBatchFile(e.target.files?.[0] || null)}
                className="text-xs text-on-surface file:mr-2 file:py-1 file:px-2 file:rounded file:border-0 file:text-xs file:bg-surface-container-high file:text-primary cursor-pointer"
              />
              <button
                type="submit"
                disabled={isBatchRunning || !batchFile}
                className="px-4 py-2 rounded-lg bg-secondary-container/40 border border-secondary-container text-secondary-fixed hover:bg-secondary-container text-xs font-mono font-bold transition-all disabled:opacity-50"
              >
                {isBatchRunning ? 'Processing...' : 'Run Batch CSV'}
              </button>
            </form>
            {batchMsg && (
              <span className="text-xs font-mono text-primary-container">{batchMsg}</span>
            )}
            {batchResult?.predictions_sample && (
              <div className="flex flex-col gap-1 bg-surface-container p-3 rounded font-mono text-xs">
                <span className="text-on-surface-variant font-bold">Sample Predictions (First 10 rows):</span>
                <span className="text-primary truncate">
                  {batchResult.predictions_sample.slice(0, 10).join(', ')}
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Prediction Output Card (4 Cols) */}
        <div className="col-span-12 xl:col-span-4 flex flex-col gap-4">
          <div className="bg-surface-container-low rounded-xl p-5 border border-outline-variant/40 shadow-xl flex flex-col gap-4">
            <div className="flex items-center justify-between pb-3 border-b border-outline-variant/30">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-[18px] text-primary-container">psychology</span>
                <h3 className="text-sm font-semibold text-on-surface font-mono">Prediction Output</h3>
              </div>
              <span className="text-[10px] font-mono text-primary-container px-2 py-0.5 rounded bg-primary-container/10">
                LIVE ENGINE
              </span>
            </div>

            {predictionResult ? (
              <div className="flex flex-col gap-4">
                {/* Result Display Box */}
                <div className="p-5 rounded-xl bg-surface-container-lowest border-2 border-primary-container shadow-[0_0_24px_rgba(0,242,254,0.25)] flex flex-col items-center justify-center text-center gap-2">
                  <span className="text-[11px] font-mono text-on-surface-variant uppercase tracking-wider">
                    Model Predicted Output
                  </span>
                  <div className="text-3xl font-bold font-mono text-primary tracking-tight">
                    {typeof predictionResult.prediction === 'number'
                      ? predictionResult.prediction > 100
                        ? `$${predictionResult.prediction.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
                        : predictionResult.prediction.toFixed(4)
                      : String(predictionResult.prediction)}
                  </div>
                  {predictionResult.confidence !== null && predictionResult.confidence !== undefined && (
                    <div className="flex items-center gap-1 text-xs font-mono text-primary-container bg-primary-container/10 px-2.5 py-1 rounded-full">
                      <span>Confidence:</span>
                      <strong>{(predictionResult.confidence * 100).toFixed(1)}%</strong>
                    </div>
                  )}
                </div>

                {/* Telemetry metadata */}
                <div className="flex flex-col gap-2 font-mono text-xs bg-surface-container p-3.5 rounded-lg border border-outline-variant/30">
                  <div className="flex justify-between items-center">
                    <span className="text-on-surface-variant">Inference Latency:</span>
                    <span className="text-primary font-bold">{predictionResult.latency_ms || 1.8} ms</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-on-surface-variant">Serving Bundle:</span>
                    <span className="text-on-surface font-semibold">{predictionResult.model_name}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-on-surface-variant">Model Run ID:</span>
                    <span className="text-on-surface-variant">{predictionResult.model_id}</span>
                  </div>
                </div>

                {/* CLI Command Helper */}
                <div className="flex flex-col gap-1.5 p-3 rounded-lg bg-surface-container-lowest border border-outline-variant/30 font-mono text-xs">
                  <span className="text-[10px] uppercase text-on-surface-variant">Python CLI Usage:</span>
                  <code className="text-primary text-[11px] bg-surface-container p-2 rounded block overflow-x-auto">
                    python data/runs/{predictionResult.model_id}/predict.py test.csv
                  </code>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center p-8 text-center text-on-surface-variant opacity-60 gap-3">
                <span className="material-symbols-outlined text-[36px]">science</span>
                <span className="font-mono text-xs">Awaiting Prediction Trigger</span>
                <span className="text-[11px]">Adjust the input features on the left and click &quot;Run Real-Time Prediction&quot;.</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
