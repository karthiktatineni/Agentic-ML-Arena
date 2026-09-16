'use client';

import React, { useState, useEffect } from 'react';
import { API_BASE_URL } from '../../config';

interface ModelRegistryItem {
  run_id: string;
  model_hash: string;
  model_name?: string;
  architecture?: string;
  score?: number | null;
  timestamp?: string | null;
  created_at?: string | null;
  gate_status?: string;
  dataset_filename?: string;
  download_url?: string;
  dataset_url?: string;
  script_url?: string;
  has_dataset?: boolean;
  has_script?: boolean;
  bundle_ready?: boolean;
  hyperparameters?: Record<string, any>;
  metrics?: Record<string, any>;
}

interface ModelRegistryViewProps {
  onPredictModel?: (modelId: string) => void;
}

export default function ModelRegistryView({ onPredictModel }: ModelRegistryViewProps) {
  const [models, setModels] = useState<ModelRegistryItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [deleteTarget, setDeleteTarget] = useState<ModelRegistryItem | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [toastMessage, setToastMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const fetchModels = async () => {
    setIsLoading(true);
    try {
      // Try /api/models first, fallback to /api/predict/models
      let res = await fetch(`${API_BASE_URL}/api/models`);
      if (!res.ok) {
        res = await fetch(`${API_BASE_URL}/api/predict/models`);
      }
      if (res.ok) {
        const data = await res.json();
        const list: ModelRegistryItem[] = data.models || [];
        setModels(list);
      } else {
        setToastMessage({ type: 'error', text: 'Failed to load models from registry.' });
      }
    } catch (err: any) {
      console.error('Failed to fetch registry models', err);
      setToastMessage({ type: 'error', text: `Failed to connect to backend: ${err.message}` });
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  const handleDeleteModel = async () => {
    if (!deleteTarget) return;
    setIsDeleting(true);
    const id = deleteTarget.run_id || deleteTarget.model_hash;
    try {
      const res = await fetch(`${API_BASE_URL}/api/models/${id}`, {
        method: 'DELETE',
      });
      if (res.ok) {
        setToastMessage({ type: 'success', text: `Model ${id} and attached dataset deleted successfully.` });
        setModels((prev) => prev.filter((m) => m.run_id !== id && m.model_hash !== id));
      } else {
        const errData = await res.json().catch(() => ({}));
        setToastMessage({ type: 'error', text: errData.detail || `Failed to delete model ${id}.` });
      }
    } catch (err: any) {
      setToastMessage({ type: 'error', text: `Error deleting model: ${err.message}` });
    } finally {
      setIsDeleting(false);
      setDeleteTarget(null);
      setTimeout(() => setToastMessage(null), 4000);
    }
  };

  return (
    <div className="flex flex-col gap-6 w-full">
      {/* Toast Notification */}
      {toastMessage && (
        <div
          className={`fixed bottom-6 right-6 z-50 p-4 rounded-xl shadow-2xl flex items-center gap-3 border text-sm font-mono backdrop-blur-md transition-all ${
            toastMessage.type === 'success'
              ? 'bg-primary-container/90 text-on-primary-container border-primary-container'
              : 'bg-error-container/90 text-on-error-container border-error-container'
          }`}
        >
          <span className="material-symbols-outlined text-[20px]">
            {toastMessage.type === 'success' ? 'check_circle' : 'error'}
          </span>
          <span>{toastMessage.text}</span>
          <button
            onClick={() => setToastMessage(null)}
            className="ml-2 hover:opacity-70 text-xs"
            type="button"
          >
            ✕
          </button>
        </div>
      )}

      {/* Header Banner */}
      <header className="bg-surface-container-low rounded-xl p-6 border border-outline-variant/40 shadow-xl flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 rounded bg-primary-container/10 text-primary-container text-xs font-mono font-bold uppercase tracking-wider">
              Artifact Store
            </span>
            <span className="text-xs text-on-surface-variant font-mono">/ Production Registry</span>
          </div>
          <h1 className="text-2xl font-bold text-on-surface flex items-center gap-2">
            <span className="material-symbols-outlined text-primary text-[28px]">inventory_2</span>
            Certified Model Registry
          </h1>
          <p className="text-sm text-on-surface-variant max-w-2xl">
            Central repository of autonomous Gate-6 certified champions. Download production joblib bundles, 
            standalone CLI prediction scripts, or inspect and download the immutable snapshot of the trained data.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={fetchModels}
            disabled={isLoading}
            className="px-4 py-2 rounded-lg bg-surface-container-high hover:bg-surface-container-highest text-primary text-xs font-mono border border-outline-variant/40 flex items-center gap-2 transition-all cursor-pointer"
            type="button"
          >
            <span className={`material-symbols-outlined text-[16px] ${isLoading ? 'animate-spin' : ''}`}>
              refresh
            </span>
            <span>Refresh Registry</span>
          </button>
        </div>
      </header>

      {/* Model Inventory Table */}
      <div className="bg-surface-container-low rounded-xl border border-outline-variant/40 shadow-xl overflow-hidden flex flex-col">
        <div className="p-4 border-b border-outline-variant/30 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-sm font-mono font-bold text-on-surface">Registered Model Champions</span>
            <span className="px-2 py-0.5 rounded-full bg-surface-container-high text-xs font-mono text-primary border border-outline-variant/40">
              {models.length} {models.length === 1 ? 'Model' : 'Models'}
            </span>
          </div>
          <span className="text-xs text-on-surface-variant font-mono">
            Encrypted &amp; Audited Artifacts
          </span>
        </div>

        {isLoading ? (
          <div className="p-12 flex flex-col items-center justify-center gap-3 text-on-surface-variant font-mono text-xs">
            <span className="material-symbols-outlined text-[36px] text-primary animate-spin">
              progress_activity
            </span>
            <span>Loading certified models from registry...</span>
          </div>
        ) : models.length === 0 ? (
          <div className="p-16 flex flex-col items-center justify-center gap-4 text-center">
            <div className="w-14 h-14 rounded-full bg-surface-container-high flex items-center justify-center text-on-surface-variant border border-outline-variant/40">
              <span className="material-symbols-outlined text-[32px]">folder_off</span>
            </div>
            <div className="flex flex-col gap-1">
              <h3 className="text-base font-bold text-on-surface font-mono">No Models Registered Yet</h3>
              <p className="text-xs text-on-surface-variant max-w-md font-sans">
                Launch an experiment in the Arena and approve the candidate champion at Gate 6 to certify and store it here.
              </p>
            </div>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left font-mono text-xs">
              <thead className="bg-surface-container-lowest/80 text-[10px] uppercase text-on-surface-variant tracking-wider border-b border-outline-variant/20">
                <tr>
                  <th className="py-3 px-4">Architecture</th>
                  <th className="py-3 px-4">Run ID / Model Hash</th>
                  <th className="py-3 px-4">Validation Score</th>
                  <th className="py-3 px-4">Certification</th>
                  <th className="py-3 px-4">Trained Data</th>
                  <th className="py-3 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-outline-variant/20">
                {models.map((item) => {
                  const runId = item.run_id || 'run';
                  const hashId = item.model_hash || runId;
                  const arch = item.architecture || item.model_name || 'Champion Model';
                  const scoreVal = item.score !== null && item.score !== undefined ? Number(item.score).toFixed(4) : 'N/A';
                  const joblibUrl = `${API_BASE_URL}/api/experiments/download/${hashId}`;
                  const datasetUrl = `${API_BASE_URL}/api/experiments/download/${runId}/dataset`;
                  const scriptUrl = `${API_BASE_URL}/api/experiments/download-script/${runId}`;

                  return (
                    <tr key={`${runId}-${hashId}`} className="hover:bg-surface-container/60 transition-colors">
                      {/* Architecture */}
                      <td className="py-4 px-4">
                        <div className="flex items-center gap-2.5">
                          <div className="w-8 h-8 rounded-lg bg-primary-container/10 border border-primary-container/30 flex items-center justify-center text-primary-container">
                            <span className="material-symbols-outlined text-[18px]">neurology</span>
                          </div>
                          <div className="flex flex-col">
                            <span className="font-bold text-on-surface text-sm">{arch}</span>
                            <span className="text-[10px] text-on-surface-variant font-sans">
                              {item.timestamp ? new Date(item.timestamp).toLocaleString() : 'Production Active'}
                            </span>
                          </div>
                        </div>
                      </td>

                      {/* Run ID & Hash */}
                      <td className="py-4 px-4">
                        <div className="flex flex-col gap-1">
                          <span className="px-2 py-0.5 rounded bg-surface-container-high text-[11px] text-primary w-fit border border-outline-variant/30 font-bold">
                            RUN: {runId}
                          </span>
                          <span className="text-[10px] text-on-surface-variant/80 font-mono truncate max-w-[160px]">
                            {hashId}.joblib
                          </span>
                        </div>
                      </td>

                      {/* Score */}
                      <td className="py-4 px-4">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-bold text-primary-container">{scoreVal}</span>
                          <span className="text-[10px] text-on-surface-variant">R² / CV</span>
                        </div>
                      </td>

                      {/* Gate Status */}
                      <td className="py-4 px-4">
                        <span className="px-2.5 py-1 rounded-full bg-tertiary-container/15 text-tertiary-fixed-dim text-[10px] font-bold border border-tertiary-container/30 inline-flex items-center gap-1">
                          <span className="material-symbols-outlined text-[12px]">verified</span>
                          {item.gate_status || 'Gate 6 Certified'}
                        </span>
                      </td>

                      {/* Trained Dataset */}
                      <td className="py-4 px-4">
                        <a
                          href={datasetUrl}
                          download={`dataset_${runId}.csv`}
                          className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-surface-container hover:bg-surface-container-high text-primary hover:text-primary-container border border-outline-variant/40 text-xs transition-all"
                          title="Download training data snapshot CSV"
                        >
                          <span className="material-symbols-outlined text-[14px]">table_chart</span>
                          <span>dataset.csv</span>
                          <span className="material-symbols-outlined text-[12px]">download</span>
                        </a>
                      </td>

                      {/* Actions */}
                      <td className="py-4 px-4 text-right">
                        <div className="flex items-center justify-end gap-2">
                          {/* Test in inference console */}
                          {onPredictModel && (
                            <button
                              onClick={() => onPredictModel(hashId)}
                              className="px-2.5 py-1.5 rounded bg-primary-container/20 hover:bg-primary-container text-on-primary-container text-xs font-mono transition-all flex items-center gap-1 border border-primary-container/40"
                              title="Test model in real-time inference console"
                              type="button"
                            >
                              <span className="material-symbols-outlined text-[14px]">psychology</span>
                              <span>Infer</span>
                            </button>
                          )}

                          {/* Download Joblib */}
                          <a
                            href={joblibUrl}
                            download={`${hashId}.joblib`}
                            className="px-2.5 py-1.5 rounded bg-surface-container-high hover:bg-surface-container-highest text-primary text-xs font-mono transition-all flex items-center gap-1 border border-outline-variant/40"
                            title="Download trained model binary (.joblib)"
                          >
                            <span className="material-symbols-outlined text-[14px]">download</span>
                            <span>.joblib</span>
                          </a>

                          {/* Download predict.py */}
                          <a
                            href={scriptUrl}
                            download="predict.py"
                            className="px-2.5 py-1.5 rounded bg-surface-container-high hover:bg-surface-container-highest text-on-surface text-xs font-mono transition-all flex items-center gap-1 border border-outline-variant/40"
                            title="Download standalone predict.py CLI script"
                          >
                            <span className="material-symbols-outlined text-[14px]">code</span>
                            <span>predict.py</span>
                          </a>

                          {/* Delete */}
                          <button
                            onClick={() => setDeleteTarget(item)}
                            className="p-1.5 rounded hover:bg-error-container/20 text-error/80 hover:text-error transition-all border border-transparent hover:border-error-container/40 cursor-pointer"
                            title="Delete model and artifacts from disk"
                            type="button"
                          >
                            <span className="material-symbols-outlined text-[16px]">delete</span>
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Delete Confirmation Modal */}
      {deleteTarget && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-surface-container-low border-2 border-error-container/60 rounded-2xl p-6 max-w-md w-full shadow-2xl flex flex-col gap-4 font-mono">
            <div className="flex items-center gap-3 text-error">
              <div className="w-10 h-10 rounded-full bg-error-container/20 flex items-center justify-center border border-error-container/40">
                <span className="material-symbols-outlined text-[24px]">warning</span>
              </div>
              <div>
                <h3 className="text-base font-bold text-on-surface">Delete Certified Model?</h3>
                <span className="text-xs text-on-surface-variant">Destructive File Operation</span>
              </div>
            </div>

            <p className="text-xs text-on-surface-variant font-sans leading-relaxed">
              This action will permanently delete the certified model bundle{' '}
              <strong className="text-on-surface font-mono">{deleteTarget.run_id}</strong> (
              <span className="text-primary font-mono">{deleteTarget.model_hash}</span>), 
              the attached <strong className="text-on-surface font-mono">dataset.csv</strong>, and remove its entry from the production registry.
            </p>

            <div className="flex items-center justify-end gap-3 pt-2 border-t border-outline-variant/30">
              <button
                onClick={() => setDeleteTarget(null)}
                disabled={isDeleting}
                className="px-4 py-2 rounded-lg bg-surface-container-high hover:bg-surface-container-highest text-on-surface text-xs font-mono transition-all cursor-pointer"
                type="button"
              >
                Cancel
              </button>
              <button
                onClick={handleDeleteModel}
                disabled={isDeleting}
                className="px-4 py-2 rounded-lg bg-error text-on-error hover:bg-error/90 text-xs font-mono font-bold transition-all flex items-center gap-1.5 shadow-lg shadow-error/30 cursor-pointer"
                type="button"
              >
                {isDeleting ? (
                  <>
                    <span className="material-symbols-outlined text-[16px] animate-spin">progress_activity</span>
                    <span>Deleting...</span>
                  </>
                ) : (
                  <>
                    <span className="material-symbols-outlined text-[16px]">delete_forever</span>
                    <span>Confirm Delete</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
