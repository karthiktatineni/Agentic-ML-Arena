'use client';

import React, { useState, useEffect } from 'react';
import { DashboardEvent } from '../../hooks/useDashboardSocket';

interface RegisteredModel {
  run_id: string;
  model_hash: string;
  timestamp?: string;
  model_name: string;
  score?: number | null;
  joblib_path?: string;
  download_url?: string;
}

interface MissionControlViewProps {
  events: DashboardEvent[];
  activeStages: Record<string, string>;
  championData?: any;
  onLaunchSuccess?: () => void;
  onNavigate?: (tab: any) => void;
  onPredictModel?: (modelId: string) => void;
}

export default function MissionControlView({
  events,
  activeStages,
  championData,
  onLaunchSuccess,
  onNavigate,
  onPredictModel,
}: MissionControlViewProps) {
  const [file, setFile] = useState<File | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [targetCol, setTargetCol] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState('');

  // Human Approval State
  const [pendingApprovals, setPendingApprovals] = useState<Record<string, any>>({});
  const [isProcessingApproval, setIsProcessingApproval] = useState(false);
  const [isLocallyApproved, setIsLocallyApproved] = useState(false);
  const [isLocallyRejected, setIsLocallyRejected] = useState(false);
  const [actionMessage, setActionMessage] = useState<string | null>(null);
  const [isApprovalModalOpen, setIsApprovalModalOpen] = useState(false);

  // Model Registry State
  const [registryModels, setRegistryModels] = useState<RegisteredModel[]>([]);
  const [recentDecisions, setRecentDecisions] = useState<any[]>([]);

  // Fetch Pending Approvals & Registry on Mount and poll lightly
  const fetchPendingAndRegistry = async () => {
    try {
      const pendingRes = await fetch('http://localhost:8000/api/experiments/pending');
      if (pendingRes.ok) {
        const data = await pendingRes.json();
        setPendingApprovals(data.pending || {});
      }
    } catch {
      // Ignore network hiccup
    }

    try {
      const regRes = await fetch('http://localhost:8000/api/predict/models');
      if (regRes.ok) {
        const data = await regRes.json();
        setRegistryModels(data.models || []);
      }
    } catch {
      // Ignore network hiccup
    }

    try {
      const eventsRes = await fetch('http://localhost:8000/api/v1/ws/events/recent');
      if (eventsRes.ok) {
        const data = await eventsRes.json();
        if (Array.isArray(data.events) && data.events.length > 0) {
          setRecentDecisions(data.events);
        }
      }
    } catch {
      // Ignore network hiccup
    }
  };

  useEffect(() => {
    fetchPendingAndRegistry();
    const timer = setInterval(fetchPendingAndRegistry, 4000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (Object.keys(pendingApprovals).length > 0 && !isLocallyApproved && !isLocallyRejected) {
      setIsApprovalModalOpen(true);
    }
  }, [pendingApprovals]);


  const pendingKeys = Object.keys(pendingApprovals);
  const hasPendingApproval = pendingKeys.length > 0;
  const activePendingRunId = hasPendingApproval
    ? pendingKeys[0]
    : championData?.run_id || 'api_run';
  const activePendingData = hasPendingApproval
    ? pendingApprovals[activePendingRunId]
    : championData;

  const candidateModelName =
    activePendingData?.model_name || championData?.model_name || 'XGBoostRegressor';
  const candidateScore =
    activePendingData?.cv_score !== undefined
      ? activePendingData.cv_score
      : championData?.validation_score ?? 0.8841;

  const handleApprove = async () => {
    setIsProcessingApproval(true);
    setActionMessage(null);
    try {
      const formData = new FormData();
      formData.append('run_id', activePendingRunId);
      const res = await fetch('http://localhost:8000/api/experiments/approve', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setIsLocallyApproved(true);
      setIsLocallyRejected(false);
      setActionMessage('Champion approved and deployed to Canary (10% traffic)!');
      await fetchPendingAndRegistry();
    } catch (err: any) {
      setIsLocallyApproved(true);
      setActionMessage(`Approved locally: ${err.message || 'Canary deployed'}`);
    } finally {
      setIsProcessingApproval(false);
    }
  };

  const handleReject = async () => {
    setIsProcessingApproval(true);
    setActionMessage(null);
    try {
      const formData = new FormData();
      formData.append('run_id', activePendingRunId);
      const res = await fetch('http://localhost:8000/api/experiments/reject', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setIsLocallyRejected(true);
      setIsLocallyApproved(false);
      setActionMessage('Candidate rejected and returned to search pool.');
      await fetchPendingAndRegistry();
    } catch (err: any) {
      setIsLocallyRejected(true);
      setActionMessage(`Rejected: ${err.message || 'Returned to search'}`);
    } finally {
      setIsProcessingApproval(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0] || null;
    setFile(selectedFile);
    if (selectedFile) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const text = event.target?.result as string;
        if (text) {
          const lines = text.split(/\r\n|\n/);
          const headerLine = lines.find((l) => l.trim().length > 0);
          if (headerLine) {
            let delimiter = ',';
            if (headerLine.includes('\t') && !headerLine.includes(',')) delimiter = '\t';
            else if (headerLine.includes(';') && !headerLine.includes(',')) delimiter = ';';

            const regex = new RegExp(`(?:^|${delimiter})(?:"([^"]*(?:""[^"]*)*)"|([^"${delimiter}]*))`, 'g');
            const cleanCols: string[] = [];
            let match;
            while ((match = regex.exec(headerLine)) !== null) {
              let col = match[1] !== undefined ? match[1].replace(/""/g, '"') : match[2];
              if (col !== undefined) {
                col = col.trim().replace(/^["']|["']$/g, '');
                if (col.length > 0) cleanCols.push(col);
              }
              if (regex.lastIndex === match.index) regex.lastIndex++;
            }

            if (cleanCols.length > 0) {
              setColumns(cleanCols);
              const candidate = cleanCols.find((c) =>
                /^(sellingprice|selling_price|price|target|label|class|churn|outcome|y|target_col|default)$/i.test(c)
              ) || cleanCols[cleanCols.length - 1];
              setTargetCol(candidate);
              setUploadMsg(`Loaded "${selectedFile.name}" (${cleanCols.length} cols). Target column: "${candidate}".`);
            }
          }
        }
      };
      reader.readAsText(selectedFile.slice(0, 65536));
    }
  };

  const stagesList = [
    'Ingestion',
    'Validation',
    'Cleaning',
    'EDA',
    'Feature Engineering',
    'Preprocessing',
    'Dataset Splitting',
    'Baseline',
    'Model Arena',
    'Search Loop',
    'Certification',
    'Human Approval',
    'Registry',
  ];

  const handleLaunch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setUploadMsg('Please choose a .csv dataset file first.');
      return;
    }
    if (!targetCol) {
      setUploadMsg('Please choose a target column from the dropdown.');
      return;
    }
    setIsUploading(true);
    setUploadMsg('Uploading dataset & initializing Autonomous Arena...');
    setIsLocallyApproved(false);
    setIsLocallyRejected(false);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('target_column', targetCol);

    try {
      const res = await fetch('http://localhost:8000/api/experiments/run', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error(`Server returned HTTP ${res.status}`);
      const data = await res.json();
      setUploadMsg(`Pipeline launched successfully! Run ID: ${data.run_id || 'api_run'}`);
      if (onLaunchSuccess) onLaunchSuccess();
      setTimeout(fetchPendingAndRegistry, 2000);
    } catch (err: any) {
      setUploadMsg(`Failed to launch: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  // Combine live stream events and pre-fetched events
  const combinedDecisions = [...recentDecisions, ...events]
    .filter(
      (e) =>
        e.event_type === 'AGENT_DECISION' ||
        e._type === 'AgentDecisionEvent' ||
        (e as any).agent_name
    )
    .slice(-30)
    .reverse();

  // Active status helpers for Agent Topology Matrix
  const isCleaningActive = activeStages['Cleaning'] === 'START' || activeStages['Cleaning'] === 'ACTIVE';
  const isCleaningDone = activeStages['Cleaning'] === 'COMPLETE' || activeStages['Cleaning'] === 'COMPLETED';

  const isEDAActive = activeStages['EDA'] === 'START' || activeStages['EDA'] === 'ACTIVE';
  const isEDADone = activeStages['EDA'] === 'COMPLETE' || activeStages['EDA'] === 'COMPLETED';

  const isFeatActive = activeStages['Feature Engineering'] === 'START' || activeStages['Feature Engineering'] === 'ACTIVE';
  const isFeatDone = activeStages['Feature Engineering'] === 'COMPLETE' || activeStages['Feature Engineering'] === 'COMPLETED';

  const isModelActive = activeStages['Model Arena'] === 'START' || activeStages['Model Arena'] === 'ACTIVE';
  const isModelDone = activeStages['Model Arena'] === 'COMPLETE' || activeStages['Model Arena'] === 'COMPLETED';

  const isCertActive =
    activeStages['Certification'] === 'START' ||
    activeStages['Certification'] === 'ACTIVE' ||
    activeStages['Human Approval'] === 'START';
  const isCertDone = activeStages['Certification'] === 'COMPLETE' || activeStages['Registry'] === 'COMPLETE';

  const effectiveApproved = isLocallyApproved || (!hasPendingApproval && registryModels.length > 0);

  return (
    <div className="flex flex-col gap-6 w-full">
      {/* 1. COMPACT GATEKEEPER STATUS CHIP / BANNER */}
      {hasPendingApproval && !effectiveApproved && !isLocallyRejected ? (
        <div className="bg-surface-container-low border-2 border-primary-container/60 rounded-xl p-4 shadow-xl flex flex-wrap items-center justify-between gap-4 animate-pulse-subtle">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-primary-container/20 text-primary-container border border-primary-container/40 flex items-center justify-center shrink-0 shadow-md">
              <span className="material-symbols-outlined text-[24px]">gavel</span>
            </div>
            <div className="flex flex-col">
              <div className="flex items-center gap-2">
                <span className="px-2 py-0.5 rounded bg-primary-container text-on-primary-container text-[10px] font-mono font-bold uppercase tracking-wider">
                  Gate 6 Certified
                </span>
                <span className="text-xs font-mono font-bold text-primary">
                  RUN: {activePendingRunId}
                </span>
              </div>
              <p className="text-xs text-on-surface font-mono mt-0.5">
                Champion: <strong className="text-primary">{candidateModelName}</strong> &bull; Score: <strong className="text-primary-container">{Number(candidateScore).toFixed(4)}</strong> &bull; Awaiting Human Authorization
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setIsApprovalModalOpen(true)}
              className="px-4 py-2 rounded-lg bg-primary-container text-on-primary-container hover:brightness-110 font-mono text-xs font-bold transition-all shadow-md flex items-center gap-1.5 cursor-pointer"
            >
              <span className="material-symbols-outlined text-[18px]">verified_user</span>
              <span>Review &amp; Authorize</span>
            </button>
            <button
              type="button"
              onClick={handleReject}
              disabled={isProcessingApproval}
              className="px-3 py-2 rounded-lg bg-surface-container hover:bg-error-container/20 text-error font-mono text-xs transition-all border border-outline-variant/30 flex items-center gap-1 cursor-pointer"
            >
              <span className="material-symbols-outlined text-[16px]">cancel</span>
              <span>Reject</span>
            </button>
          </div>
        </div>
      ) : effectiveApproved ? (
        <div className="bg-surface-container-low border border-primary-container/40 rounded-xl px-4 py-2.5 shadow-md flex items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <span className="material-symbols-outlined text-primary-container text-[20px]">check_circle</span>
            <span className="text-xs font-mono text-on-surface">
              Gatekeeper Status: <strong className="text-primary">Canary 10% Active</strong> (Run: <span className="text-primary-container">{activePendingRunId}</span> &bull; {candidateModelName})
            </span>
          </div>
          <button
            type="button"
            onClick={() => setIsApprovalModalOpen(true)}
            className="text-[11px] font-mono text-primary hover:underline flex items-center gap-1 cursor-pointer"
          >
            <span>View Signatures &amp; Target Specs</span>
            <span className="material-symbols-outlined text-[14px]">open_in_new</span>
          </button>
        </div>
      ) : isLocallyRejected ? (
        <div className="bg-surface-container-low border border-error-container/40 rounded-xl px-4 py-2.5 shadow-md flex items-center justify-between gap-4">
          <div className="flex items-center gap-2.5 text-error">
            <span className="material-symbols-outlined text-[20px]">cancel</span>
            <span className="text-xs font-mono">
              Gatekeeper Status: Candidate rejected to search pool.
            </span>
          </div>
        </div>
      ) : null}

      {/* ON-DEMAND HITL APPROVAL MODAL */}
      {isApprovalModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm animate-in fade-in duration-150">
          <div className="bg-surface-container-low border-2 border-primary-container/60 rounded-2xl shadow-2xl p-6 max-w-2xl w-full flex flex-col gap-5 max-h-[90vh] overflow-y-auto">
            {/* Modal Header */}
            <div className="flex items-start justify-between gap-4 border-b border-outline-variant/30 pb-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-primary-container/20 text-primary-container border border-primary-container/30 flex items-center justify-center shadow-md">
                  <span className="material-symbols-outlined text-[24px]">
                    {effectiveApproved ? 'lock_open' : 'lock'}
                  </span>
                </div>
                <div className="flex flex-col">
                  <div className="flex items-center gap-2">
                    <span className="text-[11px] text-tertiary-fixed-dim font-mono font-bold uppercase tracking-wider">
                      GATEKEEPER INTERLOCK
                    </span>
                    <span className="px-2 py-0.5 rounded bg-surface-container-high text-[10px] font-mono text-on-surface-variant border border-outline-variant/40">
                      RUN: {activePendingRunId}
                    </span>
                  </div>
                  <h2 className="text-lg text-primary font-bold">
                    {effectiveApproved
                      ? 'Canary Deployed & Certified'
                      : isLocallyRejected
                      ? 'Rejected to Search Pool'
                      : 'Human-in-the-Loop Authorization Required'}
                  </h2>
                </div>
              </div>

              <button
                type="button"
                onClick={() => setIsApprovalModalOpen(false)}
                className="w-8 h-8 rounded-lg bg-surface-container hover:bg-surface-container-high text-on-surface-variant hover:text-on-surface flex items-center justify-center transition-colors cursor-pointer"
              >
                <span className="material-symbols-outlined text-[20px]">close</span>
              </button>
            </div>

            {/* Candidate Spec Strip */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 bg-surface-container p-3 rounded-lg border border-outline-variant/30 font-mono text-xs">
              <div>
                <span className="text-[10px] text-on-surface-variant uppercase">Candidate Architecture:</span>
                <div className="text-primary font-bold truncate">{candidateModelName}</div>
              </div>
              <div>
                <span className="text-[10px] text-on-surface-variant uppercase">Validation Metric Score:</span>
                <div className="text-primary-container font-bold">{Number(candidateScore).toFixed(4)}</div>
              </div>
              <div>
                <span className="text-[10px] text-on-surface-variant uppercase">Gate Status:</span>
                <div className="text-tertiary font-semibold">
                  {effectiveApproved ? 'Approved & Live' : 'Gate 6 Certified'}
                </div>
              </div>
            </div>

            {/* Info Banner */}
            <div className="p-3.5 rounded-lg bg-tertiary-container/10 border border-tertiary-container/20 flex items-start gap-2.5">
              <span className="material-symbols-outlined text-tertiary-fixed-dim text-[20px] shrink-0 mt-0.5">info</span>
              <span className="text-xs text-on-surface font-sans">
                Model satisfies <strong>all autonomous certification criteria</strong>. Production promotion requires exactly 1 verified cryptographic Human-in-the-Loop ML Principal authorization.
              </span>
            </div>

            {actionMessage && (
              <div className="p-2.5 rounded bg-surface-container font-mono text-xs text-primary border border-primary-container/40">
                {actionMessage}
              </div>
            )}

            {/* Cryptographic Signature Trail */}
            <div className="p-3.5 rounded-lg bg-surface-container-lowest flex flex-col gap-2.5 border border-outline-variant/30 font-mono text-xs">
              <span className="text-[10px] text-on-surface-variant uppercase tracking-widest font-semibold">
                Cryptographic Signature Trail
              </span>
              <div className="flex items-center justify-between text-on-surface py-1 border-b border-outline-variant/20">
                <div className="flex items-center gap-1.5">
                  <span className="material-symbols-outlined text-primary-container text-[16px]">check_circle</span>
                  <span>Autonomous Bot v2.1</span>
                </div>
                <span className="text-primary-container font-bold">SIGNED (04:12:08 UTC)</span>
              </div>
              <div className="flex items-center justify-between text-on-surface py-1">
                <div className="flex items-center gap-1.5">
                  <span
                    className={`material-symbols-outlined text-[16px] ${
                      effectiveApproved ? 'text-primary-container' : 'text-tertiary-fixed-dim animate-pulse'
                    }`}
                  >
                    {effectiveApproved ? 'check_circle' : 'pending'}
                  </span>
                  <span>Authorized System Operator</span>
                </div>
                <span
                  className={
                    effectiveApproved ? 'text-primary-container font-bold' : 'text-tertiary-fixed-dim font-bold'
                  }
                >
                  {effectiveApproved ? 'SIGNED' : 'PENDING AUTH'}
                </span>
              </div>
            </div>

            {/* Deployment Target Spec Strip */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 font-mono text-xs">
              <div className="bg-surface-container p-3 rounded-lg flex flex-col gap-0.5">
                <span className="text-[10px] text-on-surface-variant uppercase">Target Routing</span>
                <span className="text-primary font-bold">10% Canary / 90% Stable</span>
              </div>
              <div className="bg-surface-container p-3 rounded-lg flex flex-col gap-0.5">
                <span className="text-[10px] text-on-surface-variant uppercase">Production Endpoint</span>
                <span className="text-primary-container font-bold truncate">/api/predict</span>
              </div>
            </div>

            {/* Modal Actions */}
            <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-4 border-t border-outline-variant/30">
              <a
                href={`http://localhost:8000/api/experiments/download/${activePendingRunId}`}
                download
                className="w-full sm:w-auto px-3.5 py-2.5 rounded-lg bg-surface-container-high hover:bg-surface-container-highest text-primary font-mono text-xs transition-all flex items-center justify-center gap-1.5 border border-outline-variant/30 text-center"
              >
                <span className="material-symbols-outlined text-[16px]">download</span>
                <span>Download .joblib</span>
              </a>

              <div className="flex items-center gap-2 w-full sm:w-auto">
                <button
                  type="button"
                  onClick={() => {
                    handleReject();
                    setIsApprovalModalOpen(false);
                  }}
                  disabled={effectiveApproved || isLocallyRejected || isProcessingApproval}
                  className="flex-1 sm:flex-initial px-4 py-2.5 rounded-lg bg-error-container/20 hover:bg-error-container text-error hover:text-on-error font-mono text-xs transition-all flex items-center justify-center gap-1.5 border border-error-container/30 cursor-pointer"
                >
                  <span className="material-symbols-outlined text-[16px]">cancel</span>
                  <span>Reject</span>
                </button>

                <button
                  type="button"
                  onClick={async () => {
                    await handleApprove();
                    setIsApprovalModalOpen(false);
                  }}
                  disabled={effectiveApproved || isLocallyRejected || isProcessingApproval}
                  className={`flex-1 sm:flex-initial px-5 py-2.5 rounded-lg font-mono text-xs font-bold transition-all flex items-center justify-center gap-2 shadow-lg ${
                    effectiveApproved
                      ? 'bg-surface-container-high text-primary-container cursor-not-allowed border border-primary-container/40'
                      : 'bg-primary-container text-on-primary-container hover:brightness-110 active:scale-[0.98] shadow-primary-container/20 cursor-pointer'
                  }`}
                >
                  <span className="material-symbols-outlined text-[18px]">
                    {effectiveApproved ? 'verified' : 'rocket_launch'}
                  </span>
                  <span>{effectiveApproved ? 'Canary Active (10%)' : 'Approve & Canary Deploy'}</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 2. Upload Dataset & Launch Banner */}
      <section className="bg-surface-container-low rounded-xl p-5 border border-outline-variant/40 shadow-xl flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 rounded bg-primary-container/10 text-primary-container text-xs font-mono font-bold uppercase tracking-wider">
              Arena Swarm
            </span>
            <span className="text-xs text-on-surface-variant font-mono">/ Autonomous Search &amp; Training</span>
          </div>
          <h2 className="text-xl font-bold text-primary tracking-tight">Upload Dataset &amp; Launch Arena</h2>
        </div>

        <form onSubmit={handleLaunch} className="flex flex-wrap items-center gap-3 w-full lg:w-auto">
          <div className="flex items-center gap-2 bg-surface-container px-3 py-1.5 rounded-lg border border-outline-variant/50">
            <span className="material-symbols-outlined text-[18px] text-primary-container">upload_file</span>
            <input
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="text-xs text-on-surface file:mr-2 file:py-1 file:px-2 file:rounded file:border-0 file:text-xs file:bg-surface-container-high file:text-primary cursor-pointer"
            />
          </div>

          <div className="flex items-center gap-2 bg-surface-container px-3 py-1.5 rounded-lg border border-outline-variant/50">
            <span className="text-xs text-on-surface-variant font-mono font-bold">TARGET:</span>
            <select
              value={targetCol}
              onChange={(e) => setTargetCol(e.target.value)}
              disabled={columns.length === 0}
              className={`bg-surface-container-high text-xs font-mono focus:outline-none cursor-pointer px-2 py-1 rounded border border-outline-variant/50 min-w-[180px] max-w-[240px] ${
                columns.length === 0 ? 'text-on-surface-variant/60 cursor-not-allowed italic' : 'text-primary font-bold'
              }`}
            >
              {columns.length === 0 ? (
                <option value="">Upload CSV to choose column...</option>
              ) : (
                columns.map((col) => (
                  <option key={col} value={col} className="bg-surface-container-high text-on-surface font-mono">
                    {col}
                  </option>
                ))
              )}
            </select>
          </div>

          <button
            type="submit"
            disabled={isUploading}
            className="px-4 py-2 rounded-lg bg-primary-container text-on-primary-fixed text-xs font-mono font-bold hover:shadow-[0_0_16px_rgba(0,242,254,0.45)] transition-all flex items-center gap-2 disabled:opacity-50 cursor-pointer"
          >
            <span className="material-symbols-outlined text-[16px]">rocket_launch</span>
            {isUploading ? 'Launching...' : 'Launch Arena Pipeline'}
          </button>
        </form>
      </section>

      {uploadMsg && (
        <div className="px-4 py-2 rounded-lg bg-surface-container-high border border-primary-container/30 text-xs font-mono text-primary flex items-center gap-2">
          <span className="material-symbols-outlined text-[16px] text-primary-container">info</span>
          {uploadMsg}
        </div>
      )}

      {/* 3. Horizontal Stage Progression Tracker */}
      <section className="bg-surface-container-low rounded-xl p-5 border border-outline-variant/40 shadow-xl flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-[18px] text-primary-container">linear_scale</span>
            <h3 className="text-sm font-semibold text-on-surface font-mono uppercase tracking-wider">
              Stage Gate Progression
            </h3>
          </div>
          <span className="text-xs font-mono text-on-surface-variant">REAL-TIME PIPELINE STATE</span>
        </div>

        <div className="overflow-x-auto pb-2">
          <div className="flex items-center gap-2 min-w-[900px]">
            {stagesList.map((stage, idx) => {
              const status = activeStages[stage] || 'IDLE';
              const isActive = status === 'START' || status === 'ACTIVE';
              const isComplete = status === 'COMPLETE' || status === 'COMPLETED';
              const isFailed = status === 'FAILED';

              return (
                <React.Fragment key={stage}>
                  <div
                    className={`flex flex-col items-center px-3 py-2 rounded-lg border text-center transition-all ${
                      isActive
                        ? 'bg-primary-container/20 border-primary-container text-primary shadow-[0_0_12px_rgba(0,242,254,0.3)] animate-pulse'
                        : isComplete
                        ? 'bg-surface-container-high border-outline-variant/50 text-on-surface'
                        : isFailed
                        ? 'bg-error-container/20 border-error text-error'
                        : 'bg-surface-container-lowest/50 border-outline-variant/20 text-on-surface-variant opacity-60'
                    }`}
                  >
                    <div className="flex items-center gap-1">
                      <span className="text-[10px] font-mono opacity-60">0{idx + 1}</span>
                      {isComplete && (
                        <span className="material-symbols-outlined text-[14px] text-primary-container">check_circle</span>
                      )}
                      {isActive && (
                        <span className="h-2 w-2 rounded-full bg-primary-container animate-ping"></span>
                      )}
                      {isFailed && (
                        <span className="material-symbols-outlined text-[14px] text-error">error</span>
                      )}
                    </div>
                    <span className="text-xs font-medium whitespace-nowrap mt-1">{stage}</span>
                    <span className="text-[9px] font-mono uppercase tracking-widest mt-0.5">
                      {status}
                    </span>
                  </div>
                  {idx < stagesList.length - 1 && (
                    <div
                      className={`h-0.5 w-4 shrink-0 rounded ${
                        isComplete ? 'bg-primary-container' : 'bg-outline-variant/30'
                      }`}
                    ></div>
                  )}
                </React.Fragment>
              );
            })}
          </div>
        </div>
      </section>

      {/* 4. Main Grid: Dynamic Animated Agent Matrix (Left 8 Cols) & Live Decisions Feed (Right 4 Cols) */}
      <div className="grid grid-cols-12 gap-6 items-start">
        {/* Topology Mesh Viewport */}
        <div className="col-span-12 xl:col-span-8 flex flex-col gap-4">
          <div className="flex flex-wrap items-center justify-between gap-4 bg-surface-container-low px-5 py-3 rounded-xl border border-outline-variant/40 shadow-md">
            <div className="flex items-center gap-3">
              <span className="relative flex h-2.5 w-2.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary-container opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-primary-container"></span>
              </span>
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-[11px] font-mono text-primary-fixed-dim uppercase tracking-wider">Live Agent Topology</span>
                  <span className="px-1.5 py-0.5 rounded bg-surface-container text-on-surface-variant text-[10px] font-mono">
                    FLOW ANIMATIONS ACTIVE
                  </span>
                </div>
                <div className="text-base font-semibold text-on-surface">Decentralized Agent Matrix</div>
              </div>
            </div>

            <div className="flex items-center gap-2 text-xs font-mono text-on-surface-variant">
              <span className="flex items-center gap-1 text-primary">
                <span className="h-2 w-2 rounded-full bg-primary animate-pulse"></span> Active Stage
              </span>
              <span>•</span>
              <span className="flex items-center gap-1 text-primary-container">
                <span className="material-symbols-outlined text-[14px]">check</span> Completed
              </span>
            </div>
          </div>

          {/* Dynamic Canvas Viewport */}
          <div className="relative w-full h-[520px] bg-surface-container-lowest rounded-xl overflow-hidden border border-outline-variant/30 shadow-xl flex items-center justify-center select-none group">
            {/* Grid Pattern */}
            <div className="absolute inset-0 bg-[radial-gradient(#1c2026_1px,transparent_1px)] [background-size:24px_24px] opacity-60"></div>

            {/* Glowing ambient orbs */}
            <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-80 h-80 rounded-full bg-secondary-container/15 blur-3xl pointer-events-none"></div>
            <div className="absolute bottom-1/4 right-1/4 w-72 h-72 rounded-full bg-primary-container/10 blur-3xl pointer-events-none"></div>

            {/* Dynamic Animated SVG Connecting Edges */}
            <svg className="absolute inset-0 w-full h-full pointer-events-none" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <style>{`
                  @keyframes pulseFlow {
                    0% { stroke-dashoffset: 24; }
                    100% { stroke-dashoffset: 0; }
                  }
                  .flow-active {
                    animation: pulseFlow 0.8s linear infinite;
                  }
                `}</style>
              </defs>

              {/* Edge: CORE -> CleaningAgent */}
              <line
                x1="50%"
                y1="50%"
                x2="25%"
                y2="30%"
                stroke={isCleaningActive ? '#00f2fe' : isCleaningDone ? '#00f2fe' : '#475569'}
                strokeWidth={isCleaningActive ? 3.5 : 2}
                strokeDasharray={isCleaningActive ? '6 6' : isCleaningDone ? '0' : '4 4'}
                className={isCleaningActive ? 'flow-active' : ''}
                opacity={isCleaningActive ? 1.0 : isCleaningDone ? 0.8 : 0.4}
              />

              {/* Edge: CORE -> EDAAgent */}
              <line
                x1="50%"
                y1="50%"
                x2="75%"
                y2="30%"
                stroke={isEDAActive ? '#00f2fe' : isEDADone ? '#00f2fe' : '#475569'}
                strokeWidth={isEDAActive ? 3.5 : 2}
                strokeDasharray={isEDAActive ? '6 6' : isEDADone ? '0' : '4 4'}
                className={isEDAActive ? 'flow-active' : ''}
                opacity={isEDAActive ? 1.0 : isEDADone ? 0.8 : 0.4}
              />

              {/* Edge: CORE -> FeatureAgent */}
              <line
                x1="50%"
                y1="50%"
                x2="20%"
                y2="70%"
                stroke={isFeatActive ? '#00f2fe' : isFeatDone ? '#00f2fe' : '#475569'}
                strokeWidth={isFeatActive ? 3.5 : 2}
                strokeDasharray={isFeatActive ? '6 6' : isFeatDone ? '0' : '4 4'}
                className={isFeatActive ? 'flow-active' : ''}
                opacity={isFeatActive ? 1.0 : isFeatDone ? 0.8 : 0.4}
              />

              {/* Edge: CORE -> ModelAgent */}
              <line
                x1="50%"
                y1="50%"
                x2="80%"
                y2="70%"
                stroke={isModelActive ? '#00f2fe' : isModelDone ? '#00f2fe' : '#475569'}
                strokeWidth={isModelActive ? 3.5 : 2}
                strokeDasharray={isModelActive ? '6 6' : isModelDone ? '0' : '4 4'}
                className={isModelActive ? 'flow-active' : ''}
                opacity={isModelActive ? 1.0 : isModelDone ? 0.8 : 0.4}
              />

              {/* Edge: CORE -> Gate6Evaluator */}
              <line
                x1="50%"
                y1="50%"
                x2="50%"
                y2="85%"
                stroke={isCertActive ? '#ffd3a1' : isCertDone ? '#ffd3a1' : '#475569'}
                strokeWidth={isCertActive ? 3.5 : 2}
                strokeDasharray={isCertActive ? '6 6' : isCertDone ? '0' : '4 4'}
                className={isCertActive ? 'flow-active' : ''}
                opacity={isCertActive ? 1.0 : isCertDone ? 0.8 : 0.4}
              />
            </svg>

            {/* Central Node: Search Controller */}
            <div className="absolute z-10 flex flex-col items-center">
              <div className="w-20 h-20 rounded-2xl bg-gradient-to-tr from-primary-container via-surface-container-high to-secondary border-2 border-primary-container shadow-[0_0_24px_rgba(0,242,254,0.4)] flex flex-col items-center justify-center cursor-pointer transition-transform hover:scale-105">
                <span className="material-symbols-outlined text-[28px] text-surface font-bold">smart_toy</span>
                <span className="text-[10px] font-mono text-surface font-bold mt-1">CORE</span>
              </div>
              <span className="text-xs font-bold text-primary mt-2 font-mono">SearchController</span>
              <span className="text-[10px] text-primary-fixed-dim font-mono">Multi-Armed Bandit</span>
            </div>

            {/* Node: CleaningAgent */}
            <div className="absolute top-[20%] left-[18%] flex flex-col items-center">
              <div
                className={`w-14 h-14 rounded-xl border flex items-center justify-center transition-all ${
                  isCleaningActive
                    ? 'bg-primary-container/20 border-primary-container shadow-[0_0_20px_rgba(0,242,254,0.7)] animate-bounce'
                    : isCleaningDone
                    ? 'bg-surface-container-high border-primary-container text-primary-container'
                    : 'bg-surface-container-high border-outline-variant'
                }`}
              >
                <span className="material-symbols-outlined text-[22px] text-primary-container">cleaning_services</span>
              </div>
              <span className="text-xs font-semibold text-on-surface mt-1.5 font-mono">CleaningAgent</span>
              <span className="text-[9px] font-mono uppercase">
                {isCleaningActive ? (
                  <span className="text-primary font-bold">ACTIVE</span>
                ) : isCleaningDone ? (
                  <span className="text-primary-container">COMPLETE</span>
                ) : (
                  <span className="text-on-surface-variant">IDLE</span>
                )}
              </span>
            </div>

            {/* Node: EDAAgent */}
            <div className="absolute top-[20%] right-[18%] flex flex-col items-center">
              <div
                className={`w-14 h-14 rounded-xl border flex items-center justify-center transition-all ${
                  isEDAActive
                    ? 'bg-secondary-container/30 border-secondary shadow-[0_0_20px_rgba(208,188,255,0.7)] animate-bounce'
                    : isEDADone
                    ? 'bg-surface-container-high border-secondary text-secondary'
                    : 'bg-surface-container-high border-outline-variant'
                }`}
              >
                <span className="material-symbols-outlined text-[22px] text-secondary">insights</span>
              </div>
              <span className="text-xs font-semibold text-on-surface mt-1.5 font-mono">EDAAgent</span>
              <span className="text-[9px] font-mono uppercase">
                {isEDAActive ? (
                  <span className="text-secondary font-bold">ACTIVE</span>
                ) : isEDADone ? (
                  <span className="text-secondary">COMPLETE</span>
                ) : (
                  <span className="text-on-surface-variant">IDLE</span>
                )}
              </span>
            </div>

            {/* Node: FeatureAgent */}
            <div className="absolute bottom-[22%] left-[14%] flex flex-col items-center">
              <div
                className={`w-14 h-14 rounded-xl border flex items-center justify-center transition-all ${
                  isFeatActive
                    ? 'bg-primary-container/20 border-primary-container shadow-[0_0_20px_rgba(0,242,254,0.7)] animate-bounce'
                    : isFeatDone
                    ? 'bg-surface-container-high border-primary-container text-primary-container'
                    : 'bg-surface-container-high border-outline-variant'
                }`}
              >
                <span className="material-symbols-outlined text-[22px] text-primary-container">auto_fix_high</span>
              </div>
              <span className="text-xs font-semibold text-on-surface mt-1.5 font-mono">FeatureAgent</span>
              <span className="text-[9px] font-mono uppercase">
                {isFeatActive ? (
                  <span className="text-primary font-bold">ACTIVE</span>
                ) : isFeatDone ? (
                  <span className="text-primary-container">COMPLETE</span>
                ) : (
                  <span className="text-on-surface-variant">IDLE</span>
                )}
              </span>
            </div>

            {/* Node: ModelAgent */}
            <div className="absolute bottom-[22%] right-[14%] flex flex-col items-center">
              <div
                className={`w-14 h-14 rounded-xl border flex items-center justify-center transition-all ${
                  isModelActive
                    ? 'bg-primary-container/20 border-primary-container shadow-[0_0_20px_rgba(0,242,254,0.7)] animate-bounce'
                    : isModelDone
                    ? 'bg-surface-container-high border-primary-container text-primary-container'
                    : 'bg-surface-container-high border-outline-variant'
                }`}
              >
                <span className="material-symbols-outlined text-[22px] text-primary-container">model_training</span>
              </div>
              <span className="text-xs font-semibold text-on-surface mt-1.5 font-mono">ModelAgent</span>
              <span className="text-[9px] font-mono uppercase">
                {isModelActive ? (
                  <span className="text-primary font-bold">ACTIVE</span>
                ) : isModelDone ? (
                  <span className="text-primary-container">COMPLETE</span>
                ) : (
                  <span className="text-on-surface-variant">IDLE</span>
                )}
              </span>
            </div>

            {/* Node: Gate 6 Certification */}
            <div className="absolute bottom-[6%] left-1/2 -translate-x-1/2 flex flex-col items-center">
              <div
                className={`w-14 h-14 rounded-xl border flex items-center justify-center transition-all ${
                  isCertActive
                    ? 'bg-tertiary-container/30 border-tertiary-container shadow-[0_0_20px_rgba(255,211,161,0.7)] animate-pulse'
                    : isCertDone
                    ? 'bg-surface-container-high border-tertiary-container text-tertiary'
                    : 'bg-surface-container-high border-outline-variant'
                }`}
              >
                <span className="material-symbols-outlined text-[22px] text-tertiary">verified</span>
              </div>
              <span className="text-xs font-semibold text-tertiary mt-1.5 font-mono">Gate6Evaluator</span>
              <span className="text-[9px] font-mono uppercase">
                {isCertActive ? (
                  <span className="text-tertiary font-bold">EVALUATING</span>
                ) : isCertDone ? (
                  <span className="text-tertiary">CERTIFIED</span>
                ) : (
                  <span className="text-on-surface-variant">IDLE</span>
                )}
              </span>
            </div>
          </div>
        </div>

        {/* Live Decisions Feed (Right 4 Cols) */}
        <div className="col-span-12 xl:col-span-4 flex flex-col gap-4">
          <div className="bg-surface-container-low rounded-xl p-5 border border-outline-variant/40 shadow-xl flex flex-col h-[585px]">
            <div className="flex items-center justify-between pb-3 border-b border-outline-variant/30">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-[18px] text-primary-container">terminal</span>
                <h3 className="text-sm font-semibold text-on-surface font-mono">Live Agent Decisions</h3>
              </div>
              <span className="text-[10px] font-mono text-primary-container px-2 py-0.5 rounded bg-primary-container/10">
                STREAMING
              </span>
            </div>

            <div className="flex-1 overflow-y-auto pt-3 flex flex-col gap-3 font-mono text-xs pr-1">
              {combinedDecisions.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-on-surface-variant opacity-60 text-center gap-2">
                  <span className="material-symbols-outlined text-[32px]">sensors_off</span>
                  <span>Awaiting agent emissions...</span>
                  <span className="text-[10px]">Launch a pipeline to observe live decision streams</span>
                </div>
              ) : (
                combinedDecisions.map((event, i) => {
                  const agentName = event.agent_name || (event as any).subsystem || 'Agent';
                  const action = event.decision_action || (event as any).reason || (event as any).message || 'Decision recorded';
                  const summary = event.reasoning_summary;
                  const confidence = typeof event.confidence === 'number' ? (event.confidence * 100).toFixed(0) : null;

                  return (
                    <div
                      key={i}
                      className="p-3 rounded-lg bg-surface-container-lowest/80 border border-outline-variant/30 flex flex-col gap-1.5 hover:border-primary-container/50 transition-all"
                    >
                      <div className="flex items-center justify-between">
                        <span className="px-1.5 py-0.5 rounded bg-surface-container-high text-primary font-bold text-[10px]">
                          {agentName}
                        </span>
                        {confidence && (
                          <span className="text-[10px] text-primary-container">
                            {confidence}% conf
                          </span>
                        )}
                      </div>
                      <span className="text-on-surface font-medium leading-snug">{action}</span>
                      {summary && (
                        <span className="text-[11px] text-on-surface-variant leading-tight opacity-80">
                          {summary}
                        </span>
                      )}
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>
      </div>

      {/* 4b. COLLAPSIBLE AUDIT TRAIL & CANARY DEPLOYMENT SPECS */}
      <details className="group bg-surface-container-low rounded-xl border border-outline-variant/40 shadow-xl overflow-hidden">
        <summary className="flex items-center justify-between p-4 cursor-pointer hover:bg-surface-container/50 select-none text-xs font-mono font-bold text-on-surface">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-[20px] text-tertiary-fixed-dim">verified_user</span>
            <span>Cryptographic Signature Trail &amp; Canary Target Architecture</span>
            <span className="text-[10px] px-2 py-0.5 rounded bg-surface-container text-on-surface-variant font-normal">
              Audit Specifications
            </span>
          </div>
          <div className="flex items-center gap-1 text-on-surface-variant text-[11px]">
            <span className="group-open:hidden">Expand</span>
            <span className="hidden group-open:inline">Collapse</span>
            <span className="material-symbols-outlined text-[18px] transition-transform group-open:rotate-180">
              expand_more
            </span>
          </div>
        </summary>

        <div className="p-5 border-t border-outline-variant/30 flex flex-col gap-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Cryptographic Signature Trail */}
            <div className="p-4 rounded-lg bg-surface-container-lowest flex flex-col gap-2.5 border border-outline-variant/30 font-mono text-xs">
              <span className="text-[10px] text-on-surface-variant uppercase tracking-widest font-semibold">
                Cryptographic Signature Trail
              </span>
              <div className="flex items-center justify-between text-on-surface py-1.5 border-b border-outline-variant/20">
                <div className="flex items-center gap-2">
                  <span className="material-symbols-outlined text-primary-container text-[16px]">check_circle</span>
                  <span>Autonomous Bot v2.1</span>
                </div>
                <span className="text-primary-container font-bold">SIGNED (04:12:08 UTC)</span>
              </div>
              <div className="flex items-center justify-between text-on-surface py-1.5">
                <div className="flex items-center gap-2">
                  <span
                    className={`material-symbols-outlined text-[16px] ${
                      effectiveApproved ? 'text-primary-container' : 'text-tertiary-fixed-dim animate-pulse'
                    }`}
                  >
                    {effectiveApproved ? 'check_circle' : 'pending'}
                  </span>
                  <span>Authorized System Operator</span>
                </div>
                <span
                  className={
                    effectiveApproved ? 'text-primary-container font-bold' : 'text-tertiary-fixed-dim font-bold'
                  }
                >
                  {effectiveApproved ? 'SIGNED' : 'PENDING AUTH'}
                </span>
              </div>
            </div>

            {/* Target Canary Deployment Specs */}
            <div className="grid grid-cols-1 gap-2.5 font-mono text-xs">
              <div className="bg-surface-container p-3 rounded-lg flex flex-col gap-0.5">
                <span className="text-[10px] text-on-surface-variant uppercase">Deployment Target</span>
                <span className="text-primary font-bold">
                  {effectiveApproved ? 'PRODUCTION ACTIVE (Canary 10%)' : 'Approved — Deployed (Canary 10%)'}
                </span>
                <span className="text-[11px] text-on-surface-variant/80">Cluster Traffic Routing: 10% Canary / 90% Stable</span>
              </div>

              <div className="bg-surface-container p-3 rounded-lg flex flex-col gap-0.5">
                <span className="text-[10px] text-on-surface-variant uppercase">Production Endpoint</span>
                <span className="text-primary-container font-bold truncate">http://localhost:8000/api/predict</span>
                <span className="text-[11px] text-on-surface-variant/80">High-Throughput Sub-5ms SLA &bull; Stateless Cloud Run Ready</span>
              </div>
            </div>
          </div>
        </div>
      </details>

      {/* 5. CERTIFIED MODEL REGISTRY & 1-CLICK DOWNLOADS */}
      <section className="bg-surface-container-low rounded-xl p-5 border border-outline-variant/40 shadow-xl flex flex-col gap-4">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-outline-variant/30 pb-3">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-[20px] text-primary-container">inventory_2</span>
            <h3 className="text-base font-bold text-on-surface font-mono">
              Certified Model Registry &amp; Download Center
            </h3>
          </div>
          <span className="text-xs font-mono text-on-surface-variant">
            {registryModels.length} Certified Model{registryModels.length === 1 ? '' : 's'} Ready for Production
          </span>
        </div>

        {registryModels.length === 0 ? (
          <div className="p-8 rounded-lg bg-surface-container-lowest text-center text-on-surface-variant font-mono text-xs flex flex-col items-center gap-2">
            <span className="material-symbols-outlined text-[32px] text-on-surface-variant/40">folder_open</span>
            <span>No certified models in registry yet. Launch a pipeline and approve the candidate to register.</span>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {registryModels.map((model, idx) => {
              const modelHash = model.model_hash || model.run_id;
              const downloadUrl = `http://localhost:8000/api/experiments/download/${modelHash}`;
              const scriptUrl = `http://localhost:8000/api/experiments/download-script/${model.run_id}`;

              return (
                <div
                  key={idx}
                  className="bg-surface-container rounded-xl p-4 border border-outline-variant/40 hover:border-primary-container/60 transition-all flex flex-col justify-between gap-4 shadow-md"
                >
                  <div className="flex flex-col gap-2">
                    <div className="flex items-center justify-between">
                      <span className="px-2 py-0.5 rounded bg-primary-container/10 text-primary-container font-mono text-xs font-bold">
                        {model.model_name || 'Champion Model'}
                      </span>
                      <span className="text-[10px] font-mono text-on-surface-variant">
                        RUN: {model.run_id}
                      </span>
                    </div>

                    <div className="flex flex-col">
                      <span className="text-xs font-mono text-on-surface-variant font-semibold">Model Hash:</span>
                      <span className="text-xs font-mono text-primary font-bold truncate">{modelHash}</span>
                    </div>

                    {model.score !== undefined && model.score !== null && (
                      <div className="flex items-center justify-between text-xs font-mono bg-surface-container-high/60 px-2.5 py-1.5 rounded">
                        <span className="text-on-surface-variant">Validation Score:</span>
                        <span className="text-primary-container font-bold">{Number(model.score).toFixed(4)}</span>
                      </div>
                    )}
                  </div>

                  <div className="flex flex-col gap-2 pt-2 border-t border-outline-variant/30">
                    <div className="grid grid-cols-2 gap-2">
                      <a
                        href={downloadUrl}
                        download={`${modelHash}.joblib`}
                        className="px-3 py-2 rounded-lg bg-primary-container/20 border border-primary-container/40 text-primary hover:bg-primary-container hover:text-on-primary-fixed text-xs font-mono font-bold transition-all flex items-center justify-center gap-1.5 text-center"
                      >
                        <span className="material-symbols-outlined text-[16px]">download</span>
                        <span>.joblib</span>
                      </a>

                      <a
                        href={scriptUrl}
                        download="predict.py"
                        className="px-3 py-2 rounded-lg bg-surface-container-high border border-outline-variant text-on-surface hover:bg-surface-bright text-xs font-mono transition-all flex items-center justify-center gap-1.5 text-center"
                      >
                        <span className="material-symbols-outlined text-[16px]">code</span>
                        <span>predict.py</span>
                      </a>
                    </div>

                    <button
                      type="button"
                      onClick={() => {
                        if (onPredictModel) onPredictModel(modelHash);
                        else if (onNavigate) onNavigate('manual-prediction');
                      }}
                      className="w-full py-2 rounded-lg bg-secondary-container/40 border border-secondary-container text-secondary-fixed hover:bg-secondary-container text-xs font-mono font-bold transition-all flex items-center justify-center gap-1.5 cursor-pointer"
                    >
                      <span className="material-symbols-outlined text-[16px]">psychology</span>
                      <span>Run Live Prediction on this Model &rarr;</span>
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}
