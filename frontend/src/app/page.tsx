'use client';

import React, { useState } from 'react';
import Navigation, { NavTab } from '../components/Navigation';
import { useDashboardSocket } from '../hooks/useDashboardSocket';
import { useBackendKeepAlive } from '../hooks/useBackendKeepAlive';

import { API_BASE_URL, WS_BASE_URL } from '../config';

// View Components
import MissionControlView from '../components/views/MissionControlView';
import ModelRegistryView from '../components/views/ModelRegistryView';
import ModelArenaLineageView from '../components/views/ModelArenaLineageView';
import ChampionCertificationView from '../components/views/ChampionCertificationView';
import ManualPredictionView from '../components/views/ManualPredictionView';
import ResourceCostMonitorView from '../components/views/ResourceCostMonitorView';

export default function DashboardPage() {
  const WS_URL = `${WS_BASE_URL}/api/v1/ws/dashboard`;
  const { isConnected, events, activeStages, stageMessages, latestMessage, championData } = useDashboardSocket(WS_URL);
  // Keep-alive heartbeat pings backend every 3.5 mins to prevent cloud spin-down
  const keepAlive = useBackendKeepAlive(210_000);
  const [activeTab, setActiveTab] = useState<NavTab>('mission-control');
  const [targetModelForPrediction, setTargetModelForPrediction] = useState<string | null>(null);
  const [isPaused, setIsPaused] = useState(false);
  const [toast, setToast] = useState<{ type: 'info' | 'error' | 'success'; text: string } | null>(null);

  // Compute live metrics from real stream or active run
  const activeRunId = championData?.run_id || 'api_run';
  const bestCv = championData?.validation_score ?? null;

  const handlePredictModel = (modelId: string) => {
    setTargetModelForPrediction(modelId);
    setActiveTab('manual-prediction');
  };

  const handlePause = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/experiments/${activeRunId}/pause`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setIsPaused(data.is_paused);
        setToast({
          type: 'info',
          text: data.is_paused ? 'Experiment paused. Training loop will hold after current step.' : 'Experiment resumed. Continuing training loop.',
        });
      } else {
        // Fallback to unparameterized endpoint
        const resFallback = await fetch(`${API_BASE_URL}/api/experiments/pause`, { method: 'POST' });
        const data = await resFallback.json();
        setIsPaused(data.is_paused);
        setToast({
          type: 'info',
          text: data.is_paused ? 'Experiment paused.' : 'Experiment resumed.',
        });
      }
    } catch (err: any) {
      setToast({ type: 'error', text: `Failed to pause/resume: ${err.message}` });
    } finally {
      setTimeout(() => setToast(null), 4000);
    }
  };

  const handleAbort = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/experiments/${activeRunId}/abort`, { method: 'POST' });
      if (res.ok) {
        setIsPaused(false);
        setToast({ type: 'error', text: 'Emergency Abort executed. Pipeline terminated.' });
      } else {
        await fetch(`${API_BASE_URL}/api/experiments/abort`, { method: 'POST' });
        setIsPaused(false);
        setToast({ type: 'error', text: 'Emergency Abort executed.' });
      }
    } catch (err: any) {
      setToast({ type: 'error', text: `Failed to abort: ${err.message}` });
    } finally {
      setTimeout(() => setToast(null), 4000);
    }
  };

  return (
    <div className="bg-surface font-sans text-on-surface min-h-screen selection:bg-primary-container selection:text-on-primary-container">
      {/* Toast Notification */}
      {toast && (
        <div
          className={`fixed top-20 right-8 z-50 p-4 rounded-xl shadow-2xl flex items-center gap-3 border text-sm font-mono backdrop-blur-md transition-all ${
            toast.type === 'error'
              ? 'bg-error-container/90 text-on-error-container border-error-container'
              : 'bg-surface-container-high/95 text-primary border-primary-container/50'
          }`}
        >
          <span className="material-symbols-outlined text-[20px]">
            {toast.type === 'error' ? 'cancel' : 'info'}
          </span>
          <span>{toast.text}</span>
          <button onClick={() => setToast(null)} className="ml-2 hover:opacity-70 text-xs">
            ✕
          </button>
        </div>
      )}

      {/* Cloud Backend Waking Up Alert (Render Free Tier Cold Start) */}
      {!keepAlive.isAwake && keepAlive.consecutiveFailures > 0 && (
        <div className="fixed top-16 left-0 w-full z-40 bg-amber-950/90 text-amber-200 border-b border-amber-500/40 px-6 py-2 text-xs font-mono flex items-center justify-between backdrop-blur-md">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-[16px] animate-spin text-amber-400">sync</span>
            <span>Cloud backend is waking up from sleep (Render cold start)... Keep-alive heartbeat pinging now.</span>
          </div>
          <button
            onClick={() => keepAlive.pingBackend()}
            className="px-2.5 py-0.5 rounded bg-amber-700/80 hover:bg-amber-600 text-white font-bold cursor-pointer transition-colors text-[11px]"
          >
            Ping Now
          </button>
        </div>
      )}

      {/* Fixed Header & Operations Deck Sidebar */}
      <Navigation
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isConnected={isConnected}
        activeRunId={activeRunId}
        bestScore={bestCv}
        isPaused={isPaused}
        onPause={handlePause}
        onAbort={handleAbort}
        isBackendAwake={keepAlive.isAwake}
        isBackendPinging={keepAlive.isPinging}
        latencyMs={keepAlive.latencyMs}
      />

      {/* Main Content Viewport */}
      <div className="pl-64 w-full">
        <main className="w-full pt-20 px-8 pb-12 bg-surface min-h-screen">
          {activeTab === 'mission-control' && (
            <MissionControlView
              events={events}
              activeStages={activeStages}
              stageMessages={stageMessages}
              latestMessage={latestMessage}
              championData={championData}
              onNavigate={(tab) => setActiveTab(tab)}
              onPredictModel={handlePredictModel}
            />
          )}

          {activeTab === 'model-registry' && (
            <ModelRegistryView
              onPredictModel={handlePredictModel}
            />
          )}

          {activeTab === 'model-arena-lineage' && (
            <ModelArenaLineageView
              championData={championData}
              events={events}
            />
          )}

          {activeTab === 'champion-certification' && (
            <ChampionCertificationView
              championData={championData}
            />
          )}

          {activeTab === 'manual-prediction' && (
            <ManualPredictionView
              initialModelId={targetModelForPrediction}
            />
          )}

          {activeTab === 'resource-cost-monitor' && (
            <ResourceCostMonitorView />
          )}
        </main>
      </div>
    </div>
  );
}

