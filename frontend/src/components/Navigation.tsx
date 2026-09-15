'use client';

import React from 'react';

export type NavTab = 
  | 'mission-control' 
  | 'model-arena-lineage' 
  | 'champion-certification' 
  | 'manual-prediction' 
  | 'resource-cost-monitor';

interface NavigationProps {
  activeTab: NavTab;
  setActiveTab: (tab: NavTab) => void;
  isConnected: boolean;
  activeRunId?: string;
  bestScore?: number;
  modelsExploredCount?: number;
  onPause?: () => void;
  onAbort?: () => void;
}

export default function Navigation({
  activeTab,
  setActiveTab,
  isConnected,
  activeRunId = 'Alpha-Run-2026',
  bestScore = 0.9412,
  modelsExploredCount = 8,
  onPause,
  onAbort,
}: NavigationProps) {
  const navItems: { id: NavTab; label: string; icon: string; pulse?: boolean }[] = [
    { id: 'mission-control', label: 'Arena', icon: 'radar', pulse: true },
    { id: 'model-arena-lineage', label: 'Model Arena & Lineage', icon: 'account_tree' },
    { id: 'champion-certification', label: 'Champion Certification', icon: 'verified' },
    { id: 'manual-prediction', label: 'Manual Prediction', icon: 'psychology' },
    { id: 'resource-cost-monitor', label: 'Resource & Cost Monitor', icon: 'query_stats' },
  ];

  return (
    <>
      {/* Top Header */}
      <header className="fixed top-0 left-0 w-full z-50 bg-surface-container-lowest/90 backdrop-blur-xl border-b border-outline-variant/40">
        <div className="h-16 w-full px-6 flex items-center justify-between gap-4">
          {/* Logo & Version */}
          <div className="flex items-center gap-3 shrink-0">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-primary-container to-secondary flex items-center justify-center shadow-md">
              <span className="material-symbols-outlined text-[20px] text-surface font-bold">token</span>
            </div>
            <div className="flex flex-col">
              <div className="flex items-center gap-1.5">
                <span className="text-headline-md text-primary tracking-tight font-bold text-lg">AutoML Arena</span>
                <span className="px-1.5 py-0.5 rounded bg-surface-container-high text-primary-fixed-dim text-xs font-mono border border-outline-variant/50">
                  SYSTEM ACTIVE
                </span>
              </div>
            </div>
            {/* Live Run Tag */}
            <div className="hidden 2xl:flex items-center gap-2 px-3 py-1 rounded bg-surface-container-low border border-outline-variant/40">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary-container opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-primary-container"></span>
              </span>
              <span className="text-xs text-on-surface-variant font-mono font-medium">RUN: {activeRunId}</span>
              <span className="text-xs text-primary-container font-semibold font-mono">
                ({isConnected ? 'LIVE STREAM' : 'DISCONNECTED'})
              </span>
            </div>
          </div>

          {/* Quick Metrics Strip */}
          <div className="hidden xl:flex items-center gap-4 px-4 py-1.5 rounded-lg bg-surface-container-low/70 border border-outline-variant/30 shrink font-mono text-xs">
            <div className="flex items-center gap-1.5">
              <span className={`h-2 w-2 rounded-full ${isConnected ? 'bg-primary-container animate-pulse' : 'bg-error'}`}></span>
              <span className="text-on-surface-variant uppercase tracking-wider">Swarm:</span>
              <span className="text-primary font-bold">{isConnected ? 'ONLINE' : 'OFFLINE'}</span>
            </div>
            <div className="w-px h-3.5 bg-outline-variant/40"></div>
            <div className="flex items-center gap-1.5">
              <span className="text-on-surface-variant uppercase tracking-wider">Host:</span>
              <span className="text-on-surface font-semibold">16.0 GB RAM</span>
            </div>
            <div className="w-px h-3.5 bg-outline-variant/40"></div>
            <div className="flex items-center gap-1.5">
              <span className="text-on-surface-variant uppercase tracking-wider">Champion CV:</span>
              <span className="text-primary font-bold">
                {bestScore !== undefined && bestScore !== null ? Number(bestScore).toFixed(4) : 'Ready'}
              </span>
            </div>
            <div className="w-px h-3.5 bg-outline-variant/40"></div>
            <div className="flex items-center gap-1.5">
              <span className="text-on-surface-variant uppercase tracking-wider">Gate:</span>
              <span className="text-tertiary-fixed-dim font-semibold">Gate 6 HITL</span>
            </div>
          </div>

          {/* Actions & Profile */}
          <div className="flex items-center gap-3 shrink-0">
            <div className="flex items-center gap-2">
              <button
                onClick={onPause}
                className="px-3 py-1.5 rounded border border-tertiary-container/40 bg-surface-container-high/40 text-tertiary text-xs font-mono hover:bg-surface-container-high transition-all"
                type="button"
              >
                Pause Experiment
              </button>
              <button
                onClick={onAbort}
                className="px-3 py-1.5 rounded border border-error-container/60 bg-error-container/10 text-error text-xs font-mono hover:bg-error-container hover:text-on-error transition-all"
                type="button"
              >
                Emergency Abort
              </button>
            </div>
            <div className="flex items-center gap-2 pl-2 border-l border-outline-variant/40">
              <div className="flex flex-col text-right hidden sm:flex">
                <span className="text-xs text-on-surface font-semibold leading-tight font-mono">System Operator</span>
                <span className="text-[10px] text-primary-container font-mono leading-tight">Autonomous Agent Lab</span>
              </div>
              <div className="w-8 h-8 rounded-full bg-surface-container-high border border-primary-container/40 flex items-center justify-center font-bold text-primary font-mono text-xs">
                <span className="material-symbols-outlined text-[16px]">terminal</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Operations Deck Sidebar */}
      <aside className="fixed left-0 top-16 h-[calc(100vh-4rem)] w-64 bg-surface-container-lowest border-r border-outline-variant/40 z-40 flex flex-col justify-between p-3">
        <div className="flex flex-col gap-1">
          <div className="px-3 py-2">
            <span className="text-[11px] uppercase tracking-widest text-on-surface-variant/80 font-bold font-mono">
              Operations Deck
            </span>
          </div>
          <nav className="flex flex-col gap-1">
            {navItems.map((item) => {
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`group flex items-center justify-between px-3 py-2 rounded text-sm transition-all text-left ${
                    isActive
                      ? 'bg-surface-container-high text-primary border-l-2 border-primary-container font-semibold shadow-sm'
                      : 'text-on-surface-variant hover:bg-surface-container-high hover:text-on-surface'
                  }`}
                  type="button"
                >
                  <div className="flex items-center gap-2.5">
                    <span className="material-symbols-outlined text-[18px]">{item.icon}</span>
                    <span className="font-medium">{item.label}</span>
                  </div>
                  {item.pulse && isActive && (
                    <span className="flex h-2 w-2 rounded-full bg-primary-container animate-pulse"></span>
                  )}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Core Telemetry Widget */}
        <div className="p-3 rounded bg-surface-container-low border border-outline-variant/30 flex flex-col gap-1.5 font-mono text-xs">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-on-surface-variant uppercase">Local Host Runtime</span>
            <span className={`text-[10px] font-bold ${isConnected ? 'text-primary-container' : 'text-error'}`}>
              {isConnected ? 'LIVE WS' : 'OFFLINE'}
            </span>
          </div>
          <div className="flex justify-between items-center text-on-surface">
            <span className="text-on-surface-variant text-[11px]">System RAM</span>
            <span className="font-semibold text-[11px]">14.8 / 16.0 GB (Host)</span>
          </div>
          <div className="w-full bg-surface-container-high h-1.5 rounded overflow-hidden">
            <div className="bg-primary-container h-full w-[88%]"></div>
          </div>
          <div className="flex justify-between items-center text-on-surface">
            <span className="text-on-surface-variant text-[11px]">Worker Engine</span>
            <span className="text-primary font-semibold text-[11px]">FastAPI Async</span>
          </div>
        </div>
      </aside>
    </>
  );
}
