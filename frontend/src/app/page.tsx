'use client';

import React, { useState } from 'react';
import Navigation, { NavTab } from '../components/Navigation';
import { useDashboardSocket } from '../hooks/useDashboardSocket';

// View Components
import MissionControlView from '../components/views/MissionControlView';
import ModelArenaLineageView from '../components/views/ModelArenaLineageView';
import ChampionCertificationView from '../components/views/ChampionCertificationView';
import ManualPredictionView from '../components/views/ManualPredictionView';
import ResourceCostMonitorView from '../components/views/ResourceCostMonitorView';

export default function DashboardPage() {
  const WS_URL = 'ws://localhost:8000/api/v1/ws/dashboard';
  const { isConnected, events, activeStages, championData } = useDashboardSocket(WS_URL);
  const [activeTab, setActiveTab] = useState<NavTab>('mission-control');
  const [targetModelForPrediction, setTargetModelForPrediction] = useState<string | null>(null);

  // Compute live metrics from real stream or active run
  const activeRunId = championData?.run_id || 'api_run';
  const bestCv = championData?.validation_score ?? null;

  const handlePredictModel = (modelId: string) => {
    setTargetModelForPrediction(modelId);
    setActiveTab('manual-prediction');
  };

  return (
    <div className="bg-surface font-sans text-on-surface min-h-screen selection:bg-primary-container selection:text-on-primary-container">
      {/* Fixed Header & Operations Deck Sidebar */}
      <Navigation
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isConnected={isConnected}
        activeRunId={activeRunId}
        bestScore={bestCv}
      />

      {/* Main Content Viewport */}
      <div className="pl-64 w-full">
        <main className="w-full pt-20 px-8 pb-12 bg-surface min-h-screen">
          {activeTab === 'mission-control' && (
            <MissionControlView
              events={events}
              activeStages={activeStages}
              championData={championData}
              onNavigate={(tab) => setActiveTab(tab)}
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
