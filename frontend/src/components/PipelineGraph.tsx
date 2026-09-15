'use client';

import React from 'react';

const STAGES = [
  { id: 'Ingestion', label: 'Data Ingestion & PII Scan' },
  { id: 'Validation', label: 'Schema Validation' },
  { id: 'Leakage Detection', label: 'Leakage Detection' },
  { id: 'Cleaning', label: 'AI Data Cleaning' },
  { id: 'EDA', label: 'AI Exploratory Analysis' },
  { id: 'Feature Engineering', label: 'AI Feature Engineering' },
  { id: 'Preprocessing', label: 'Preprocessing' },
  { id: 'Dataset Splitting', label: 'Dataset Splitting' },
  { id: 'Baseline', label: 'Baseline Construction' },
  { id: 'Model Arena', label: 'Model Arena' },
  { id: 'Search Loop', label: 'HPO Search Loop' },
  { id: 'Certification', label: 'Certification (90% Gate)' },
  { id: 'Final Test', label: 'Final Holdout Test' },
  { id: 'Human Approval', label: 'Human Approval' },
  { id: 'Registry', label: 'Model Registry' },
];

interface PipelineGraphProps {
  activeStages: Record<string, string>; // stage_name -> "START" | "COMPLETE" | "FAILED"
}

export default function PipelineGraph({ activeStages }: PipelineGraphProps) {
  return (
    <div className="glass-panel" style={{ overflowY: 'auto', maxHeight: 'calc(100vh - 120px)' }}>
      <h2>Pipeline Execution</h2>
      <div className="pipeline-graph">
        {STAGES.map((stage, idx) => {
          const status = activeStages[stage.id];
          
          let stateClass = 'idle';
          if (status === 'START') stateClass = 'active';
          else if (status === 'COMPLETE') stateClass = 'complete';
          else if (status === 'FAILED') stateClass = 'failed';
          else {
             // If a later stage is started/complete, this one should be complete implicitly
             const activeKeys = Object.keys(activeStages);
             const activeIdxs = activeKeys.map(k => STAGES.findIndex(s => s.id === k)).filter(i => i !== -1);
             const maxActiveIdx = activeIdxs.length > 0 ? Math.max(...activeIdxs) : -1;
             
             if (idx < maxActiveIdx) {
               stateClass = 'complete';
             }
          }

          return (
            <div key={stage.id} className={`node ${stateClass}`}>
              <div className="node-content">
                <div className="node-title">{stage.label}</div>
                {stateClass === 'active' && (
                  <div className="node-sublabel">Processing...</div>
                )}
                {stateClass === 'failed' && (
                  <div className="node-sublabel" style={{ color: 'var(--accent-red)' }}>Failed / Rejected</div>
                )}
              </div>
              <div className="status-icon">
                {stateClass === 'complete' && (
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12"></polyline>
                  </svg>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
