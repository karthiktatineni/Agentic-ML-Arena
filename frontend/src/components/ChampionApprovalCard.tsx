'use client';

import React, { useState } from 'react';

interface ChampionData {
  run_id: string;
  experiment_hash: string;
  model_name: string;
  metrics: Record<string, any>;
  hyperparameters: Record<string, any>;
}

export default function ChampionApprovalCard({ champion }: { champion: ChampionData | null }) {
  const [status, setStatus] = useState<'pending' | 'approved' | 'rejected'>('pending');
  const [isSubmitting, setIsSubmitting] = useState(false);

  if (!champion) return null;

  const handleAction = async (action: 'approve' | 'reject') => {
    setIsSubmitting(true);
    try {
      const formData = new FormData();
      formData.append('run_id', champion.run_id);
      
      const res = await fetch(`http://localhost:8000/api/v1/experiments/${action}`, {
        method: 'POST',
        body: formData,
      });
      
      if (res.ok) {
        setStatus(action === 'approve' ? 'approved' : 'rejected');
      } else {
        console.error('Failed to submit approval action');
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="glass-panel" style={{ marginTop: '20px', border: '2px solid var(--accent-purple)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
        <h2 style={{ margin: 0, color: 'var(--accent-purple)' }}>🏆 Champion Certified!</h2>
      </div>
      
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '20px' }}>
        <div>
          <h3 style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Model Class</h3>
          <p style={{ fontSize: '1.2rem', fontWeight: 600 }}>{champion.model_name || 'Ensemble'}</p>
        </div>
        <div>
          <h3 style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Validation Score</h3>
          <p style={{ fontSize: '1.5rem', fontWeight: 700, color: champion.metrics?.below_threshold ? 'var(--accent-red)' : 'var(--accent-green)' }}>
            {champion.metrics?.score != null ? (champion.metrics.score * 100).toFixed(2) + '%' : 'N/A'}
          </p>
        </div>
        {champion.metrics?.mean_cv_score != null && (
          <div>
            <h3 style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Mean CV Score</h3>
            <p style={{ fontSize: '1.1rem', fontWeight: 600 }}>{(champion.metrics.mean_cv_score * 100).toFixed(2)}%</p>
          </div>
        )}
        {champion.metrics?.primary_metric && (
          <div>
            <h3 style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Primary Metric</h3>
            <p style={{ fontSize: '1.1rem', fontWeight: 600 }}>{champion.metrics.primary_metric.toUpperCase()}</p>
          </div>
        )}
      </div>
      
      {champion.metrics?.below_threshold && (
        <div style={{ background: 'rgba(231, 76, 60, 0.15)', padding: '10px 14px', borderRadius: '8px', marginBottom: '16px', border: '1px solid var(--accent-red)' }}>
          ⚠️ Score below {((champion.metrics.threshold || 0.9) * 100).toFixed(0)}% deployment threshold. Model will NOT be saved as joblib.
        </div>
      )}
      
      <div style={{ background: 'rgba(0,0,0,0.2)', padding: '12px', borderRadius: '8px', marginBottom: '20px' }}>
        <h3 style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '8px' }}>Hyperparameters</h3>
        <pre style={{ margin: 0, fontSize: '0.8rem', whiteSpace: 'pre-wrap', color: 'var(--text-primary)' }}>
          {JSON.stringify(champion.hyperparameters, null, 2)}
        </pre>
      </div>
      
      {status === 'pending' ? (
        <div style={{ display: 'flex', gap: '12px' }}>
          <button 
            className="submit-btn" 
            style={{ flex: 1, background: 'var(--accent-green)' }}
            onClick={() => handleAction('approve')}
            disabled={isSubmitting}
          >
            {isSubmitting ? 'Processing...' : 'Approve & Register'}
          </button>
          <button 
            className="submit-btn" 
            style={{ flex: 1, background: 'var(--accent-red)' }}
            onClick={() => handleAction('reject')}
            disabled={isSubmitting}
          >
            {isSubmitting ? 'Processing...' : 'Reject'}
          </button>
        </div>
      ) : (
        <div style={{ textAlign: 'center', padding: '10px', borderRadius: '8px', background: status === 'approved' ? 'rgba(46, 204, 113, 0.1)' : 'rgba(231, 76, 60, 0.1)', color: status === 'approved' ? 'var(--accent-green)' : 'var(--accent-red)' }}>
          {status === 'approved' ? '✅ Model Approved and Registered' : '❌ Model Rejected'}
        </div>
      )}
    </div>
  );
}
