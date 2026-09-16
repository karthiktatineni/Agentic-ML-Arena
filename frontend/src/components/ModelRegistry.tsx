'use client';

import React, { useEffect, useState } from 'react';
import { API_BASE_URL } from '../config';

export default function ModelRegistry() {
  const [registry, setRegistry] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchRegistry = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/experiments/registry`);
      if (res.ok) {
        const data = await res.json();
        setRegistry(data.models || []);
      }
    } catch (e) {
      console.error('Failed to fetch registry', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRegistry();
    // Poll every 5 seconds for updates
    const interval = setInterval(fetchRegistry, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="glass-panel" style={{ marginTop: '20px' }}>
      <h2>Model Registry</h2>
      {loading && registry.length === 0 ? (
        <div style={{ color: 'var(--text-secondary)' }}>Loading registry...</div>
      ) : registry.length === 0 ? (
        <div style={{ color: 'var(--text-secondary)' }}>No models registered yet.</div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {registry.map((entry, idx) => (
            <div key={idx} style={{ 
              background: 'rgba(255,255,255,0.05)', 
              padding: '12px', 
              borderRadius: '8px',
              borderLeft: '4px solid var(--accent-green)'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span style={{ fontWeight: 'bold' }}>{entry.champion?.model_name || 'Ensemble'}</span>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  {new Date(entry.timestamp).toLocaleString()}
                </span>
              </div>
              <div style={{ fontSize: '0.9rem' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Run ID:</span> {entry.run_id}
              </div>
              <div style={{ fontSize: '0.9rem' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Score:</span>{' '}
                <span style={{ color: 'var(--accent-green)' }}>
                  {entry.champion?.metrics?.score ? entry.champion.metrics.score.toFixed(4) : 'N/A'}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
