'use client';

import React, { useState, useEffect } from 'react';
import { API_BASE_URL } from '../config';

export default function DatasetInputForm() {
  const [file, setFile] = useState<File | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [target, setTarget] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!file) {
      setColumns([]);
      setTarget('');
      return;
    }

    // Read the first chunk of the file to extract headers
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result as string;
      if (text) {
        // Split by newline and get the first line
        const firstLine = text.split(/\r?\n/)[0];
        // Split by comma to get headers (handle basic CSVs)
        if (firstLine) {
          const headers = firstLine.split(',').map(h => h.trim().replace(/^["']|["']$/g, ''));
          setColumns(headers);
          if (headers.length > 0) {
            setTarget(headers[headers.length - 1]); // Default to last column
          }
        }
      }
    };
    // Only need to read a small chunk to get the first line
    reader.readAsText(file.slice(0, 4096));
  }, [file]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError(true);
      setMessage("Please select a file.");
      return;
    }

    setLoading(true);
    setMessage('');
    setError(false);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('target_column', target);

      const res = await fetch(`${API_BASE_URL}/api/v1/experiments/run`, {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Failed to start pipeline');
      }
      setMessage(data.message);
    } catch (err: any) {
      setError(true);
      setMessage(err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="glass-panel" style={{ marginBottom: '24px' }}>
      <h2>New Experiment</h2>
      <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '16px' }}>
        Point the Arena to a local CSV dataset to launch a real search loop.
      </p>
      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        <div>
          <label style={{ display: 'block', marginBottom: '4px', fontSize: '0.85rem' }}>Upload Dataset (CSV)</label>
          <input
            type="file"
            accept=".csv"
            required
            onChange={(e) => setFile(e.target.files ? e.target.files[0] : null)}
            style={{
              width: '100%',
              padding: '10px',
              borderRadius: '8px',
              border: '1px solid var(--border-color)',
              background: 'rgba(0,0,0,0.2)',
              color: 'white'
            }}
          />
        </div>
        
        {columns.length > 0 && (
          <div>
            <label style={{ display: 'block', marginBottom: '4px', fontSize: '0.85rem' }}>Select Target Column</label>
            <select
              required
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              style={{
                width: '100%',
                padding: '10px',
                borderRadius: '8px',
                border: '1px solid var(--border-color)',
                background: 'rgba(0,0,0,0.2)',
                color: 'white'
              }}
            >
              {columns.map((col, idx) => (
                <option key={idx} value={col} style={{ color: 'black' }}>
                  {col}
                </option>
              ))}
            </select>
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          style={{
            marginTop: '8px',
            padding: '12px',
            borderRadius: '8px',
            border: 'none',
            background: loading ? 'var(--text-secondary)' : 'var(--accent-blue)',
            color: 'white',
            fontWeight: 'bold',
            cursor: loading ? 'not-allowed' : 'pointer',
            transition: 'background 0.2s'
          }}
        >
          {loading ? 'Launching...' : 'Launch Pipeline'}
        </button>
      </form>
      
      {message && (
        <div style={{
          marginTop: '12px',
          padding: '10px',
          borderRadius: '6px',
          backgroundColor: error ? 'rgba(239, 68, 68, 0.1)' : 'rgba(16, 185, 129, 0.1)',
          color: error ? 'var(--accent-red)' : 'var(--accent-green)',
          fontSize: '0.85rem'
        }}>
          {message}
        </div>
      )}
    </div>
  );
}
