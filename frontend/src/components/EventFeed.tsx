'use client';

import React, { useEffect, useRef } from 'react';
import { DashboardEvent } from '../hooks/useDashboardSocket';

interface EventFeedProps {
  events: DashboardEvent[];
}

export default function EventFeed({ events }: EventFeedProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new event
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [events]);

  return (
    <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <h2>Live Agent Feed</h2>
      <div className="event-feed" ref={scrollRef} style={{ flex: 1 }}>
        {events.map((ev, idx) => (
          <div key={idx} className={`event-item ${ev._type}`}>
            <div className="event-meta">
              <span>{ev._type}</span>
              <span>{new Date(ev.timestamp).toLocaleTimeString()}</span>
            </div>
            {ev._type === 'PipelineStageUpdatedEvent' && (
              <div>
                <strong>{ev.stage_name}</strong>: {ev.status}
              </div>
            )}
            {ev._type === 'AgentDecisionEvent' && (
              <div>
                <strong>{ev.agent_name}</strong> [Trial {ev.run_id}] - <em>{ev.decision_action}</em>
                <br />
                {ev.reasoning_summary && <span style={{ color: 'var(--text-secondary)' }}>{ev.reasoning_summary}</span>}
              </div>
            )}
            {ev._type !== 'PipelineStageUpdatedEvent' && ev._type !== 'AgentDecisionEvent' && (
              <pre style={{ fontSize: '0.8rem', whiteSpace: 'pre-wrap', margin: '4px 0 0' }}>
                {JSON.stringify(ev, null, 2)}
              </pre>
            )}
          </div>
        ))}
        {events.length === 0 && (
          <div style={{ color: 'var(--text-secondary)', fontStyle: 'italic', padding: '20px' }}>
            Waiting for events...
          </div>
        )}
      </div>
    </div>
  );
}
