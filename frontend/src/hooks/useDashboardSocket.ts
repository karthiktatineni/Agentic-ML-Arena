import { useEffect, useState, useRef } from 'react';

export type EventType = 
  | 'PipelineStageUpdatedEvent' 
  | 'AgentDecisionEvent' 
  | 'ChampionCertifiedEvent' 
  | 'CertificationRejectedEvent' 
  | string;


export interface DashboardEvent {
  _type: EventType;
  timestamp: string;
  [key: string]: any;
}

export function useDashboardSocket(url: string) {
  const [isConnected, setIsConnected] = useState(false);
  const [events, setEvents] = useState<DashboardEvent[]>([]);
  const [activeStages, setActiveStages] = useState<Record<string, string>>({}); // stage_name -> status
  const [championData, setChampionData] = useState<any>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let ws: WebSocket;
    let reconnectTimer: NodeJS.Timeout;

    const connect = () => {
      ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        console.log('WebSocket connected to', url);
      };

      ws.onmessage = (message) => {
        try {
          const data = JSON.parse(message.data) as DashboardEvent;
          setEvents((prev) => [...prev, data].slice(-200)); // Keep last 200 events

          if (data._type === 'PipelineStageUpdatedEvent' && data.stage_name) {
            setActiveStages((prev) => ({
              ...prev,
              [data.stage_name]: data.status,
            }));
          } else if (data._type === 'ChampionCertifiedEvent' || data.event_type === 'CHAMPION_CERTIFIED') {
            setChampionData(data);
          }
        } catch (e) {
          console.error('Failed to parse WebSocket message', e);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        // Exponential backoff or simple retry
        reconnectTimer = setTimeout(() => {
          connect();
        }, 3000);
      };

      ws.onerror = (err) => {
        console.error('WebSocket error', err);
        ws.close();
      };
    };

    connect();

    return () => {
      clearTimeout(reconnectTimer);
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [url]);

  return { isConnected, events, activeStages, championData };
}
