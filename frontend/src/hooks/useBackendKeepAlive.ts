'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { API_BASE_URL } from '../config';

interface KeepAliveState {
  isAwake: boolean;
  isPinging: boolean;
  lastPingAt: Date | null;
  latencyMs: number | null;
  consecutiveFailures: number;
}

/**
 * useBackendKeepAlive
 * 
 * Periodically sends a lightweight heartbeat ping to the backend (e.g., /api/health)
 * to prevent cloud platforms like Render / Railway free tiers from spinning down after
 * inactivity (15-min idle timeout).
 * 
 * Interval: 3.5 minutes (210,000 ms), safely under the 14-minute sleep window.
 */
export function useBackendKeepAlive(intervalMs: number = 210_000) {
  const [state, setState] = useState<KeepAliveState>({
    isAwake: true,
    isPinging: false,
    lastPingAt: null,
    latencyMs: null,
    consecutiveFailures: 0,
  });

  const lastPingTimestampRef = useRef<number>(0);

  const pingBackend = useCallback(async () => {
    setState((prev) => ({ ...prev, isPinging: true }));
    const startTime = performance.now();

    try {
      // Primary ping endpoint /api/health with /health fallback
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 12000); // 12s timeout for wakeups

      let res = await fetch(`${API_BASE_URL}/api/health`, {
        method: 'GET',
        signal: controller.signal,
        cache: 'no-store',
      }).catch(() => null);

      if (!res || !res.ok) {
        // Fallback to /health or /api/ping
        res = await fetch(`${API_BASE_URL}/health`, {
          method: 'GET',
          cache: 'no-store',
        }).catch(() => null);
      }

      clearTimeout(timeoutId);

      const endTime = performance.now();
      const latency = Math.round(endTime - startTime);
      lastPingTimestampRef.current = Date.now();

      if (res && res.ok) {
        setState({
          isAwake: true,
          isPinging: false,
          lastPingAt: new Date(),
          latencyMs: latency,
          consecutiveFailures: 0,
        });
      } else {
        setState((prev) => ({
          ...prev,
          isAwake: false,
          isPinging: false,
          consecutiveFailures: prev.consecutiveFailures + 1,
        }));
      }
    } catch {
      setState((prev) => ({
        ...prev,
        isAwake: false,
        isPinging: false,
        consecutiveFailures: prev.consecutiveFailures + 1,
      }));
    }
  }, []);

  // Initial ping and recurring timer
  useEffect(() => {
    // Immediate ping on boot
    pingBackend();

    const intervalTimer = setInterval(() => {
      pingBackend();
    }, intervalMs);

    // Also ping when user returns to tab if more than 2 minutes have elapsed
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        const elapsedSinceLastPing = Date.now() - lastPingTimestampRef.current;
        if (elapsedSinceLastPing > 120_000) {
          pingBackend();
        }
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      clearInterval(intervalTimer);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [pingBackend, intervalMs]);

  return {
    ...state,
    pingBackend,
  };
}
