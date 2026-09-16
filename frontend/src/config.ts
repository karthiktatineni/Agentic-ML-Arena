/**
 * Centralized API & WebSocket Configuration.
 * 
 * Supports local development and production deployments (Render, Vercel, etc.).
 * To point the frontend to a remote backend (e.g. on Render), set:
 *   NEXT_PUBLIC_API_URL=https://your-backend-service.onrender.com
 * in frontend/.env.local or your deployment platform environment variables.
 */

const rawApiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const API_BASE_URL = rawApiUrl.replace(/\/$/, '');

export const WS_BASE_URL = (
  process.env.NEXT_PUBLIC_WS_URL || 
  API_BASE_URL.replace(/^https:\/\//, 'wss://').replace(/^http:\/\//, 'ws://')
).replace(/\/$/, '');
