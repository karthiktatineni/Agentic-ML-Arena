import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Set

from app.events.bus import event_bus

router = APIRouter()

from collections import deque

# Store active connections
active_connections: Set[WebSocket] = set()
recent_event_history: deque = deque(maxlen=150)

async def broadcast_event(event):
    """Callback function subscribed to the global event bus."""
    if hasattr(event, "model_dump"):
        payload = event.model_dump()
        payload["_type"] = event.__class__.__name__
    else:
        payload = {"data": str(event)}
        
    message = json.dumps(payload, default=str)
    recent_event_history.append(message)
    
    # Broadcast to all connected clients
    disconnected = set()
    for connection in list(active_connections):
        try:
            await connection.send_text(message)
        except Exception:
            disconnected.add(connection)
            
    # Cleanup dead connections
    for conn in disconnected:
        active_connections.discard(conn)

_subscribed = False
def init_ws():
    global _subscribed
    if not _subscribed:
        event_bus.subscribe(broadcast_event)
        _subscribed = True

@router.get("/events/recent")
async def get_recent_events():
    """Return buffered recent events for clients reconnecting or reloading."""
    return [json.loads(m) for m in list(recent_event_history)]

@router.websocket("/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    init_ws()
    await websocket.accept()
    active_connections.add(websocket)
    
    # Replay recent event history so newly connected clients immediately see active state
    for past_msg in list(recent_event_history):
        try:
            await websocket.send_text(past_msg)
        except Exception:
            break
            
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.discard(websocket)
