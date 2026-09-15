from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import predict, ws, experiments, system

from contextlib import asynccontextmanager
import asyncio
from app.events.bus import event_bus

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the event bus processing loop
    task = asyncio.create_task(event_bus.process_events())
    yield
    # Cancel the task on shutdown
    task.cancel()

app = FastAPI(
    title="AutoML Arena API",
    description="API and WebSocket interfaces for AutoML Arena v2",
    version="2.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict.router, prefix="/api/v1/predict", tags=["predict"])
app.include_router(ws.router, prefix="/api/v1/ws", tags=["websocket"])
app.include_router(experiments.router, prefix="/api/v1/experiments", tags=["experiments"])
app.include_router(system.router, prefix="/api/v1/system", tags=["system"])

# Alias /api for direct access
app.include_router(predict.router, prefix="/api/predict", tags=["predict-alias"])
app.include_router(ws.router, prefix="/api/ws", tags=["websocket-alias"])
app.include_router(experiments.router, prefix="/api/experiments", tags=["experiments-alias"])
app.include_router(system.router, prefix="/api/system", tags=["system-alias"])

if __name__ == "__main__":
    import uvicorn
    # Typically run via `uvicorn app.api.main:app`
    uvicorn.run(app, host="0.0.0.0", port=8000)
