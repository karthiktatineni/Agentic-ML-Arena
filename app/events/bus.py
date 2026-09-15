"""In-Memory Event Bus (PRD v2 Section 54/55)."""

import asyncio
import logging
from typing import Callable, Any, Awaitable, List
from app.events.schemas import BaseEvent

logger = logging.getLogger(__name__)

class EventBus:
    """In-memory asyncio Queue backed event bus."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EventBus, cls).__new__(cls)
            cls._instance._queue = asyncio.Queue()
            cls._instance._subscribers = []
        return cls._instance
        
    def publish(self, event: BaseEvent):
        """Publish an event to the bus."""
        self._persist_durable_event(event)
        # Non-blocking put_nowait so emitters don't block.
        try:
            self._queue.put_nowait(event)
        except Exception as e:
            logger.error(f"Failed to publish event {event.event_type}: {e}")

    def _persist_durable_event(self, event: BaseEvent):
        """Persist governance events that must outlive the in-memory bus."""
        if event.event_type not in {"CHAMPION_CERTIFIED", "CERTIFICATION_REJECTED"}:
            return

        try:
            from app.governance.certification_storage import ValidationCertificationStore

            ValidationCertificationStore.save_event(event)
        except Exception as e:
            logger.error(f"Failed to persist {event.event_type}: {e}")
            
    def subscribe(self, callback: Callable[[BaseEvent], Awaitable[None]]):
        """Subscribe an async callback to all events."""
        self._subscribers.append(callback)
        
    async def process_events(self):
        """Long-running task to process events from the queue."""
        logger.info("EventBus started processing.")
        while True:
            try:
                event = await self._queue.get()
                for sub in self._subscribers:
                    try:
                        await sub(event)
                    except Exception as e:
                        logger.error(f"Error in event subscriber handling {event.event_type}: {e}")
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"EventBus error: {e}")

# Global singleton
event_bus = EventBus()
