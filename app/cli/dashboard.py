"""Live CLI Dashboard using rich (PRD v2 Section 8 & 56)."""

import asyncio
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text

from app.api.schemas.experiment import ExperimentObject
from app.events.bus import event_bus
from app.events.schemas import (
    BaseEvent, PipelineStageUpdatedEvent, ExperimentResultEvent, 
    ModelEliminatedEvent, ChampionChangedEvent, AgentDecisionEvent
)

PIPELINE_STAGES = [
    "Ingestion", "Dataset Splitting", "Leakage Detection",
    "Experiment State: QUEUED", "Experiment State: RUNNING", 
    "Experiment State: EVALUATING", "Experiment State: COMPLETED",
    "Model Arena", "Champion Certification"
]

class Dashboard:
    def __init__(self):
        self.console = Console()
        self.experiments: Dict[str, ExperimentObject] = {}
        self.messages: List[str] = []
        
        self.selection_val_size: int = 0
        self.comparisons_made: int = 0
        self.corrected_alpha: float = 0.05
        
        self.stage_statuses = {s: "IDLE" for s in PIPELINE_STAGES}
        
    def handle_event(self, event: BaseEvent):
        """Update internal state based on event type."""
        
        if isinstance(event, PipelineStageUpdatedEvent):
            self.stage_statuses[event.stage_name] = event.status
            if event.message:
                self.messages.append(f"[cyan]\[Stage][/cyan] {event.stage_name}: {event.message}")
                
            if event.metadata.get("strategy") == "nested_cv":
                self.selection_val_size = event.metadata.get("total_rows", 0) # Outer folds effectively use all
            elif event.metadata.get("strategy") == "three_way":
                self.selection_val_size = event.metadata.get("selection_val_rows", 0)
                
        elif isinstance(event, AgentDecisionEvent):
            self.messages.append(f"[magenta]\[{event.agent_name}][/magenta] {event.decision_action} (conf: {event.confidence:.2f})")
            
        elif isinstance(event, ModelEliminatedEvent):
            self.messages.append(f"[red]\[Eliminated][/red] {event.experiment_hash[:8]}: {event.reason}")
            if event.experiment_hash in self.experiments:
                self.experiments[event.experiment_hash].state = "ELIMINATED"
                
        elif isinstance(event, ExperimentResultEvent):
            # Upsert experiment
            if event.experiment_hash not in self.experiments:
                from app.api.schemas.experiment import ExperimentMetrics
                self.experiments[event.experiment_hash] = ExperimentObject(
                    experiment_hash=event.experiment_hash,
                    run_id=event.run_id,
                    model_family=event.model_family,
                    state=event.state,
                    metrics=ExperimentMetrics(primary_metric="f1", mean_cv_score=event.cv_score)
                )
            else:
                self.experiments[event.experiment_hash].state = event.state
                
        elif isinstance(event, ChampionChangedEvent):
            self.messages.append(f"[bold yellow]\[Champion Provisional][/bold yellow] {event.experiment_hash[:8]} score: {event.cv_score:.4f}")
            self.comparisons_made += 1
            self.corrected_alpha = 0.05 / self.comparisons_made

        # Trim messages
        if len(self.messages) > 15:
            self.messages = self.messages[-15:]
            
    def generate_pipeline_graph(self) -> Panel:
        """Visual flow graph of the pipeline stages."""
        lines = []
        for stage in PIPELINE_STAGES:
            status = self.stage_statuses.get(stage, "IDLE")
            if status == "COMPLETED":
                icon = "[green]✓[/green]"
                style = "dim"
            elif status == "ACTIVE":
                icon = "[bold yellow]►[/bold yellow]"
                style = "bold white"
            elif status == "FAILED":
                icon = "[bold red]✗[/bold red]"
                style = "bold red"
            else:
                icon = "[dim]○[/dim]"
                style = "dim"
                
            lines.append(f"{icon} [{style}]{stage}[/{style}]")
            
        return Panel("\n".join(lines), title="Pipeline Flow Graph", border_style="blue")
        
    def generate_leaderboard(self) -> Table:
        """Generate the leaderboard table based on known experiments."""
        table = Table(title="Leaderboard (Top 10)", expand=True)
        table.add_column("Rank", style="cyan", width=6)
        table.add_column("Hash", style="dim", width=12)
        table.add_column("Model Family", style="magenta")
        table.add_column("State", style="blue")
        table.add_column("CV Score", justify="right", style="green")
        
        # Filter and sort
        exps = list(self.experiments.values())
        exps.sort(key=lambda x: getattr(x.metrics, "mean_cv_score", 0.0) if x.metrics else 0.0, reverse=True)
        
        for idx, exp in enumerate(exps[:10]):
            state_color = "green" if exp.state in ("COMPLETED", "RANKED", "CERTIFIED") else ("red" if exp.state == "ELIMINATED" else "yellow")
            cv_score = f"{exp.metrics.mean_cv_score:.4f}" if exp.metrics else "N/A"
            table.add_row(
                str(idx + 1),
                exp.experiment_hash[:8],
                exp.model_family,
                f"[{state_color}]{exp.state}[/{state_color}]",
                cv_score
            )
            
        return table
        
    def generate_activity_stream(self) -> Panel:
        """Display recent log messages."""
        text = "\n".join(self.messages)
        return Panel(Text.from_markup(text), title="Agent Activity Stream")
        
    def generate_integrity_panel(self) -> Panel:
        """Validation Integrity Panel (PRD v2 Section 56)."""
        text = (
            f"[bold]Selection-Validation Set Size:[/bold] {self.selection_val_size}\n"
            f"[bold]Global Comparisons Made (m):[/bold] {self.comparisons_made}\n"
            f"[bold]Corrected Significance (α/m):[/bold] {self.corrected_alpha:.6f}\n"
            f"\n"
            f"[dim]Tracking exact statistical comparisons prevents leaderboard overfitting.[/dim]"
        )
        return Panel(Text.from_markup(text), title="Validation Integrity", border_style="cyan")
        
    def generate_layout(self) -> Layout:
        """Create the overall dashboard layout."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main")
        )
        layout["main"].split_row(
            Layout(name="left_panel", ratio=1),
            Layout(name="leaderboard", ratio=2),
            Layout(name="sidebar", ratio=1)
        )
        
        layout["left_panel"].update(self.generate_pipeline_graph())
        
        layout["sidebar"].split_column(
            Layout(name="integrity", size=8),
            Layout(name="activity")
        )
        
        header_text = Text("AutoML Arena Live Dashboard", justify="center", style="bold white on blue")
        layout["header"].update(header_text)
        layout["leaderboard"].update(self.generate_leaderboard())
        layout["sidebar"]["integrity"].update(self.generate_integrity_panel())
        layout["sidebar"]["activity"].update(self.generate_activity_stream())
        
        return layout

    async def run_live(self):
        """Run the dashboard asynchronously consuming events."""
        # Subscribe to bus
        async def on_event(event: BaseEvent):
            self.handle_event(event)
            
        event_bus.subscribe(on_event)
        
        # Start background event processing loop if not already running
        bus_task = asyncio.create_task(event_bus.process_events())
        
        with Live(self.generate_layout(), refresh_per_second=4, screen=True) as live:
            try:
                while True:
                    live.update(self.generate_layout())
                    await asyncio.sleep(0.25)
            except asyncio.CancelledError:
                bus_task.cancel()
