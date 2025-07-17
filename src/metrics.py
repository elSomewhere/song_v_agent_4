"""Metrics collection and aggregation for VC-RAG-SBG system."""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from collections import defaultdict

from src.models import Metrics, WorkflowState, LogEntry


class MetricsCollector:
    """Collects and aggregates metrics throughout the workflow."""
    
    def __init__(self, state: WorkflowState):
        self.state = state
        self.model_usage = defaultdict(int)
        self.stage_latencies = defaultdict(list)
        self.error_counts = defaultdict(int)
        
    def _get_state_attr(self, attr_name: str, default=None):
        """Helper method to get state attribute, handling both WorkflowState and AddableValuesDict."""
        if hasattr(self.state, attr_name):
            return getattr(self.state, attr_name, default)
        else:
            return self.state.get(attr_name, default)
    
    def collect_from_logs(self) -> Metrics:
        """Aggregate metrics from workflow logs."""
        # Process logs
        logs = self._get_state_attr('logs', [])
        for log in logs:
            # Count model usage
            if log.get("model"):
                self.model_usage[log["model"]] += 1
            
            # Track latencies by stage
            if log.get("latency_ms"):
                self.stage_latencies[log["stage"]].append(log["latency_ms"])
            
            # Count errors
            if log.get("status") == "error":
                self.error_counts[log["stage"]] += 1
        
        # Calculate derived metrics
        start_time = self._get_state_attr('start_time')
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds() if start_time else 0
        
        # Frame statistics
        accepted_frames = self._get_state_attr('accepted_frames', [])
        frames_accepted = len(accepted_frames)
        frames_rejected = sum(1 for log in logs
                            if log.get("stage") == "reviewer" and log.get("status") == "rejected")
        
        # Quality control metrics
        retry_attempts = sum(1 for log in logs
                           if log.get("stage") == "retry")
        
        edit_attempts = sum(1 for log in logs
                          if log.get("stage") == "edit")
        
        accept_rate = frames_accepted / max(1, frames_accepted + frames_rejected)
        
        # Count variations
        variations_created = sum(1 for log in logs
                               if log.get("stage") == "renderer" and log.get("status") == "success")
        
        # Build metrics object
        metrics = Metrics(
            # Basic run info
            run_id=self._get_state_attr('trace_id', ''),
            start_time=start_time or datetime.now(),
            end_time=end_time,
            elapsed_s=elapsed,
            
            # Token and cost tracking
            total_tokens=self._get_state_attr('total_tokens', 0),
            total_cost_usd=self._get_state_attr('total_cost', 0.0),
            scenes_processed=self._get_state_attr('current_scene_idx', 0),
            shots_generated=len(accepted_frames),
            variations_created=variations_created,
            frames_accepted=frames_accepted,
            frames_rejected=frames_rejected,
            accept_rate=accept_rate,
            retry_attempts=retry_attempts,
            edit_attempts=edit_attempts,
            
            # Model usage and performance
            models_used=dict(self.model_usage),
            errors=[]  # Initialize empty errors list
        )
        
        return metrics
    
    def save_metrics(self, output_dir: Optional[Path] = None) -> Path:
        """Save metrics to JSON file."""
        if output_dir is None:
            # Handle both WorkflowState objects and AddableValuesDict from LangGraph
            if hasattr(self.state, 'output_dir'):
                output_dir = Path(self.state.output_dir)
            else:
                output_dir = Path(self.state["output_dir"])
        
        metrics = self.collect_from_logs()
        metrics_path = output_dir / "metrics.json"
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics.model_dump(), f, indent=2, default=str)
        
        return metrics_path
    
    def get_stage_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary statistics by stage."""
        summary = {}
        
        for stage, latencies in self.stage_latencies.items():
            if latencies:
                summary[stage] = {
                    "calls": len(latencies),
                    "avg_latency_ms": sum(latencies) / len(latencies),
                    "min_latency_ms": min(latencies),
                    "max_latency_ms": max(latencies),
                    "errors": self.error_counts.get(stage, 0)
                }
        
        return summary
    
    def get_cost_breakdown(self) -> Dict[str, float]:
        """Get cost breakdown by model."""
        cost_by_model = defaultdict(float)
        
        for log in self.state.logs:
            if log.get("model") and log.get("cost_usd"):
                cost_by_model[log["model"]] += log["cost_usd"]
        
        return dict(cost_by_model)
    
    def get_token_breakdown(self) -> Dict[str, int]:
        """Get token usage breakdown by model."""
        tokens_by_model = defaultdict(int)
        
        for log in self.state.logs:
            if log.get("model") and log.get("tokens"):
                tokens_by_model[log["model"]] += log["tokens"]
        
        return dict(tokens_by_model)
    
    def append_to_report(self, report_path: Path) -> None:
        """Append detailed metrics to existing report."""
        metrics = self.collect_from_logs()
        stage_summary = self.get_stage_summary()
        cost_breakdown = self.get_cost_breakdown()
        token_breakdown = self.get_token_breakdown()
        
        with open(report_path, 'a') as f:
            f.write("\n\n## Detailed Metrics\n\n")
            
            # Stage performance
            f.write("### Stage Performance\n")
            for stage, stats in stage_summary.items():
                f.write(f"- **{stage}**: {stats['calls']} calls, "
                       f"avg {stats['avg_latency_ms']:.0f}ms")
                if stats['errors'] > 0:
                    f.write(f", {stats['errors']} errors")
                f.write("\n")
            
            # Cost breakdown
            f.write("\n### Cost Breakdown by Model\n")
            for model, cost in cost_breakdown.items():
                f.write(f"- {model}: ${cost:.4f}\n")
            
            # Token breakdown
            f.write("\n### Token Usage by Model\n")
            for model, tokens in token_breakdown.items():
                f.write(f"- {model}: {tokens:,} tokens\n")
            
            # Quality metrics
            f.write("\n### Quality Metrics\n")
            f.write(f"- Accept Rate: {metrics.accept_rate:.1%}\n")
            f.write(f"- Average Retries per Shot: "
                   f"{(metrics.retry_attempts + metrics.edit_attempts) / max(1, metrics.shots_generated):.2f}\n")
            
            # Budget utilization
            f.write("\n### Budget Utilization\n")
            budget_usd = self._get_state_attr('budget_usd', 35.0)
            f.write(f"- Budget: ${budget_usd}\n")
            f.write(f"- Spent: ${metrics.total_cost_usd:.2f}\n")
            f.write(f"- Utilization: {(metrics.total_cost_usd / budget_usd * 100):.1f}%\n")

    def export_text_report(self, output_dir: Path) -> Path:
        """Export a formatted text report."""
        metrics = self.collect_from_logs()
        report_path = output_dir / "run_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("# VC-RAG-SBG Run Report\n\n")
            f.write(f"**Run ID:** {metrics.run_id}\n")
            f.write(f"**Duration:** {metrics.elapsed_s:.1f} seconds\n")
            f.write(f"**Total Cost:** ${metrics.total_cost_usd:.2f}\n")
            f.write(f"**Total Tokens:** {metrics.total_tokens:,}\n\n")
            
            f.write("## Generation Stats\n")
            f.write(f"- Scenes Processed: {metrics.scenes_processed}\n")
            f.write(f"- Shots Generated: {metrics.shots_generated}\n")
            f.write(f"- Variations Created: {metrics.variations_created}\n")
            f.write(f"- Frames Accepted: {metrics.frames_accepted}\n")
            f.write(f"- Accept Rate: {metrics.accept_rate:.1%}\n\n")
            
            f.write("## Quality Control\n")
            f.write(f"- Retry Attempts: {metrics.retry_attempts}\n")
            f.write(f"- Edit Attempts: {metrics.edit_attempts}\n")
            f.write(f"- Frames Rejected: {metrics.frames_rejected}\n\n")
            
            f.write("## Model Usage\n")
            for model, count in metrics.models_used.items():
                f.write(f"- {model}: {count} calls\n")
            f.write("\n")
            
            f.write("## Output Location\n")
            output_dir_str = self._get_state_attr('output_dir', str(output_dir))
            f.write(f"{output_dir_str}\n\n")
        
        return report_path


def create_metrics_collector(state: WorkflowState) -> MetricsCollector:
    """Factory function to create metrics collector."""
    return MetricsCollector(state) 