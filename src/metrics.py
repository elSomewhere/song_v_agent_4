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
        
    def collect_from_logs(self) -> Metrics:
        """Aggregate metrics from workflow logs."""
        # Process logs
        for log in self.state.logs:
            # Count model usage
            if log.get("model"):
                self.model_usage[log["model"]] += 1
            
            # Track latencies by stage
            if log.get("latency_ms"):
                self.stage_latencies[log["stage"]].append(log["latency_ms"])
            
            # Count errors
            if log.get("status") == "error":
                self.error_counts[log["stage"]] += 1
        
        # Calculate aggregates
        end_time = datetime.now()
        elapsed = (end_time - self.state.start_time).total_seconds()
        
        # Count outcomes
        frames_accepted = len(self.state.accepted_frames)
        frames_rejected = sum(1 for log in self.state.logs 
                             if log.get("stage") == "policy" 
                             and log.get("status") == "give_up")
        retry_attempts = sum(1 for log in self.state.logs 
                            if log.get("stage") == "policy" 
                            and log.get("status") == "retry_new")
        edit_attempts = sum(1 for log in self.state.logs 
                           if log.get("stage") == "policy" 
                           and log.get("status") == "retry_edit")
        
        # Count variations
        variations_created = sum(1 for log in self.state.logs 
                               if log.get("stage") == "variation_mgr"
                               and log.get("status") == "success")
        
        # Calculate accept rate
        total_decisions = frames_accepted + frames_rejected
        accept_rate = frames_accepted / max(1, total_decisions)
        
        # Collect errors
        errors = []
        for stage, count in self.error_counts.items():
            errors.append(f"{stage}: {count} errors")
        
        # Build metrics object
        metrics = Metrics(
            run_id=self.state.trace_id,
            start_time=self.state.start_time,
            end_time=end_time,
            elapsed_s=elapsed,
            total_tokens=self.state.total_tokens,
            total_cost_usd=self.state.total_cost,
            scenes_processed=self.state.current_scene_idx,
            shots_generated=len(self.state.accepted_frames),
            variations_created=variations_created,
            frames_accepted=frames_accepted,
            frames_rejected=frames_rejected,
            retry_attempts=retry_attempts,
            edit_attempts=edit_attempts,
            accept_rate=accept_rate,
            models_used=dict(self.model_usage),
            errors=errors
        )
        
        return metrics
    
    def save_metrics(self, output_dir: Optional[Path] = None) -> Path:
        """Save metrics to JSON file."""
        if output_dir is None:
            output_dir = Path(self.state.output_dir)
        
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
            f.write(f"- Budget: ${self.state.budget_usd}\n")
            f.write(f"- Spent: ${metrics.total_cost_usd:.2f}\n")
            f.write(f"- Utilization: {(metrics.total_cost_usd / self.state.budget_usd * 100):.1f}%\n")


def create_metrics_collector(state: WorkflowState) -> MetricsCollector:
    """Factory function to create metrics collector."""
    return MetricsCollector(state) 