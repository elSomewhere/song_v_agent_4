"""Utility functions for the VC-RAG-SBG system."""

import os
import json
import jsonlines
import base64
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from PIL import Image
import io
import hashlib
from tenacity import retry, stop_after_attempt, wait_exponential
import openai
from openai import OpenAI

from src.models import LogEntry, WorkflowState


# Token cost mapping (as of 2024)
COST_PER_1K_TOKENS = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
    "gpt-image-1": {"input": 0.0025, "output": 0.01},  # Approximate based on tokens
}

# Image generation cost approximations
IMAGE_GEN_COST = {
    "gpt-image-1": {
        "1024x1024": {"low": 0.02, "medium": 0.08, "high": 0.32},
        "1024x1536": {"low": 0.03, "medium": 0.12, "high": 0.48},
        "1536x1024": {"low": 0.03, "medium": 0.12, "high": 0.48}
    },
    "dall-e-2": {
        "1024x1024": {"standard": 0.02, "hd": 0.02}
    },
    "dall-e-3": {
        "1024x1024": {"standard": 0.04, "hd": 0.08},
        "1024x1792": {"standard": 0.08, "hd": 0.12},
        "1792x1024": {"standard": 0.08, "hd": 0.12}
    }
}


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def get_openai_client() -> OpenAI:
    """Get OpenAI client instance."""
    return OpenAI()


def log_entry(state: WorkflowState, stage: str, status: str, 
              model: Optional[str] = None, tokens: Optional[int] = None,
              cost_usd: Optional[float] = None, error: Optional[str] = None,
              extra: Dict[str, Any] = None) -> None:
    """Add log entry to state and write to logs.jsonl."""
    entry = LogEntry(
        stage=stage,
        trace_id=state.trace_id,
        model=model,
        tokens=tokens,
        cost_usd=cost_usd,
        status=status,
        error=error,
        extra=extra or {}
    )
    
    # Convert to dict and handle datetime serialization
    entry_dict = entry.model_dump()
    # Convert datetime to ISO string for JSON serialization
    if 'ts' in entry_dict and isinstance(entry_dict['ts'], datetime):
        entry_dict['ts'] = entry_dict['ts'].isoformat()
    
    state.logs.append(entry_dict)
    
    # Write to logs.jsonl
    log_path = Path(state.output_dir) / "logs.jsonl"
    with jsonlines.open(log_path, mode='a') as writer:
        writer.write(entry_dict)


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for a model call."""
    if model not in COST_PER_1K_TOKENS:
        return 0.0
    
    costs = COST_PER_1K_TOKENS[model]
    input_cost = (input_tokens / 1000) * costs["input"]
    output_cost = (output_tokens / 1000) * costs["output"]
    return input_cost + output_cost


def calculate_image_cost(model: str, size: str, quality: str) -> float:
    """Calculate cost for image generation."""
    if model not in IMAGE_GEN_COST:
        return 0.0
    
    size_costs = IMAGE_GEN_COST[model].get(size, IMAGE_GEN_COST[model]["1024x1024"])
    return size_costs.get(quality, size_costs["medium"])


def check_budget(state: WorkflowState) -> bool:
    """Check if we're within budget."""
    return state.total_cost < state.budget_usd


def load_image_as_base64(path: str) -> str:
    """Load an image file as base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def save_base64_image(b64_data: str, path: str) -> None:
    """Save a base64 image to file."""
    image_data = base64.b64decode(b64_data)
    with open(path, "wb") as f:
        f.write(image_data)


def create_thumbnail(image_path: str, thumb_path: str, size: Tuple[int, int] = (256, 256)) -> None:
    """Create a thumbnail of an image."""
    with Image.open(image_path) as img:
        img.thumbnail(size)
        img.save(thumb_path)


def get_image_hash(image_path: str) -> str:
    """Get SHA256 hash of an image file."""
    with open(image_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
def call_openai_with_retry(client: OpenAI, **kwargs) -> Any:
    """Call OpenAI API with retry logic."""
    try:
        if "model" in kwargs and kwargs["model"].startswith("gpt-image"):
            # Image generation calls
            if "image" in kwargs:
                # Edit endpoint
                return client.images.edit(**kwargs)
            else:
                # Generation endpoint
                return client.images.generate(**kwargs)
        elif "model" in kwargs and kwargs["model"].startswith("text-embedding"):
            # Embedding calls
            return client.embeddings.create(**kwargs)
        else:
            # Chat completion calls
            return client.chat.completions.create(**kwargs)
    except Exception as e:
        print(f"OpenAI API error: {e}")
        raise


def ensure_directory(path: str) -> None:
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_workflow_state(state: WorkflowState) -> None:
    """Save workflow state to JSON."""
    state_path = Path(state.output_dir) / "state.json"
    with open(state_path, 'w') as f:
        json.dump(state.model_dump(), f, indent=2, cls=DateTimeEncoder)


def count_tokens_approx(text: str) -> int:
    """Approximate token count (GPT-4 tokenizer approximation)."""
    # Rough approximation: 1 token ≈ 4 characters
    return len(text) // 4


def format_timestamp() -> str:
    """Get formatted timestamp for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_json_response(response: str) -> Dict[str, Any]:
    """Parse JSON from potentially messy LLM response."""
    # Try to find JSON in the response
    start_idx = response.find('{')
    end_idx = response.rfind('}') + 1
    
    if start_idx != -1 and end_idx > start_idx:
        json_str = response[start_idx:end_idx]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON array
    start_idx = response.find('[')
    end_idx = response.rfind(']') + 1
    
    if start_idx != -1 and end_idx > start_idx:
        json_str = response[start_idx:end_idx]
        try:
            return {"data": json.loads(json_str)}
        except json.JSONDecodeError:
            pass
    
    raise ValueError("No valid JSON found in response")


def get_context_window(items: List[Any], current_idx: int, window_size: int) -> List[Any]:
    """Get context window of items around current index."""
    start = max(0, current_idx - window_size // 2)
    end = min(len(items), current_idx + window_size // 2 + 1)
    return items[start:end]


def format_scene_prompt(scene: Dict[str, Any], style: str) -> str:
    """Format a scene into a prompt for image generation."""
    prompt_parts = []
    
    if scene.get("description"):
        prompt_parts.append(scene["description"])
    
    if scene.get("location"):
        prompt_parts.append(f"Location: {scene['location']}")
    
    if scene.get("time"):
        prompt_parts.append(f"Time: {scene['time']}")
    
    if style:
        prompt_parts.append(f"Style: {style}")
    
    return " | ".join(prompt_parts)


def merge_logs(log_files: List[str], output_file: str) -> None:
    """Merge multiple log files into one."""
    all_logs = []
    
    for log_file in log_files:
        if Path(log_file).exists():
            with jsonlines.open(log_file) as reader:
                all_logs.extend(list(reader))
    
    # Sort by timestamp
    all_logs.sort(key=lambda x: x.get('ts', ''))
    
    with jsonlines.open(output_file, mode='w') as writer:
        for log in all_logs:
            writer.write(log) 