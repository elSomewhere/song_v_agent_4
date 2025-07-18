"""Utility functions for the VC-RAG-SBG system."""

import os
import json
import jsonlines
import base64
import time
import yaml
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


def get_image_size_from_aspect_ratio(aspect_ratio: str) -> str:
    """Map aspect ratio name to OpenAI API image size."""
    aspect_ratio_map = {
        "square": "1024x1024",
        "landscape": "1536x1024", 
        "portrait": "1024x1536",
        "auto": "auto"
    }
    return aspect_ratio_map.get(aspect_ratio, "1024x1024")


# Load pricing configuration from external file
def _load_pricing_config() -> Dict[str, Any]:
    """Load pricing configuration from external YAML file."""
    pricing_path = Path("pricing.yaml")
    if pricing_path.exists():
        try:
            with open(pricing_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Failed to load pricing.yaml: {e}")
    
    # Fallback to built-in defaults
    return {
        "token_costs": {
            "gpt-4o": {"input": 0.0025, "output": 0.01},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
            "gpt-image-1": {"input": 0.0025, "output": 0.01},
        },
        "image_costs": {
            "gpt-image-1": {
                "1024x1024": {"low": 0.02, "medium": 0.08, "high": 0.32},
                "1024x1536": {"low": 0.03, "medium": 0.12, "high": 0.48},
                "1536x1024": {"low": 0.03, "medium": 0.12, "high": 0.48}
            }
        }
    }

# Load pricing configuration at startup
_PRICING_CONFIG = _load_pricing_config()
COST_PER_1K_TOKENS = _PRICING_CONFIG.get("token_costs", {})
IMAGE_GEN_COST = _PRICING_CONFIG.get("image_costs", {})


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects and Pydantic models."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        # Handle Pydantic models
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        # Handle BaseModel instances that might not have model_dump
        if hasattr(obj, 'dict'):
            return obj.dict()
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
    pricing_config = _load_pricing_config()
    image_costs = pricing_config.get("image_costs", {})
    
    if model not in image_costs:
        return 0.0
    
    size_costs = image_costs[model].get(size, image_costs[model].get("1024x1024", {}))
    return size_costs.get(quality, size_costs.get("medium", 0.0))


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


def create_thumbnail(image_path: str, thumb_path: str, size: Tuple[int, int] = (600, 600)) -> None:
    """Create a thumbnail of an image."""
    img = Image.open(image_path)
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
        model = kwargs.get("model", "")
        
        if model.startswith("text-embedding") or model == "image-embed-1":
            # Embedding calls (both text and image embeddings)
            return client.embeddings.create(**kwargs)
        elif model == "gpt-image-1":
            # Only gpt-image-1 supported for image generation
            if "image" in kwargs:
                # Edit endpoint
                return client.images.edit(**kwargs)
            else:
                # Generation endpoint
                return client.images.generate(**kwargs)
        else:
            # Chat completion calls (text models)
            return client.chat.completions.create(**kwargs)
    except Exception as e:
        print(f"OpenAI API error: {e}")
        raise


def ensure_directory(path: str) -> None:
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_workflow_state(state: WorkflowState) -> None:
    """Save workflow state to JSON."""
    # Handle both WorkflowState objects and AddableValuesDict from LangGraph
    if hasattr(state, 'output_dir'):
        output_dir = state.output_dir
        state_data = state.model_dump() if hasattr(state, 'model_dump') else dict(state)
    else:
        output_dir = state["output_dir"]
        state_data = dict(state)
    
    state_path = Path(output_dir) / "state.json"
    with open(state_path, 'w') as f:
        json.dump(state_data, f, indent=2, cls=DateTimeEncoder)


def count_tokens_approx(text: str) -> int:
    """Approximate token count (GPT-4 tokenizer approximation)."""
    # Rough approximation: 1 token ≈ 4 characters
    return len(text) // 4


def format_timestamp() -> str:
    """Get formatted timestamp for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_json_response(response: str) -> Dict[str, Any]:
    """Parse JSON from potentially messy LLM response with improved heuristics."""
    import re
    
    # First, try direct JSON parsing in case the response is clean
    try:
        parsed = json.loads(response.strip())
        if isinstance(parsed, dict) and len(parsed) > 0:
            return parsed
    except json.JSONDecodeError:
        pass
    
    # Remove markdown code blocks and backticks
    clean_response = response
    clean_response = re.sub(r'```json\s*', '', clean_response, flags=re.IGNORECASE)
    clean_response = re.sub(r'```[^`]*```', '', clean_response, flags=re.DOTALL)
    clean_response = re.sub(r'`[^`]*`', '', clean_response)
    
    # Remove common prefixes that GPT might add (but be careful not to truncate valid JSON)
    clean_response = re.sub(r'^.*?(?=\{)', '', clean_response, flags=re.DOTALL)
    # Don't truncate at first }, instead look for the last } (complete JSON object)
    # clean_response = re.sub(r'\}.*?$', '}', clean_response, flags=re.DOTALL)  # REMOVED - was causing truncation
    
    # Try parsing the cleaned response
    try:
        parsed = json.loads(clean_response.strip())
        if isinstance(parsed, dict) and len(parsed) > 0:
            return parsed
    except json.JSONDecodeError:
        pass
    
    # Look for JSON objects with improved regex
    json_patterns = [
        r'\{.*\}',  # Greedy pattern for full JSON - try first
        r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}',  # Nested pattern
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Original pattern
        r'\{.*?\}',  # Simple non-greedy pattern - try last
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match.strip())
                if isinstance(parsed, dict) and len(parsed) > 0:
                    return parsed
            except json.JSONDecodeError:
                continue
    
    # Fallback: try line by line for multiline JSON
    lines = response.split('\n')
    json_lines = []
    in_json = False
    brace_count = 0
    
    for line in lines:
        line = line.strip()
        if '{' in line and not in_json:
            in_json = True
            json_lines = [line]
            brace_count = line.count('{') - line.count('}')
        elif in_json:
            json_lines.append(line)
            brace_count += line.count('{') - line.count('}')
            if brace_count <= 0:
                break
    
    if json_lines:
        try:
            json_str = '\n'.join(json_lines)
            parsed = json.loads(json_str)
            if isinstance(parsed, dict) and len(parsed) > 0:
                return parsed
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON array
    start_idx = response.find('[')
    end_idx = response.rfind(']') + 1
    
    if start_idx != -1 and end_idx > start_idx:
        json_str = response[start_idx:end_idx]
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, list) and len(parsed) > 0:
                return {"data": parsed}
        except json.JSONDecodeError:
            pass
    
    # Last resort: print the response for debugging
    print(f"[DEBUG] Failed to parse JSON from response: {response[:500]}...")
    
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


def get_style_embedding(image_b64: str, config: Dict[str, Any], state: Any = None) -> List[float]:
    """Generate style embedding for an image using image-embed-1.
    
    Args:
        image_b64: Base64 encoded image data
        config: Configuration dictionary
        state: WorkflowState for logging and cost tracking (optional)
        
    Returns:
        List of floats representing the style embedding vector
    """
    # Check if style embedding is enabled
    if not config.get("style_embedding_enabled", False):
        # Return zero vector if disabled
        dim = config.get("style_embedding_dimension", 1024)
        return [0.0] * dim
    
    # Create cache directory for style embeddings
    cache_dir = Path(".cache") / "style_emb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate cache key from image hash
    image_hash = hashlib.sha256(image_b64.encode()).hexdigest()
    cache_file = cache_dir / f"{image_hash}.json"
    
    # Check cache first
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                if state:
                    log_entry(state, "style_embedding", "cache_hit", 
                             extra={"image_hash": image_hash[:8]})
                return cached_data["embedding"]
        except Exception as e:
            if state:
                log_entry(state, "style_embedding", "cache_error", error=str(e))
    
    # Generate new embedding
    try:
        client = get_openai_client()
        model = config["models"].get("embedding_style", "image-embed-1")
        
        # Call image embedding API
        response = call_openai_with_retry(
            client,
            model=model,
            input=f"data:image/jpeg;base64,{image_b64}"
        )
        
        embedding = response.data[0].embedding
        cost = 0.0005  # Approximate cost per image embed
        
        # Update state if provided
        if state:
            state.total_cost += cost
            log_entry(state, "style_embedding", "success",
                     model=model, cost_usd=cost,
                     extra={"dimension": len(embedding)})
        
        # Cache the result
        cache_data = {
            "embedding": embedding,
            "model": model,
            "timestamp": datetime.now().isoformat()
        }
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            if state:
                log_entry(state, "style_embedding", "cache_write_error", error=str(e))
        
        return embedding
        
    except Exception as e:
        if state:
            log_entry(state, "style_embedding", "error", error=str(e))
        # Return zero vector on error
        dim = config.get("style_embedding_dimension", 1024)
        return [0.0] * dim 