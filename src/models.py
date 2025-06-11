"""Pydantic models for the VC-RAG-SBG system."""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import uuid4


class Entity(BaseModel):
    """Entity in a scene (character, prop, etc)."""
    name: str
    pose: Optional[str] = None
    emotion: Optional[str] = None
    description: Optional[str] = None


class Camera(BaseModel):
    """Camera configuration for a shot."""
    type: str  # "static", "tracking", "pan", etc.
    angle: str  # "low", "high", "eye-level", etc.
    distance: str  # "close-up", "medium", "wide", "full"
    movement: Optional[str] = None


class ScenePlan(BaseModel):
    """Plan for a single shot/frame."""
    scene_id: int
    shot_id: int
    entities: List[Entity]
    camera: Camera
    image_prompt: str
    style_notes: Optional[str] = None
    
    
class ReviewedPlan(BaseModel):
    """Plan after review with context."""
    approved_plan: ScenePlan
    visual_context: List[str]  # Reference image IDs/paths
    negative_prompt: str
    estimated_tokens: int


class RefMeta(BaseModel):
    """Metadata for a reference image."""
    frame_id: str = Field(default_factory=lambda: str(uuid4()))
    category: Literal["character", "environment", "props", "other"]
    entity: str
    tags: List[str]
    confidence: float
    clip_embedding: List[float]  # 1536-D from text-embedding-3-large
    thumb_path: str
    source: Literal["user_upload", "generated"]
    original_path: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class QAResult(BaseModel):
    """Result from quality assessment."""
    status: Literal["pass", "retry", "fail"]
    quality_score: float
    specific_issues: List[str] = []
    retry_guidance: Optional[str] = None


class SceneData(BaseModel):
    """Parsed scene from script."""
    scene_id: int
    raw_text: str
    description: Optional[str] = None
    location: Optional[str] = None
    time: Optional[str] = None
    entities: List[str] = []


class WorkflowState(BaseModel):
    """State for the LangGraph workflow."""
    # Input data
    script_path: str
    style_path: str
    entities_path: str
    refs_dir: Optional[str] = None
    output_dir: str
    
    # Parsed data
    scenes: List[SceneData] = []
    style_text: str = ""
    entities_dict: Dict[str, Any] = {}
    ref_index: List[RefMeta] = []
    
    # Current processing state
    current_scene_idx: int = 0
    current_shot_idx: int = 0
    current_variation_idx: int = 0
    current_plan: Optional[ScenePlan] = None
    reviewed_plan: Optional[ReviewedPlan] = None
    variations: List[ScenePlan] = []
    
    # QA and policy
    fast_qa_result: Optional[QAResult] = None
    vision_qa_result: Optional[QAResult] = None
    fast_qa_flag: bool = False
    policy_action: Optional[Literal["accept", "retry_new", "retry_edit", "give_up"]] = None
    retry_count: int = 0
    edit_retry_count: int = 0
    
    # Generated outputs
    current_image_path: Optional[str] = None
    current_image_b64: Optional[str] = None
    accepted_frames: List[Dict[str, Any]] = []
    
    # Metrics and logging
    total_tokens: int = 0
    total_cost: float = 0.0
    start_time: datetime = Field(default_factory=datetime.now)
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    logs: List[Dict[str, Any]] = []
    
    # Configuration
    config: Dict[str, Any] = {}
    budget_usd: float = 35.0
    n_variations: int = 3
    max_retries: int = 2
    max_edit_retries: int = 1
    
    # Memory context
    episodic_memory: List[Dict[str, Any]] = []
    visual_memory: List[Dict[str, Any]] = []


class LogEntry(BaseModel):
    """Entry for the logs.jsonl file."""
    ts: datetime = Field(default_factory=datetime.now)
    stage: str
    trace_id: str
    model: Optional[str] = None
    tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    latency_ms: Optional[int] = None
    status: str
    error: Optional[str] = None
    extra: Dict[str, Any] = {}


class Metrics(BaseModel):
    """Run-level metrics."""
    run_id: str
    start_time: datetime
    end_time: datetime
    elapsed_s: float
    total_tokens: int
    total_cost_usd: float
    scenes_processed: int
    shots_generated: int
    variations_created: int
    frames_accepted: int
    frames_rejected: int
    retry_attempts: int
    edit_attempts: int
    accept_rate: float
    models_used: Dict[str, int]  # model name -> call count
    errors: List[str] = []


class ImageGenerationRequest(BaseModel):
    """Request for image generation."""
    prompt: str
    negative_prompt: Optional[str] = None
    reference_images: List[str] = []  # Base64 or file IDs
    size: str = "1024x1024"
    quality: str = "standard"
    n: int = 1
    response_format: str = "b64_json"
    

class ImageEditRequest(BaseModel):
    """Request for image editing."""
    image: str  # Base64 of image to edit
    instruction: str
    mask: Optional[str] = None  # Base64 of mask
    size: str = "1024x1024"
    n: int = 1
    response_format: str = "b64_json" 