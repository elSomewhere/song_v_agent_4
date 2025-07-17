"""Prompt building utilities for both OpenAI and Midjourney rendering."""

from typing import List, Any
from src.models import WorkflowState


def build_raw_prompt(state: WorkflowState, variation: Any) -> str:
    """Build the complete image generation prompt.
    
    This is the canonical prompt builder used by both OpenAI image generation
    and Midjourney prompt conversion.
    """
    # ------------------------------------------------------------------
    # Standardised multi-section prompt for the image model
    # ------------------------------------------------------------------

    sections: List[str] = []

    # 1) Global narrative / design context
    if state.static_summary:
        sections.append("<CONTEXT>\n" + state.static_summary.strip() + "\n</CONTEXT>")

    # 2) Art-style rules (short excerpt + any shot-specific notes)
    style_block_parts: List[str] = []
    if state.style_text:
        style_block_parts.append(state.style_text[:400].strip())
    if variation.style_notes:
        style_block_parts.append(variation.style_notes.strip())
    if style_block_parts:
        sections.append("<STYLE_GUIDE>\n" + "\n".join(style_block_parts) + "\n</STYLE_GUIDE>")

    # 3) The actual shot description (what to draw)
    sections.append("<SHOT_PROMPT>\n" + variation.image_prompt.strip() + "\n</SHOT_PROMPT>")

    # 4) Camera metadata
    cam = variation.camera
    cam_desc = f"type={cam.type}; angle={cam.angle}; distance={cam.distance}"
    if cam.movement:
        cam_desc += f"; movement={cam.movement}"
    sections.append("<CAMERA>\n" + cam_desc + "\n</CAMERA>")

    # 5) Negative prompt / avoid list
    if state.reviewed_plan and state.reviewed_plan.negative_prompt:
        sections.append("<NEGATIVE>\n" + state.reviewed_plan.negative_prompt.strip() + "\n</NEGATIVE>")

    # Final prompt string (double line breaks between blocks for clarity)
    return "\n\n".join(sections) 