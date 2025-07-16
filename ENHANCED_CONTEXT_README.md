# Enhanced Context System for Long Storyboards

## Problem Statement

The original VC-RAG-SBG system had limited context awareness for long storyboards:
- Small context windows (`ctx_window: 4`, `ctx_images: 6`)
- Only looked at immediate neighboring frames
- No character/environment consistency tracking across many scenes
- Fragmented memory without global narrative understanding

## Solution: Enhanced Global Context System

### 🎯 New Configuration Parameters

The enhanced system adds several new configuration options to `config.yaml`:

```yaml
# Enhanced context windows for long storyboards
ctx_images: 6                         # Reference images per shot
ctx_window: 4                         # Local context window

# NEW: Global context parameters
global_context_enabled: true          # Enable enhanced context tracking
global_ctx_window: 20                 # Look back across many scenes
global_ctx_weight: 0.3                # Weight global vs local context
character_consistency_window: 50      # Track character appearances
environment_consistency_window: 30    # Track environment continuity

# NEW: Advanced context modes
context_mode: "enhanced"              # basic | enhanced | global
consistency_tracking:
  characters: true                    # Track character consistency
  environments: true                  # Track environment continuity  
  style_elements: true                # Track visual style consistency
  narrative_flow: true                # Track narrative progression
```

### 🔧 Technical Implementation

#### 1. Enhanced Memory Service (`src/memory.py`)

**New Methods Added:**
- `get_enhanced_visual_context()` - Gets local + global context
- `_get_global_context()` - Builds comprehensive global context
- `_get_character_appearances()` - Tracks character consistency across many scenes
- `_get_environment_continuity()` - Tracks environment/setting consistency
- `_get_style_consistency()` - Monitors visual style across storyboard
- `_get_narrative_flow()` - Tracks narrative progression
- `build_global_consistency_prompt()` - Builds context-aware prompts

#### 2. Enhanced Planner Node (`src/nodes/planner.py`)

- Uses enhanced context when `context_mode: "enhanced"`
- Includes global consistency information in planning prompts
- Considers character/environment history from many previous scenes

#### 3. Enhanced Reviewer Node (`src/nodes/reviewer.py`)

- Leverages global context for consistency checking
- Reviews character appearances against long-term history
- Validates style consistency across the entire storyboard
- Includes global consistency context in review prompts

### 📊 Configuration Levels

#### Basic Context (Original System)
```yaml
ctx_images: 3
ctx_window: 4
context_mode: "basic"
global_context_enabled: false
budget_usd: 25
```
- **Use for:** Quick prototypes, short scenes (1-10 shots)
- **Pros:** Fast, low cost
- **Cons:** May have consistency issues

#### Enhanced Context (Recommended)
```yaml
ctx_images: 6
ctx_window: 8
context_mode: "enhanced"
global_context_enabled: true
global_ctx_window: 15
character_consistency_window: 30
environment_consistency_window: 20
budget_usd: 40
```
- **Use for:** Most storyboards (10-50 shots)
- **Pros:** Balanced speed and consistency
- **Cons:** Moderate token usage increase

#### Global Context (Maximum Consistency)
```yaml
ctx_images: 10
ctx_window: 12
context_mode: "enhanced"
global_context_enabled: true
global_ctx_window: 50
character_consistency_window: 100
environment_consistency_window: 75
token_cap: 6000
budget_usd: 75
```
- **Use for:** Long narratives, final production (50+ shots)
- **Pros:** Maximum consistency across entire storyboard
- **Cons:** Higher cost and processing time

### 🚀 Usage Examples

#### Quick Setup with Enhanced Context
```bash
# Use the enhanced configuration for better consistency
python run.py --script data/script.md \
              --style data/style.md \
              --entities data/entities.md \
              --refs data/refs \
              --config config_enhanced.yaml \
              --budget 50
```

#### Generate Configuration Files
```bash
# Generate example configurations
python enhanced_context_examples.py

# This creates:
# - config_basic.yaml (fast, basic consistency)
# - config_enhanced.yaml (balanced approach)
# - config_global.yaml (maximum consistency)
```

#### Run with Specific Configuration
```bash
# For long storyboards requiring maximum consistency
python run.py --config config_global.yaml

# For quick prototyping
python run.py --config config_basic.yaml
```

### 📈 Benefits of Enhanced Context

#### Character Consistency
- Tracks character appearances across 50-100 previous frames
- Maintains visual consistency for characters across long narratives
- Prevents character design drift in extended sequences

#### Environment Continuity
- Remembers environment/setting details from previous scenes
- Maintains spatial relationships and lighting consistency
- Preserves architectural and atmospheric elements

#### Style Consistency
- Monitors visual style adherence across the entire storyboard
- Tracks quality scores and style evolution
- Maintains artistic coherence throughout long sequences

#### Narrative Flow
- Understands story progression and character arcs
- Maintains emotional and narrative consistency
- Preserves story beats and dramatic tension

### 💰 Cost Considerations

| Context Level | Token Usage | Estimated Cost | Generation Time |
|---------------|-------------|----------------|-----------------|
| Basic         | ~2,000/shot | $25-35 total   | Fast            |
| Enhanced      | ~3,500/shot | $40-60 total   | Moderate        |
| Global        | ~5,500/shot | $75-100 total  | Slower          |

### 🔍 How It Works

1. **Memory Indexing**: As frames are generated, they're indexed with rich metadata
2. **Context Retrieval**: When planning new shots, the system looks back across many previous frames
3. **Consistency Analysis**: Character appearances, environments, and style elements are analyzed
4. **Smart Prompting**: Global context information is injected into planner and reviewer prompts
5. **Quality Control**: The system actively monitors and maintains consistency

### 🛠️ Technical Notes

#### Memory Tables
- `episodic_text` - Stores frame summaries and metadata
- `visual_ctx` - Stores visual information and embeddings
- `canonical_text` - Stores canonical entity descriptions

#### Context Windows
- **Local Context**: Recent frames (ctx_window parameter)
- **Global Context**: Extended history (global_ctx_window parameter)
- **Character Tracking**: Long-term character consistency (character_consistency_window)
- **Environment Tracking**: Setting continuity (environment_consistency_window)

#### Prompt Engineering
Global context is injected into prompts using structured sections:
```
<CHARACTER_CONSISTENCY>
Recent character appearances (15 frames):
- Scene 3.2: Helena, Joy
- Scene 4.1: Helena
</CHARACTER_CONSISTENCY>

<ENVIRONMENT_CONSISTENCY>
Environment continuity (8 references):
- battlefield: smoke, debris, ruins
- fortress: stone walls, torches
</ENVIRONMENT_CONSISTENCY>
```

### 📝 Troubleshooting

#### High Token Usage
- Reduce `global_ctx_window` from 50 to 20
- Disable some consistency tracking features
- Use `context_mode: "basic"` for faster processing

#### Memory Issues
- The system automatically limits context windows to prevent memory overflow
- LanceDB handles large vector stores efficiently
- Clear `.cache/` directory to reset memory

#### Inconsistent Results
- Increase `global_ctx_window` for better long-range consistency
- Enable all consistency tracking features
- Increase `character_consistency_window` and `environment_consistency_window`

### 🔮 Future Enhancements

Potential improvements for even better consistency:
- **Visual Similarity Matching**: Compare actual generated images, not just prompts
- **Character Relationship Tracking**: Understand character interactions and relationships
- **Temporal Consistency**: Track changes over time (aging, weather, lighting)
- **Style Transfer**: Automatically maintain artistic style consistency
- **Semantic Scene Understanding**: Better understanding of spatial relationships

---

## Quick Start

1. **Enable enhanced context** in your `config.yaml`:
   ```yaml
   context_mode: "enhanced"
   global_context_enabled: true
   ```

2. **Run with your data**:
   ```bash
   python run.py --config config_enhanced.yaml
   ```

3. **Monitor consistency** in the generated output and logs

The enhanced context system significantly improves consistency for long storyboards while remaining configurable for different use cases and budgets. 