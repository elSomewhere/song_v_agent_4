# VC-RAG-SBG: Visual-Context-Aware Retrieval-Augmented-Generation Storyboard Generator

A sophisticated storyboard generation system that uses GPT-4o and gpt-image-1 to create consistent, high-quality storyboard frames from scripts. The system employs visual context awareness, memory-based retrieval, and quality control to maintain consistency across generated frames.

## Features

- **Script-to-Storyboard**: Automatically parses scripts and generates storyboard frames
- **Visual Context Awareness**: Uses LanceDB vector storage to maintain visual consistency
- **Reference Image Support**: Incorporates user-provided reference images for characters and environments
- **Multi-Variation Generation**: Creates multiple camera angle variations for each shot
- **Quality Control**: Two-tier quality assessment (Fast QA + Vision QA) with retry logic
- **Budget Management**: Tracks costs and respects budget limits
- **Comprehensive Logging**: Detailed JSONL logs and metrics for every run

## Requirements

- Python 3.8+
- OpenAI API key with access to:
  - GPT-4o
  - gpt-image-1 (for image generation)
  - text-embedding-3-large
  - GPT-4o vision

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd vc-rag-sbg
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your OpenAI API key:
```bash
OPENAI_API_KEY=your-api-key-here
```

## Usage

### Basic Usage

Run with default example data:
```bash
python run.py
```

### Custom Input Files

```bash
python run.py --script path/to/script.md \
              --style path/to/style.md \
              --entities path/to/entities.md \
              --refs path/to/refs/directory
```

### Command Line Options

- `--script`: Path to script markdown file (default: data/script.md)
- `--style`: Path to style guide markdown file (default: data/style.md)
- `--entities`: Path to entities/characters markdown file (default: data/entities.md)
- `--refs`: Path to reference images directory (default: data/refs)
- `--budget`: Budget limit in USD (overrides config)
- `--variations`: Number of camera variations per shot (overrides config)
- `--config`: Path to config file (default: config.yaml)
- `--no-refs`: Skip reference image processing

### Input File Formats

#### script.md
A markdown file containing the script with scene markers:
```markdown
# Scene 1: Opening
The camera pans across a desolate wasteland...

# Scene 2: First Encounter
Helena stands atop the fortress wall...
```

#### style.md
A markdown file describing the visual style:
```markdown
# Visual Style Guide
- Dark, post-apocalyptic aesthetic
- Muted color palette with occasional vibrant accents
- Cinematic framing with dramatic lighting
```

#### entities.md
A markdown file with character/entity descriptions, optionally including JSON:
```markdown
# Characters and Entities

```json
{
  "Helena": {
    "description": "Young woman, late 20s, battle-worn armor",
    "features": "Dark hair, determined expression, scar on left cheek"
  },
  "Joy": {
    "description": "Small floating robot companion",
    "features": "Spherical body, glowing blue eyes, cheerful demeanor"
  }
}
```
```

#### Reference Images
Place reference images in the refs directory. Supported formats: PNG, JPG, JPEG, WebP

## Output Structure

Each run creates a timestamped output directory:
```
output/
└── run_20240115_143022/
    ├── frames/           # Generated storyboard frames
    ├── variations/       # Alternative camera angles
    ├── memory/          # LanceDB vector storage
    ├── logs.jsonl       # Detailed execution logs
    ├── metrics.json     # Run statistics
    ├── report.md        # Human-readable summary
    └── state.json       # Complete workflow state
```

## Configuration

Edit `config.yaml` to customize:
- Context window sizes
- Number of variations
- Retry limits
- Budget constraints
- Model selections
- Quality thresholds

## Architecture

The system uses a LangGraph workflow with the following nodes:

1. **Script Preprocessing**: Parses script into scenes
2. **Reference Processing**: Tags and indexes reference images
3. **Planning**: Creates shot plans using GPT-4o
4. **Review**: Ensures visual consistency
5. **Variation Manager**: Generates camera angle variations
6. **Renderer**: Creates images using gpt-image-1
7. **Quality Assessment**: Fast QA + sampled Vision QA
8. **Policy Engine**: Decides accept/retry/give-up
9. **Memory Update**: Updates vector database

## Cost Management

- Tracks token usage and image generation costs
- Respects budget limits with automatic stopping
- Provides detailed cost breakdowns in metrics

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure your OpenAI API key is properly set in `.env`
2. **Budget Exceeded**: Increase budget in config or via `--budget` flag
3. **Memory Error**: Ensure sufficient RAM for LanceDB operations
4. **Image Generation Failures**: Check API quotas and rate limits

### Debug Mode

Enable verbose logging by checking `logs.jsonl` for detailed error messages.

## License

[Your License Here]

## Contributing

[Contributing Guidelines] # song_v_agent_4
