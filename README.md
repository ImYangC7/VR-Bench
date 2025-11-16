# VR-Bench: Visual Reasoning Benchmark for Vision-Language Models

[中文文档](README_CN.md) | English

VR-Bench is a comprehensive benchmark for evaluating Vision-Language Models (VLMs) on spatial reasoning and planning tasks through various puzzle games. It provides a unified framework for dataset generation, evaluation, and analysis.

## 🎮 Supported Games

VR-Bench includes five different puzzle games, each testing different aspects of visual reasoning:

- **Maze**: Navigate from start to goal in a grid-based maze
- **Sokoban**: Push boxes to target positions while avoiding walls
- **3D Maze**: Multi-level maze with ladders connecting different floors
- **PathFinder**: Find paths through irregular mazes with labeled waypoints
- **TrapField**: Navigate through a field while avoiding traps

## ✨ Key Features

- **Procedural Generation**: Automatically generate diverse puzzle levels with configurable difficulty
- **Texture Customization**: Support for custom visual themes through texture skins
- **Video Rendering**: Generate solution videos with smooth animations (24 FPS)
- **VLM Evaluation**: Built-in framework for testing various VLMs (GPT, Gemini, Qwen, etc.)
- **Comprehensive Metrics**: Success Rate (SR), Path Ratio (PR), Move Ratio (MR)
- **Parallel Processing**: Multi-threaded generation and evaluation for efficiency
- **Deduplication**: Automatic detection and removal of duplicate levels

## 📋 Requirements

- Python >= 3.10
- CUDA-compatible GPU (optional, for local VLM inference)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/SNHuan/VR-Bench.git
cd VR-Bench

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
# - API keys for VLM evaluation
# - Dataset paths
# - CUDA configuration
```

### 3. Download Dataset

```bash
# Download pre-generated dataset from Hugging Face
python dataset_init.py --output-dir ./dataset_VR
```

### 4. Generate Custom Levels

```bash
# Edit config/config.yaml to configure game type and difficulty
# Then run batch generation
python -m generation.batch_generate config/config.yaml
```

### 5. Evaluate VLMs

```bash
# Start local VLM server (optional, for local models)
bash scripts/start_sglang_server.sh

# Run evaluation
bash scripts/run_vlm_eval.sh
```

## 📁 Project Structure

```
VR-Bench/
├── core/                   # Core framework
│   ├── schema/            # Unified state representation
│   ├── renderer.py        # Base rendering engine
│   ├── texture_handler.py # Texture management
│   └── game_adapter.py    # Game adapter interface
├── games/                 # Game implementations
│   ├── maze/             # Maze game
│   ├── sokoban/          # Sokoban game
│   ├── maze3d/           # 3D Maze game
│   ├── pathfinder/       # PathFinder game
│   └── trapfield/        # TrapField game
├── generation/           # Dataset generation
│   ├── batch_generate.py # Batch generation tool
│   └── generate_videos.py # Video generation
├── evaluation/           # VLM evaluation
│   └── vlm_eval/        # Evaluation framework
├── config/              # Configuration files
│   ├── config.yaml      # Generation config
│   └── vlm/            # Evaluation configs
├── skins/              # Texture assets
└── scripts/            # Utility scripts
```

## 🎯 Usage Examples

### Generate Maze Dataset

```bash
# Edit config/config.yaml
game_type: "maze"
skins_root: "skins/maze"
difficulties:
  small:
    maze_size: 9
    count: 100

# Run generation
python -m generation.batch_generate config/config.yaml
```

### Evaluate on Sokoban

```bash
# Edit config/vlm/sokoban_eval.yaml
# Configure models and dataset path

# Run evaluation
python -m evaluation.vlm_eval.run_vlm_eval config/vlm/sokoban_eval.yaml
```

## 📊 Evaluation Metrics

- **Success Rate (SR)**: Percentage of levels solved correctly
- **Path Ratio (PR)**: Ratio of correct consecutive actions from the start
- **Move Ratio (MR)**: Binary metric for exact solution match
- **Step Count**: Number of actions in the solution

## 🔧 Configuration

### Generation Config (`config/config.yaml`)

- `game_type`: Game to generate (maze, sokoban, pathfinder, trapfield, maze3d)
- `skins_root`: Path to texture assets
- `difficulties`: Difficulty levels and parameters
- `generation.max_attempts`: Max attempts to generate valid level
- `parallel.max_workers`: Number of parallel workers

### Evaluation Config (`config/vlm/*.yaml`)

- `game`: Game type to evaluate
- `dataset`: Path to dataset
- `models`: List of VLMs to test
- `workers`: Number of parallel evaluation workers
- `max_levels`: Maximum levels to evaluate (-1 for all)

## 🎨 Custom Textures

Each game supports custom texture skins for visual variety:

1. Create a new folder under `skins/<game_name>/`
2. Add required texture images (PNG/JPG format)
3. Specify the skin path in configuration

Required texture files vary by game. Refer to existing skin folders for examples.

### Texture Requirements by Game

- **Maze**: wall, floor, player, goal
- **Sokoban**: wall, floor, player, box, target
- **PathFinder**: Custom background and path textures
- **TrapField**: floor, trap, player, goal

## 🔬 Adding New Games

VR-Bench uses an adapter pattern for easy extensibility:

1. Create a new game directory under `games/`
2. Implement the `GameAdapter` interface:
   - `generate_level()`: Level generation logic
   - `save_level()`: Save level data and render outputs
   - `get_level_hash()`: For deduplication
   - `is_duplicate()`: Duplicate detection
3. Implement game-specific logic and rendering
4. Create an executor in `evaluation/vlm_eval/executors/`
5. Register in `generation/batch_generate.py`

See existing game implementations for reference.

## 🐛 Troubleshooting

### Common Issues

**Issue**: CUDA out of memory during VLM inference
- **Solution**: Reduce batch size or use tensor parallelism with multiple GPUs

**Issue**: Video generation fails
- **Solution**: Ensure ffmpeg is installed: `pip install imageio-ffmpeg`

**Issue**: API rate limiting
- **Solution**: Reduce `workers` in evaluation config or add delays

**Issue**: Duplicate levels generated
- **Solution**: Increase `max_duplicate_retries` in generation config

## 📚 Citation

If you use VR-Bench in your research, please cite:

```bibtex
@misc{vrbench2025,
  title={VR-Bench: Visual Reasoning Benchmark for Vision-Language Models},
  author={VR-Bench Team},
  year={2025},
  url={https://github.com/SNHuan/VR-Bench}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🔗 Related Resources

- [Hugging Face Dataset](https://huggingface.co/datasets/amagipeng/VR-Bench)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

VR-Bench builds upon various open-source projects and research in visual reasoning and VLM evaluation.

## 📧 Contact

For questions and feedback, please open an issue on GitHub or contact the maintainers.

