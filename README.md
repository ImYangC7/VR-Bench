<div align="center">

# VR-Bench: Visual Reasoning Benchmark for Vision-Language Models

</div>

<div align="center" style="font-size: 15pt">

<a href='https://arxiv.org/abs/2511.15065'><img src='https://img.shields.io/badge/Arxiv-2511.15065-purple'></a>
<a href='https://huggingface.co/papers/2511.15065'><img src='https://img.shields.io/badge/HF%20Paper-2511.15065-blue'></a>
<a href='https://imyangc7.github.io/VRBench_Web/'><img src='https://img.shields.io/badge/Project-Website-green'></a>
<a href='https://huggingface.co/datasets/amagipeng/VR-Bench'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow'></a>
<a href='https://huggingface.co/HY-Wan/Wan-R1'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange'></a>

</div>

[中文文档](README_CN.md) | English

VR-Bench is a comprehensive benchmark for evaluating Vision-Language Models (VLMs) on spatial reasoning and planning tasks through various puzzle games. It provides a unified framework for dataset generation, evaluation, and analysis.

## 🧩 Benchmark Overview

Overview of VR-Bench. (A) Maze Types. VR-Bench comprises five maze types—Regular Maze, Irregular Maze, 3D Maze, Trapfield, and Sokoban—covering both 2D and 3D settings as well as diverse task structures, yielding a broad range of spatial reasoning scenarios. (B) Reasoning via Video Paradigm. VR-Bench adopts a chain-of-frame reasoning paradigm, requiring models to produce frame-by-frame inferences that capture sequential visual reasoning. (C) Benchmark Performance. Leading VLMs and video models are evaluated on four core metrics across all maze types, revealing clear differences in spatial reasoning capability. (D) Additional Analysis. VR-Bench also supports evaluations on difficulty generalization, texture generalization, maze-type generalization, and test-time scaling, enabling a comprehensive assessment of model robustness and generalization.

![video reason](./resource/video_reason.svg)

To evaluate the generalization ability on the VTR task and enhance robustness in adapting to diverse maze scenarios, we introduce variations across two key dimensions: (1) **Difficulty Level**: We define three difficulty grades (Easy, Medium, and Hard) by adjusting the maze size (e.g., expanding from 5×5 to 7×7), modifying the number of maze branches, and adding obstacles; (2) **Maze Texture**: We vary the textures of maze obstacles, paths, and other components using textures generated via procedural methods and generative models, which exposes the policies to a broad visual distribution and mitigates overfitting to clean, synthetic environments.

![variant](./resource/variant.svg)

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
git clone https://github.com/ImYangC7/VR-Bench.git
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
@article{yang2025vrbench,
      title={Reasoning via Video: The First Evaluation of Video Models' Reasoning Abilities through Maze-Solving Tasks}, 
      author={Cheng Yang and Haiyuan Wan and Yiran Peng and Xin Cheng and Zhaoyang Yu and Jiayi Zhang and Junchi Yu and Xinlei Yu and Xiawu Zheng and Dongzhan Zhou and Chenglin Wu},
      journal={arXiv preprint arXiv:2511.15065},
      year={2025}
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

