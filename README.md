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

> If you encounter any difficulties in using or reproducing the code, please contact me directly (Email: iamyangcheng7@gmail.com, Wechat: Forever_1k). The parameter settings during the evaluation process and the selection of crackers will affect the evaluation results.

![](https://raw.githubusercontent.com/ImYangC7/Repo-recorder/main/generated/FoundationAgents__VR-Bench_status.svg)

## 🎊 News

- [2025.12.11] We have added dynamic prompt templates for video models, as well as the `metadata.csv` generator script `generate_metadata.py` in the prompts folder. Please refer to [PR #10](https://github.com/FoundationAgents/VR-Bench/pull/10).
- [2025.12.10] We have added a dynamic prompt template for VLM that supports adaptation to different skins. Please refer to [PR #9](https://github.com/FoundationAgents/VR-Bench/pull/9).
- [2025.12.03] Refactored tracker code for improved standardization and added comprehensive tracker documentation (NCC, Optical Flow, CSRT) with usage examples.
> **📝 Note on Paper Reproduction**: The results in our paper were obtained using **CSRT tracker**. If you want to exactly reproduce the paper results, use `--tracker-type csrt`. However, we recommend **NCC tracker** for general use as it provides more stable and accurate trajectory extraction in puzzle game scenarios.
- [2025.11.26] We apologize for the earlier omission. We have now added all our current maze textures to the skin folder to enable normal generation. In future releases, we will use nanobanana to support automatic skin generation. Please follow our updates.
- [2025.11.24] We have released the training scripts and corresponding configurations used to train Wan-R1.
- [2025.11.19] We have released evaluation code for all tasks.
  
## 🧩 Benchmark Overview

Overview of VR-Bench. (A) Maze Types. VR-Bench comprises five maze types—Regular Maze, Irregular Maze, 3D Maze, Trapfield, and Sokoban—covering both 2D and 3D settings as well as diverse task structures, yielding a broad range of spatial reasoning scenarios. (B) Reasoning via Video Paradigm. VR-Bench adopts a chain-of-frame reasoning paradigm, requiring models to produce frame-by-frame inferences that capture sequential visual reasoning. (C) Benchmark Performance. Leading VLMs and video models are evaluated on four core metrics across all maze types, revealing clear differences in spatial reasoning capability. (D) Additional Analysis. VR-Bench also supports evaluations on difficulty generalization, texture generalization, maze-type generalization, and test-time scaling, enabling a comprehensive assessment of model robustness and generalization.

![video reason](./resource/video_reason.svg)

To evaluate the generalization ability on the VTR task and enhance robustness in adapting to diverse maze scenarios, we introduce variations across two key dimensions: (1) **Difficulty Level**: We define three difficulty grades (Easy, Medium, and Hard) by adjusting the maze size (e.g., expanding from 5×5 to 7×7), modifying the number of maze branches, and adding obstacles; (2) **Maze Texture**: We vary the textures of maze obstacles, paths, and other components using textures generated via procedural methods and generative models, which exposes the policies to a broad visual distribution and mitigates overfitting to clean, synthetic environments.

![variant](./resource/variant.svg)

## 🎮 Supported Games

VR-Bench includes five different puzzle games, each testing different aspects of visual reasoning:

- **Regular Maze**: Basic spatial navigation and path planning in grid-based mazes
- **Sokoban**: Push boxes to target positions, requiring understanding of object interactions and push mechanics (highest logical difficulty)
- **3D Maze**: Multi-level maze with height and occlusion, testing reasoning ability in 3D space
- **PathFinder (Irregular Maze)**: Navigate through irregular mazes with curved paths, testing pure visual perception without coordinate memory
- **TrapField**: Navigate from start to goal while avoiding specific trap regions, testing constraint-based reasoning

## ✨ Key Features

- **Procedural Generation**: Automatically generate diverse puzzle levels with configurable difficulty
- **Texture Customization**: Support for custom visual themes through texture skins
- **Video Rendering**: Generate solution videos with smooth animations (24 FPS)
- **VLM Evaluation**: Built-in framework for testing various VLMs (GPT, Gemini, Qwen, etc.)
- **Comprehensive Metrics**: SR (Success Rate), PR (Precision Rate), SD (Step Deviation), EM (Exact Match), MF (Mask Fidelity)
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

### 2. Download Dataset

```bash
# Download pre-generated dataset from Hugging Face
python dataset_init.py --output-dir ./dataset_VR
```

### 3. Generate Custom Levels

```bash
# Option A: call Python directly
# Edit config/config.yaml to configure game type, skins_root, output_root, and difficulties
python -m generation.batch_generate config/config.yaml
python generation/generate_videos.py <DATASET_DIR> --workers <N> --skin <SKIN_PATH>

# Option B: use the helper shell scripts (equivalent to the above)
bash scripts/generate_by_skins.sh config/config.yaml
bash scripts/generate_videos.sh <DATASET_DIR> [workers]
```

## 🏋️‍♂️ Training Models

We use [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) for diffusion model training and inference. To install:

```bash
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

After installation, **make sure to update your dataset paths, hyperparameters, and output directory in the training script** before launching your experiment.

Here is a reference configuration:

```bash
accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/example_video_dataset \
  --dataset_metadata_path data/example_video_dataset/metadata.csv \
  --height 512 \
  --width 512 \
  --num_frames 193 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-TI2V-5B_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image" 
```

Edit the script above with your specific data locations.


## 🧪 Inference

After training your model, you can run inference with the provided script:

1. **Copy the inference script**: Copy the evaluation script from VR-Bench to DiffSynth-Studio:
   ```bash
   cp VR-Bench/scripts/Wan2.2-TI2V-5B_lora.py DiffSynth-Studio/examples/wanvideo/model_inference/
   ```

2. **Update paths**: Edit the copied script to update the paths according to your setup:
   - Update the LoRA checkpoint path
   - Update the input image path
   - Update the output video path
   - Customize the prompt as needed

3. **Run inference**:
   ```bash
   cd DiffSynth-Studio/examples/wanvideo/model_inference/
   python Wan2.2-TI2V-5B_lora.py
   ```

The script will generate videos based on your trained model and save them to the specified output directory.


## 🎯 Evaluation Method

### Video models (trajectory reasoning)

```bash
# Evaluate generated videos against GT trajectories (auto-matches difficulties)
bash scripts/videomodel_evaluate.sh

# Or run directly
python evaluation/videomodel_eval/batch_evaluate.py \
  DATASET_DIR OUTPUT_DIR RESULT_DIR \   # DATASET_DIR=GT dataset root, OUTPUT_DIR=model outputs, RESULT_DIR=eval outputs
  --gpu   # optional
```

#### Tracker Types

The trajectory extraction system supports three tracking algorithms, selectable via `--tracker-type`:

| Tracker | Parameter | Algorithm | Best For |
|---------|-----------|-----------|----------|
| **NCC** | `ncc` | Normalized Cross-Correlation | Fixed-appearance targets (default, recommended) |
| **Optical Flow** | `optical_flow` | Lucas-Kanade Sparse Optical Flow | Smooth continuous motion |
| **CSRT** | `csrt` | Discriminative Correlation Filter | Deformable targets, partial occlusion |

**NCC Tracker** (Default, Recommended)
- **Algorithm**: Template matching using `cv2.TM_CCOEFF_NORMED` (normalized correlation coefficient)
- **Pros**: Fast, highly accurate for fixed-appearance objects, more stable trajectory extraction
- **Cons**: Sensitive to rotation/scale changes
- **Best for**: Puzzle game videos where player icons have fixed appearance (our main use case)

**Optical Flow Tracker**
- **Algorithm**: Lucas-Kanade pyramid optical flow tracking feature points
- **Pros**: Handles continuous motion well, computationally efficient
- **Cons**: May drift over long sequences, requires good feature points
- **Best for**: Smooth trajectory videos with gradual movements

**CSRT Tracker**
- **Algorithm**: Channel and Spatial Reliability Tracking (OpenCV built-in)
- **Pros**: Robust to partial occlusion and deformation
- **Cons**: May occasionally lose target in maze environments (e.g., Sokoban), slower, requires `opencv-contrib-python`
- **Best for**: General-purpose tracking with appearance changes


**Usage Examples**:

```bash
# Use default NCC tracker （default search margin 50px）
python evaluation/videomodel_eval/batch_evaluate.py DATASET OUTPUT RESULT

# Use NCC with full-image search
python evaluation/videomodel_eval/batch_evaluate.py DATASET OUTPUT RESULT \
  --tracker-type ncc --search-margin 0

# Use optical flow tracker
python evaluation/videomodel_eval/batch_evaluate.py DATASET OUTPUT RESULT \
  --tracker-type optical_flow

# Use CSRT tracker
python evaluation/videomodel_eval/batch_evaluate.py DATASET OUTPUT RESULT \
  --tracker-type csrt
```

### VLMs (planning/action reasoning)

1) Configure environment: `cp .env.example .env` and fill API keys, dataset paths, CUDA, etc.  
2) (Optional/local models) start the VLM service:

```bash
bash scripts/start_sglang_server.sh
```

3) Run VLM evaluation on the dataset results:

```bash
bash scripts/run_vlm_eval.sh
```


## 📊 Evaluation Metrics

- **PR (Precision Rate)**: Fraction of resampled points that stay within a small tolerance to the GT path; measures path shape consistency.
- **SR (Success Rate)**: Whether the generated trajectory (player or box for Sokoban) enters the goal bounding box at least once.
- **SD (Step Deviation)**: Relative path-length overrun vs GT (`len_gen / len_gt - 1`), only defined when SR=1 and non-negative.
- **EM (Exact Match)**: Perfect flag (1/0) when PR exceeds a threshold and |SD| is small, conditioned on SR=1.
- **MF (Mask Fidelity)**: Background stability score [0,1]; compares sampled frames to the first frame while masking start/goal/player regions.



## 📁 Project Structure

```
VR-Bench/
├── core/                   # Core framework
├── games/                  # Game implementations
├── generation/             # Dataset generation
├── evaluation/
│   ├── videomodel_eval/    # Evaluate video models’ trajectory reasoning
│   └── vlm_eval/           # Evaluate VLMs’ planning / action reasoning
├── config/                 # Generation & evaluation configs
├── skins/                  # Texture assets
└── scripts/                # Utility scripts
```

## 🔧 Configuration

### Generation Config (`config/config.yaml`)

- `game_type`: Game to generate (maze, sokoban, pathfinder, trapfield, maze3d)
- `skins_root`: Path to texture assets
- `difficulties`: Difficulty levels and parameters
- `generation.max_attempts`: Max attempts to generate valid level
- `parallel.max_workers`: Number of parallel workers

### VLM Evaluation Config (`config/vlm/*.yaml`)

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

Contributions are welcome! Please feel free to submit a Pull Request. 

## 🔗 Related Resources

- [Hugging Face Dataset](https://huggingface.co/datasets/amagipeng/VR-Bench)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

Thanks to [AutoEnv](https://github.com/FoundationAgents/AutoEnv), [Game-RL](https://github.com/tongjingqi/Game-RL), [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), and [MiniVeo3-Reasoner](https://github.com/thuml/MiniVeo3-Reasoner) for providing basic support for this project!

## 📧 Contact

For questions and feedback, please open an issue on GitHub or contact the maintainers.

