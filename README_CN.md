<div align="center">

# VR-Bench: 视觉语言模型的视觉推理基准

</div>

<div align="center" style="font-size: 15pt">

<a href='https://arxiv.org/abs/2511.15065'><img src='https://img.shields.io/badge/Arxiv-2511.15065-purple'></a>
<a href='https://huggingface.co/papers/2511.15065'><img src='https://img.shields.io/badge/HF%20Paper-2511.15065-blue'></a>
<a href='https://imyangc7.github.io/VRBench_Web/'><img src='https://img.shields.io/badge/Project-Website-green'></a>
<a href='https://huggingface.co/datasets/amagipeng/VR-Bench'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow'></a>
<a href='https://huggingface.co/HY-Wan/Wan-R1'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange'></a>

</div>

中文文档 | [English](README.md)

VR-Bench 是一个综合基准，用于通过多种益智游戏评估视觉语言模型（VLMs）在空间推理与规划任务上的表现，提供统一的数据生成、评估与分析框架。

## 🧩 基准概览

VR-Bench 总览：（A）迷宫类型：包含规则/不规则/3D 迷宫、Trapfield、Sokoban，覆盖 2D/3D 场景与多样任务结构，提供丰富的空间推理情境。（B）视频推理范式：采用逐帧链式推理，要求模型输出帧级推断以体现序列化视觉推理。（C）基准表现：在所有迷宫类型上对 VLM 与视频模型进行四个核心指标的评估，凸显空间推理能力差异。（D）附加分析：支持难度、纹理、迷宫类型泛化以及测试时扩展等维度的评估，全面衡量鲁棒性与泛化能力。

![video reason](./resource/video_reason.svg)

为评估 VTR 任务的泛化能力并提升在多样迷宫场景中的鲁棒性，我们在两个维度上做变换：（1）**难度等级**：通过调整迷宫规模（如 5×5 到 7×7）、分支数量与障碍，设置简单/中等/困难；（2）**迷宫纹理**：使用程序化与生成式纹理改变障碍、路径等组件，扩大视觉分布，缓解对干净合成环境的过拟合。

![variant](./resource/variant.svg)

## 🎮 支持的游戏

- **Maze（迷宫）**：网格迷宫中从起点到终点
- **Sokoban（推箱子）**：推箱到目标且避开墙壁
- **3D Maze（3D 迷宫）**：多层迷宫，梯子连接楼层
- **PathFinder（路径查找）**：不规则迷宫中带标记路径点的寻路
- **TrapField（陷阱场）**：避开陷阱完成导航

## ✨ 核心特性

- 程序化生成：多样关卡，难度可配置
- 纹理自定义：支持自定义视觉主题
- 视频渲染：24 FPS 流畅解题视频
- VLM 评估：内置多种 VLM 测试（GPT、Gemini、Qwen 等）
- 全面指标：SR、PR、MR
- 并行处理：多线程生成与评估
- 去重机制：自动检测/移除重复关卡

## 📋 环境要求

- Python >= 3.10
- CUDA 兼容 GPU（可选，用于本地 VLM 推理）

## 🚀 快速开始

### 1. 安装

```bash
# 克隆仓库
git clone https://github.com/ImYangC7/VR-Bench.git
cd VR-Bench

# 安装依赖
pip install -r requirements.txt
```

### 2. 下载数据集

```bash
# 从 Hugging Face 下载预生成的数据集
python dataset_init.py --output-dir ./dataset_VR
```

### 3. 生成自定义关卡

```bash
# 编辑 config/config.yaml 配置游戏类型和难度
# 然后运行批量生成
python -m generation.batch_generate config/config.yaml
```

## 🎯 模型评估

### 视频模型（轨迹推理）

```bash
# 基于生成视频与 GT 轨迹对比，自动匹配难度
bash scripts/videomodel_evaluate.sh

# 或直接调用
python evaluation/videomodel_eval/batch_evaluate.py \
  DATASET_DIR OUTPUT_DIR RESULT_DIR \   # DATASET_DIR=GT 数据集根目录，OUTPUT_DIR=模型输出目录，RESULT_DIR=评估结果目录
  --threshold 0.05 \
  --num-samples 1000 \
  --workers 4 \
  --fidelity-pixel-threshold 5 \
  --frame-step 1 \
  --tracker-type template \
  --search-margin 50 \
  --gpu   # 可选
```

### VLM（规划/动作推理）

1）配置环境：`cp .env.example .env`，填写 API 密钥、数据集路径、CUDA 等。  
2）（可选，本地模型）启动 VLM 服务：

```bash
bash scripts/start_sglang_server.sh
```

3）对数据集结果运行评估：

```bash
bash scripts/run_vlm_eval.sh
```

## 📁 项目结构

```
VR-Bench/
├── core/                   # 核心框架
├── games/                  # 游戏实现
├── generation/             # 数据集生成
├── evaluation/
│   ├── videomodel_eval/    # 评估视频模型的轨迹推理
│   └── vlm_eval/           # 评估 VLM 的规划/动作推理
├── config/                 # 生成与评估配置
├── skins/                  # 纹理资源
└── scripts/                # 实用脚本
```

## 🎯 使用示例

### 生成迷宫数据集

```bash
# 编辑 config/config.yaml
game_type: "maze"
skins_root: "skins/maze"
difficulties:
  small:
    maze_size: 9
    count: 100

# 运行生成
python -m generation.batch_generate config/config.yaml
```

### 评估视频模型

```bash
bash scripts/videomodel_evaluate.sh
```

### 评估 VLM

```bash
python -m evaluation.vlm_eval.run_vlm_eval config/vlm/sokoban_eval.yaml
```

## 📊 评估指标

### 视频模型指标（videomodel_eval）
- **PR（Precision Rate）**：重采样后，与 GT 路径距离在阈值内的点占比，衡量轨迹形状贴合度。
- **SR（Success Rate）**：生成轨迹（推箱子时用箱子轨迹）是否进入目标框，取值 0/1。
- **SD（Step Deviation）**：路径长度相对超长比例 `len_gen / len_gt - 1`，仅在 SR=1 且非负时有效。
- **EM（Exact Match）**：在 SR=1 且 PR/|SD| 达到阈值时记为 1，否则 0。
- **MF（Mask Fidelity）**：背景稳定度 [0,1]；对比采样帧与首帧（遮掉起点/终点/玩家区域）衡量背景变化。

### VLM 指标（vlm_eval）
- **SR / PR / MR / Step**：成功率、路径正确性、匹配率和步数（由 VLM 评估器定义）。

## 🏋️‍♂️ 训练模型

我们使用 [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) 进行扩散模型的训练和推理。安装方法：

```bash
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

安装完成后，**确保在启动实验前更新训练脚本中的数据集路径、超参数和输出目录**。

参考配置如下：

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

请根据您的具体数据位置编辑上述脚本。

## 🧪 评测

训练完模型后，您可以使用我们提供的推理脚本进行评测：

1. **复制推理脚本**：将评测脚本从 VR-Bench 复制到 DiffSynth-Studio：
   ```bash
   cp VR-Bench/scripts/Wan2.2-TI2V-5B_lora.py DiffSynth-Studio/examples/wanvideo/model_inference/
   ```

2. **更新路径**：编辑复制的脚本，根据您的设置更新路径：
   - 更新 LoRA 检查点路径
   - 更新输入图像路径
   - 更新输出视频路径
   - 根据需要自定义提示词

3. **运行评测**：
   ```bash
   cd DiffSynth-Studio/examples/wanvideo/model_inference/
   python Wan2.2-TI2V-5B_lora.py
   ```

脚本将基于您训练的模型生成视频，并保存到指定的输出目录。

## 🔧 配置

### 生成配置（`config/config.yaml`）
- `game_type`：生成的游戏类型（maze/sokoban/pathfinder/trapfield/maze3d）
- `skins_root`：纹理资源路径
- `difficulties`：难度等级与参数
- `generation.max_attempts`：生成有效关卡的最大尝试次数
- `parallel.max_workers`：并行工作进程数

### VLM 评估配置（`config/vlm/*.yaml`）
- `game`：评估的游戏类型
- `dataset`：数据集路径
- `models`：待测 VLM 列表
- `workers`：并行评估进程数
- `max_levels`：最大评估关卡数（-1 表示全部）

## 🎨 自定义纹理

1. 在 `skins/<game_name>/` 下创建新文件夹  
2. 添加所需纹理（PNG/JPG）  
3. 在配置中指定皮肤路径  

各游戏所需纹理请参考现有皮肤目录：
- Maze：wall, floor, player, goal
- Sokoban：wall, floor, player, box, target
- PathFinder：自定义背景与路径纹理
- TrapField：floor, trap, player, goal

## 🔬 扩展新游戏

1. 在 `games/` 下创建新目录  
2. 实现 `GameAdapter` 接口（generate_level/save_level/get_level_hash/is_duplicate）  
3. 编写游戏逻辑与渲染  
4. 在 `evaluation/vlm_eval/executors/` 中添加执行器  
5. 在 `generation/batch_generate.py` 中注册  

## 🐛 问题排查

- **CUDA OOM**：减小 batch 或用多 GPU 并行  
- **视频生成失败**：确保安装 ffmpeg：`pip install imageio-ffmpeg`  
- **API 速率限制**：减少 `workers` 或增加延时  
- **生成重复关卡**：提高 `max_duplicate_retries`  
- **纹理加载失败**：检查纹理格式与路径  

## 📚 引用

```bibtex
@article{yang2025vrbench,
      title={Reasoning via Video: The First Evaluation of Video Models' Reasoning Abilities through Maze-Solving Tasks}, 
      author={Cheng Yang and Haiyuan Wan and Yiran Peng and Xin Cheng and Zhaoyang Yu and Jiayi Zhang and Junchi Yu and Xinlei Yu and Xiawu Zheng and Dongzhan Zhou and Chenglin Wu},
      journal={arXiv preprint arXiv:2511.15065},
      year={2025}
}
```

## 🤝 贡献

欢迎提交 Pull Request。对于重大改动：
1. Fork 仓库并创建分支  
2. 提交更改并更新相关文档/注释  
3. 确认测试通过后发起 PR  

## 🔗 相关资源

- [Hugging Face Dataset](https://huggingface.co/datasets/amagipeng/VR-Bench)

## 📝 许可证

MIT 许可证，详见 `LICENSE`。

## 🙏 致谢

感谢视觉推理与 VLM 领域的相关开源项目与研究成果。

## 📧 联系方式

如有问题与反馈，请在 GitHub 提 Issue 或联系维护者。
