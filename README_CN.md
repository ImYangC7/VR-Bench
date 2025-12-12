<div align="center">

# VR-Bench：视觉语言模型的视觉推理基准

</div>

<div align="center" style="font-size: 15pt">

<a href='https://arxiv.org/abs/2511.15065'><img src='https://img.shields.io/badge/Arxiv-2511.15065-purple'></a>
<a href='https://huggingface.co/papers/2511.15065'><img src='https://img.shields.io/badge/HF%20Paper-2511.15065-blue'></a>
<a href='https://imyangc7.github.io/VRBench_Web/'><img src='https://img.shields.io/badge/Project-Website-green'></a>
<a href='https://huggingface.co/datasets/amagipeng/VR-Bench'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow'></a>
<a href='https://huggingface.co/HY-Wan/Wan-R1'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange'></a>

</div>

中文文档 | [English](README.md)

VR-Bench 是一个综合基准，用于通过多种益智游戏评估视觉语言模型（VLM）在空间推理与规划任务上的能力，提供统一的数据生成、评估与分析框架。

> 如果在使用或复现过程中遇到问题，可直接联系我（Email: iamyangcheng7@gmail.com，微信: Forever_1k）。评估时的参数选择与破题策略会影响结果。

## 🎊 动态

- [2025.12.11] 我们为视频模型新增了动态提示词模板，同时在 prompts 目录下添加了用于生成 `metadata.csv` 文件的脚本 `generate_metadata.py`。更多详情请参考 [PR #10](https://github.com/FoundationAgents/VR-Bench/pull/10)
- [2025.12.10] 我们新增了一款 VLM 的动态提示词模板，该模板可适配不同的皮肤。更多详情请参考 [PR #9](https://github.com/FoundationAgents/VR-Bench/pull/9)
- [2025.12.03] 重构追踪器代码，提升代码规范性，并在 README 中补充了追踪器详细说明（NCC、Flow、CSRT）及使用示例。
- [2025.11.26] 抱歉此前遗漏了皮肤素材，现已补全并放入 skins 目录，方便正常生成。后续版本将用 nanobanana 支持自动皮肤生成，敬请关注。
- [2025.11.24] 发布用于训练 Wan-R1 的脚本与配置。
- [2025.11.19] 发布全部任务的评估代码。

## 🧩 基准概览

VR-Bench 总览： (A) 迷宫类型：包含规则迷宫、不规则迷宫、3D 迷宫、Trapfield、Sokoban，覆盖 2D/3D 场景与多样任务结构，提供丰富的空间推理情境。 (B) 视频推理范式：采用逐帧链式推理，要求模型输出帧级推断以体现序列化视觉推理。 (C) 基准表现：在所有迷宫类型上对 VLM 与视频模型进行四个核心指标的评估，凸显空间推理能力差异。 (D) 附加分析：支持难度泛化、纹理泛化、迷宫类型泛化以及测试时扩展等维度的评估，全面衡量鲁棒性与泛化能力。

![video reason](./resource/video_reason.svg)

为评估 VTR 任务的泛化能力并提升在多样迷宫场景中的鲁棒性，我们在两个维度上做变化：（1）**难度等级**：通过调整迷宫规模（如 5×5 到 7×7）、分支数量与障碍，设置简单/中等/困难；（2）**迷宫纹理**：用程序化与生成式纹理改变障碍、路径等组件，扩大视觉分布，缓解对干净合成环境的过拟合。

![variant](./resource/variant.svg)

## 🎮 支持的游戏

- **Regular Maze（常规迷宫）**：基础的空间导航与路径规划
- **Sokoban（推箱子）**：最高难度的逻辑任务，模型必须理解物体间的相互作用力与推动规则
- **3D Maze（3D 迷宫）**：引入高度与遮挡，测试模型在立体空间中的推理能力
- **PathFinder（不规则迷宫）**：摒弃网格，使用曲线路径，考验纯视觉感知而非坐标记忆
- **TrapField（陷阱场）**：要求模型不仅要由起点到终点，还必须避开特定的陷阱区域，考验反向约束逻辑

## ✨ 核心特性

- **程序化生成**：自动生成多样关卡，难度可配置
- **纹理自定义**：通过皮肤支持多种视觉主题
- **视频渲染**：生成 24 FPS 的平滑解题视频
- **VLM 评估**：内置多种 VLM 测试（GPT、Gemini、Qwen 等）
- **全面指标**：SR（成功率）、PR（精确率）、SD（步骤偏差）、EM（精确匹配）、MF（掩码保真度）
- **并行处理**：多线程生成与评估
- **去重机制**：自动检测并移除重复关卡

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
# 方案 A：直接调用 Python
# 编辑 config/config.yaml（game_type、skins_root、output_root、difficulties 等）
python -m generation.batch_generate config/config.yaml
python generation/generate_videos.py <DATASET_DIR> --workers <N> --skin <SKIN_PATH>

# 方案 B：使用脚本（等价调用）
bash scripts/generate_by_skins.sh config/config.yaml
bash scripts/generate_videos.sh <DATASET_DIR> [workers]
```

## 🏋️‍♂️ 训练模型

我们使用 [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) 进行扩散模型的训练与推理。安装方法：

```bash
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

安装完成后，**请先在训练脚本中更新数据集路径、超参数与输出目录**，再启动实验。

参考配置：

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

请根据自己的数据位置与实验需求修改。

## 🎯 评估方式

### 视频模型（轨迹推理）

```bash
# 将生成视频与 GT 轨迹对比（自动匹配难度）
bash scripts/videomodel_evaluate.sh

# 或直接运行
python evaluation/videomodel_eval/batch_evaluate.py \
  DATASET_DIR OUTPUT_DIR RESULT_DIR \   # DATASET_DIR=GT 数据集根目录，OUTPUT_DIR=模型输出目录，RESULT_DIR=评估输出目录
  --gpu   # 可选
```

#### 追踪器类型

轨迹提取系统支持三种追踪算法，可通过 `--tracker-type` 参数选择：

| 追踪器 | 参数值 | 算法 | 适用场景 |
|--------|--------|------|----------|
| **NCC** | `ncc` | 归一化互相关 | 固定外观目标（默认，推荐） |
| **Optical Flow** | `optical_flow` | Lucas-Kanade 稀疏光流 | 平滑连续运动 |
| **CSRT** | `csrt` | 判别相关滤波 | 可变形目标、部分遮挡 |

**NCC 追踪器**（默认，推荐）
- **算法**：使用 `cv2.TM_CCOEFF_NORMED`（归一化相关系数）进行模板匹配
- **优点**：速度快，对固定外观目标精度高，轨迹提取更稳定
- **缺点**：对旋转/缩放变化敏感
- **最适用于**：玩家图标外观固定的 puzzle 游戏视频（我们的主要场景）

**Optical Flow 追踪器**
- **算法**：Lucas-Kanade 金字塔光流追踪特征点
- **优点**：擅长处理连续运动，计算效率高
- **缺点**：长序列可能产生漂移，需要良好的特征点
- **最适用于**：运动平滑的轨迹视频

**CSRT 追踪器**
- **算法**：通道与空间可靠性追踪（OpenCV 内置）
- **优点**：对部分遮挡和形变具有鲁棒性
- **缺点**：在迷宫环境（如 Sokoban）中偶尔会丢失目标，速度较慢，需要 `opencv-contrib-python`
- **最适用于**：目标外观会变化的通用追踪场景

> **📝 论文复现说明**：论文中的结果使用的是 **CSRT 追踪器**。如需完全复现论文结果，请使用 `--tracker-type csrt`。但我们推荐日常使用 **NCC 追踪器**，因为它在 puzzle 游戏场景下提供更稳定、更准确的轨迹提取。

**使用示例**：

```bash
# 使用默认 NCC 追踪器（默认搜索边距 50px）
python evaluation/videomodel_eval/batch_evaluate.py DATASET OUTPUT RESULT

# 使用 NCC 全图搜索
python evaluation/videomodel_eval/batch_evaluate.py DATASET OUTPUT RESULT \
  --tracker-type ncc --search-margin 0

# 使用光流追踪器
python evaluation/videomodel_eval/batch_evaluate.py DATASET OUTPUT RESULT \
  --tracker-type optical_flow

# 使用 CSRT 追踪器
python evaluation/videomodel_eval/batch_evaluate.py DATASET OUTPUT RESULT \
  --tracker-type csrt
```

### VLM（规划/动作推理）

1）配置环境：`cp .env.example .env` 并填入 API 密钥、数据集路径、CUDA 等。  
2）（可选，本地模型）启动 VLM 服务：

```bash
bash scripts/start_sglang_server.sh
```

3）对数据集结果运行 VLM 评估：

```bash
bash scripts/run_vlm_eval.sh
```

## 📊 评估指标

- **PR（Precision Rate）**：重采样点落在 GT 路径小偏差范围内的比例，衡量轨迹形状一致性  
- **SR（Success Rate）**：生成轨迹（推箱子时用箱子/玩家轨迹）是否至少一次进入目标框  
- **SD（Step Deviation）**：路径长度相对超长比 `len_gen / len_gt - 1`，仅在 SR=1 且非负时定义  
- **EM（Exact Match）**：当 PR 超阈值且 |SD| 足够小时（且 SR=1）记为 1，否则为 0  
- **MF（Mask Fidelity）**：背景稳定度 [0,1]；比较采样帧与首帧（遮住起点/终点/玩家区域）衡量背景变化  

## 🧪 评测

训练完模型后，可用我们提供的推理脚本进行评测：

1. **复制推理脚本**：将评测脚本从 VR-Bench 复制到 DiffSynth-Studio：
   ```bash
   cp VR-Bench/scripts/Wan2.2-TI2V-5B_lora.py DiffSynth-Studio/examples/wanvideo/model_inference/
   ```

2. **更新路径**：根据您的环境编辑复制后的脚本：
   - 更新 LoRA 检查点路径
   - 更新输入图像路径
   - 更新输出视频路径
   - 按需自定义提示词

3. **运行评测**：
   ```bash
   cd DiffSynth-Studio/examples/wanvideo/model_inference/
   python Wan2.2-TI2V-5B_lora.py
   ```

脚本将基于您训练的模型生成视频，并保存到指定输出目录。

## 📁 项目结构

```
VR-Bench/
├── core/                   # 核心框架
├── games/                  # 游戏实现
├── generation/             # 数据生成
├── evaluation/
│  ├── videomodel_eval/     # 评估视频模型的轨迹推理
│  └── vlm_eval/            # 评估 VLM 的规划/动作推理
├── config/                 # 生成与评估配置
├── skins/                  # 纹理资源
└── scripts/                # 实用脚本
```

## 🔧 配置

### 生成配置（`config/config.yaml`）
- `game_type`：生成的游戏类型（maze/sokoban/pathfinder/trapfield/maze3d）
- `skins_root`：纹理资源路径
- `difficulties`：难度等级与参数
- `generation.max_attempts`：生成有效关卡的最大尝试次数
- `parallel.max_workers`：并行进程数

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

各游戏示例：
- Maze：wall, floor, player, goal
- Sokoban：wall, floor, player, box, target
- PathFinder：自定义背景与路径纹理
- TrapField：floor, trap, player, goal

## 🔬 添加新游戏

VR-Bench 采用适配器模式，扩展流程：

1. 在 `games/` 下创建新目录  
2. 实现 `GameAdapter` 接口：
   - `generate_level()`：关卡生成逻辑
   - `save_level()`：保存关卡数据并渲染输出
   - `get_level_hash()`：用于去重
   - `is_duplicate()`：重复检测
3. 编写游戏特定逻辑与渲染
4. 在 `evaluation/vlm_eval/executors/` 中添加执行器
5. 在 `generation/batch_generate.py` 中注册

可参考现有游戏实现。

## 🐛 问题排查

### 常见问题

**CUDA 内存不足（VLM 推理）**：减小 batch，或使用多卡张量并行  
**视频生成失败**：确保安装 ffmpeg：`pip install imageio-ffmpeg`  
**API 速率限制**：降低评估配置中的 `workers` 或增加延时  
**生成重复关卡**：提高生成配置的 `max_duplicate_retries`  

## 📚 引用

如在研究中使用 VR-Bench，请引用：

```bibtex
@article{yang2025vrbench,
  title={Reasoning via Video: The First Evaluation of Video Models' Reasoning Abilities through Maze-Solving Tasks},
  author={Cheng Yang and Haiyuan Wan and Yiran Peng and Xin Cheng and Zhaoyang Yu and Jiayi Zhang and Junchi Yu and Xinlei Yu and Xiawu Zheng and Dongzhan Zhou and Chenglin Wu},
  journal={arXiv preprint arXiv:2511.15065},
  year={2025}
}
```

## 🤝 贡献

欢迎提交 Pull Request。

## 🔗 相关资源

- [Hugging Face Dataset](https://huggingface.co/datasets/amagipeng/VR-Bench)

## 📝 许可证

本项目基于 MIT 许可证，详见 `LICENSE`。

## 🙏 致谢

感谢 [AutoEnv](https://github.com/FoundationAgents/AutoEnv)、[Game-RL](https://github.com/tongjingqi/Game-RL)、[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) 和 [MiniVeo3-Reasoner](https://github.com/thuml/MiniVeo3-Reasoner) 为本项目提供基础支持！

## 📧 联系方式

如有问题或反馈，请在 GitHub 提 Issue 或联系维护者。

