# VR-Bench: 视觉语言模型的视觉推理基准测试

中文文档 | [English](README.md)

VR-Bench 是一个综合性的基准测试框架，用于评估视觉语言模型（VLMs）在空间推理和规划任务上的能力。通过多种益智游戏，提供统一的数据集生成、评估和分析框架。

## 🎮 支持的游戏

VR-Bench 包含五种不同的益智游戏，每种游戏测试视觉推理的不同方面：

- **Maze（迷宫）**: 在网格迷宫中从起点导航到终点
- **Sokoban（推箱子）**: 将箱子推到目标位置，同时避开墙壁
- **3D Maze（3D迷宫）**: 多层迷宫，通过梯子连接不同楼层
- **PathFinder（路径查找）**: 在带有标记路径点的不规则迷宫中寻找路径
- **TrapField（陷阱场）**: 在场地中导航，同时避开陷阱

## ✨ 核心特性

- **程序化生成**: 自动生成多样化的关卡，支持可配置的难度等级
- **纹理自定义**: 通过纹理皮肤支持自定义视觉主题
- **视频渲染**: 生成流畅的解决方案动画视频（24 FPS）
- **VLM评估**: 内置框架支持测试各种VLM（GPT、Gemini、Qwen等）
- **全面指标**: 成功率（SR）、路径比率（PR）、移动比率（MR）
- **并行处理**: 多线程生成和评估，提高效率
- **去重机制**: 自动检测和移除重复关卡

## 📋 环境要求

- Python >= 3.10
- CUDA兼容的GPU（可选，用于本地VLM推理）

## 🚀 快速开始

### 1. 安装

```bash
# 克隆仓库
git clone https://github.com/SNHuan/VR-Bench.git
cd VR-Bench

# 安装依赖
pip install -r requirements.txt
```

### 2. 环境配置

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，配置：
# - VLM评估所需的API密钥
# - 数据集路径
# - CUDA配置
```

### 3. 下载数据集

```bash
# 从 Hugging Face 下载预生成的数据集
python dataset_init.py --output-dir ./dataset_VR
```

### 4. 生成自定义关卡

```bash
# 编辑 config/config.yaml 配置游戏类型和难度
# 然后运行批量生成
python -m generation.batch_generate config/config.yaml
```

### 5. 评估VLM

```bash
# 启动本地VLM服务器（可选，用于本地模型）
bash scripts/start_sglang_server.sh

# 运行评估
bash scripts/run_vlm_eval.sh
```

## 📁 项目结构

```
VR-Bench/
├── core/                   # 核心框架
│   ├── schema/            # 统一状态表示
│   ├── renderer.py        # 基础渲染引擎
│   ├── texture_handler.py # 纹理管理
│   └── game_adapter.py    # 游戏适配器接口
├── games/                 # 游戏实现
│   ├── maze/             # 迷宫游戏
│   ├── sokoban/          # 推箱子游戏
│   ├── maze3d/           # 3D迷宫游戏
│   ├── pathfinder/       # 路径查找游戏
│   └── trapfield/        # 陷阱场游戏
├── generation/           # 数据集生成
│   ├── batch_generate.py # 批量生成工具
│   └── generate_videos.py # 视频生成
├── evaluation/           # VLM评估
│   └── vlm_eval/        # 评估框架
├── config/              # 配置文件
│   ├── config.yaml      # 生成配置
│   └── vlm/            # 评估配置
├── skins/              # 纹理资源
└── scripts/            # 实用脚本
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

### 在推箱子游戏上评估

```bash
# 编辑 config/vlm/sokoban_eval.yaml
# 配置模型和数据集路径

# 运行评估
python -m evaluation.vlm_eval.run_vlm_eval config/vlm/sokoban_eval.yaml
```

## 📊 评估指标

- **成功率 (SR)**: 正确解决的关卡百分比
- **路径比率 (PR)**: 从起点开始连续正确动作的比率
- **移动比率 (MR)**: 完全匹配最优解的二元指标
- **步数统计**: 解决方案中的动作数量

## 🔧 配置说明

### 生成配置 (`config/config.yaml`)

- `game_type`: 要生成的游戏类型（maze, sokoban, pathfinder, trapfield, maze3d）
- `skins_root`: 纹理资源路径
- `difficulties`: 难度等级和参数
- `generation.max_attempts`: 生成有效关卡的最大尝试次数
- `parallel.max_workers`: 并行工作进程数

### 评估配置 (`config/vlm/*.yaml`)

- `game`: 要评估的游戏类型
- `dataset`: 数据集路径
- `models`: 要测试的VLM列表
- `workers`: 并行评估工作进程数
- `max_levels`: 最大评估关卡数（-1表示全部）

## 🎨 自定义纹理

每个游戏都支持自定义纹理皮肤：

1. 在 `skins/<game_name>/` 下创建新文件夹
2. 添加所需的纹理图片（PNG/JPG格式）
3. 在配置文件中指定皮肤路径

所需纹理文件因游戏而异，请参考现有皮肤文件夹。

### 各游戏纹理要求

- **Maze**: wall, floor, player, goal
- **Sokoban**: wall, floor, player, box, target
- **PathFinder**: 自定义背景和路径纹理
- **TrapField**: floor, trap, player, goal

## 🔬 扩展新游戏

VR-Bench 使用适配器模式，便于添加新游戏：

1. 在 `games/` 下创建新游戏目录
2. 实现 `GameAdapter` 接口：
   - `generate_level()`: 关卡生成逻辑
   - `save_level()`: 保存关卡数据和渲染输出
   - `get_level_hash()`: 用于去重
   - `is_duplicate()`: 重复检测
3. 实现游戏特定逻辑和渲染
4. 在 `evaluation/vlm_eval/executors/` 创建执行器
5. 在 `generation/batch_generate.py` 中注册

详细说明请参考现有游戏实现。

## 🐛 常见问题

### 问题排查

**问题**: VLM推理时CUDA内存不足
- **解决方案**: 减小批处理大小或使用多GPU张量并行

**问题**: 视频生成失败
- **解决方案**: 确保已安装ffmpeg：`pip install imageio-ffmpeg`

**问题**: API速率限制
- **解决方案**: 减少评估配置中的 `workers` 数量或添加延迟

**问题**: 生成重复关卡
- **解决方案**: 增加生成配置中的 `max_duplicate_retries`

**问题**: 纹理加载失败
- **解决方案**: 检查纹理文件格式（支持PNG/JPG）和路径配置

## 💡 最佳实践

### 数据集生成

1. **从小规模开始**: 先生成少量关卡测试配置
2. **验证可解性**: 确保 `check_solvable: true`
3. **使用多皮肤**: 为同一游戏准备多个纹理皮肤增加多样性
4. **合理设置难度**: 根据目标逐步增加难度参数

### VLM评估

1. **预热模型**: 首次运行前先测试API连接
2. **监控成本**: 使用本地模型或设置 `max_levels` 限制
3. **保存结果**: 评估结果自动保存到 `output_dir`
4. **批量测试**: 在配置文件中列出多个模型进行对比

## 📊 性能优化

- **并行生成**: 根据CPU核心数调整 `max_workers`
- **GPU利用**: 使用SGLang进行高效的本地VLM推理
- **缓存模型**: 设置 `HF_CACHE_DIR` 避免重复下载
- **视频压缩**: 调整FPS和分辨率平衡质量与文件大小

## 📚 引用

如果您在研究中使用了 VR-Bench，请引用：

```bibtex
@misc{vrbench2025,
  title={VR-Bench: Visual Reasoning Benchmark for Vision-Language Models},
  author={VR-Bench Team},
  year={2025},
  url={https://github.com/SNHuan/VR-Bench}
}
```

## 🤝 贡献

欢迎贡献！请随时提交 Pull Request。对于重大更改：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 贡献指南

- 遵循现有代码风格
- 添加适当的注释和文档
- 确保所有测试通过
- 更新相关文档

## 🔗 相关资源

- [Hugging Face Dataset](https://huggingface.co/datasets/amagipeng/VR-Bench)

## 📝 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件。

## 🙏 致谢

VR-Bench 基于多个开源项目和视觉推理、VLM评估领域的研究成果。

## 📧 联系方式

如有问题和反馈，请在 GitHub 上提交 issue 或联系维护者。

