# VideoCut — 智能视频合成与防重复系统

基于 [Feasibility.md](./Feasibility.md) 实现的云端素材库 + 分镜脚本驱动 + 防重复视频合成流水线。

## 功能概览

- **云端素材库**：Asset / Clip / ClipVariant 模型，支持按标签、时长、景别检索与多版本二次剪辑。
- **分镜脚本**：YAML/JSON 格式 Shot List，支持 `library` 与 `user_upload` 插槽。
- **视频合成**：FFmpeg 拼接、转场、缩放/裁剪、可选 BGM 与字幕。
- **防重复**：随机化参数（转场/滤镜/缩放）+ 成片帧哈希与历史库相似度检测，不通过则自动重试。

## 项目结构

```
VideoCut/
├── config/           # 配置
│   └── settings.py
├── models/           # 数据模型（Asset, Clip, ShotList, RandomParams 等）
├── agent/            # 编排与核心逻辑
│   └── graph.py      # 素材库、分镜解析、随机参数、FFmpeg 剪辑、成片查重与重试
├── api/              # FastAPI
│   └── main.py       # 防重复 API、合成触发
├── shot_lists/       # 分镜脚本示例
│   └── cruise_scenic_v1.yaml
├── Feasibility.md    # 技术可行性文档
├── .env.example      # 环境变量示例
└── requirements.txt
```

## 环境与运行

### 1. 安装依赖

```bash
cd VideoCut
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

**系统依赖**：需安装 [FFmpeg](https://ffmpeg.org/) 并加入 PATH。

### 2. 配置

可选：复制 `.env.example` 为 `.env` 并按需修改：

```bash
cp .env.example .env
```

示例 `.env` 项：

```env
SIMILARITY_THRESHOLD=0.85
MAX_RETRY_ON_DUPLICATE=3
FRAME_SAMPLE_INTERVAL_SEC=1.0
DATA_DIR=./data
DATABASE_URL=sqlite:///./qmt.db
```

### 3. 启动 API 服务

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

- 健康检查：`GET http://localhost:8000/health`
- 防重复参数：`GET http://localhost:8000/api/anti-duplicate/params`
- 成片查重：`POST http://localhost:8000/api/anti-duplicate/check`，body: `{"video_path": "绝对路径", "exclude_video_ids": [], "threshold": 0.85}`
- 触发生成：`POST http://localhost:8000/api/synthesis/run`，form: `shot_list_path=cruise_scenic_v1.yaml`，可选 `seed=12345`

### 4. 命令行直接跑合成（不经过 API）

在项目根目录下运行：

```python
from pathlib import Path
from agent.graph import run_synthesis

result = run_synthesis(
    shot_list_path=str(Path("shot_lists/cruise_scenic_v1.yaml").resolve()),
    user_uploads={"intro": "D:/path/to/your_intro.mp4"},  # 可选
    seed=42,
)
print(result)  # {"success": True, "output_path": "...", "video_id": "...", ...}
```

## 素材库使用（当前为内存存储）

```python
from models.asset import Asset, AssetType, AssetMetadata, Clip, ClipVariant, TransformType
from agent.graph import add_asset, add_clip, add_clip_variant, generate_all_variants_for_clip

# 注册一条素材
asset = Asset(
    type=AssetType.UPLOAD,
    source_file_path="D:/videos/cruise_deck.mp4",
    duration_sec=120,
    metadata=AssetMetadata(tags=["cruise", "deck", "wide"], shot_type="wide"),
)
add_asset(asset)

# 从素材切出一段 Clip
clip = Clip(asset_id=asset.id, in_point_sec=10, out_point_sec=25, duration_sec=15)
add_clip(clip)

# 为该 Clip 生成多版本（镜像、变速、裁剪）
variants = generate_all_variants_for_clip(clip, asset.source_file_path)
for v in variants:
    add_clip_variant(v)
    if clip.variant_ids is None:
        clip.variant_ids = []
    clip.variant_ids.append(v.id)
```

## 分镜脚本格式

见 `shot_lists/cruise_scenic_v1.yaml`：

- `name` / `version` / `total_duration_target`
- `shots`: 列表，每项含 `slot`、`type`（`library` | `user_upload`）、`user_slot_id`（用户片段 key）、`constraints`（tags、duration_range、shot_type）

## 防重复策略

1. **素材层**：每个 Clip 可对应多个 ClipVariant（镜像、变速、裁剪），合成时随机选用。
2. **合成层**：每次生成使用随机转场、时长、缩放、裁剪比、滤镜、BGM 索引。
3. **成片层**：生成后对视频做帧采样与感知哈希，与历史成片比对；相似度 ≥ 阈值则重试（更换 seed），直至通过或达到最大重试次数。

阈值与重试次数见 `config/settings.py` 或 `GET /api/anti-duplicate/params`。

## 扩展与生产化

- **持久化**：将 `agent/graph.py` 中素材库与防重复的内存字典改为 SQLAlchemy + PostgreSQL，指纹可存表或 Redis。
- **对象存储**：上传与成片输出改为 OSS/COS，`source_file_path` / `output_path` 存 URL 或 bucket key。
- **任务队列**：ClipVariant 生成、成片剪辑等改为 Celery/Dramatiq 异步任务。
- **LangGraph**：在 `agent/graph.py` 中进一步接入 LangGraph 状态图，实现更复杂分支与人工干预节点。

详细设计见 [Feasibility.md](./Feasibility.md)。
