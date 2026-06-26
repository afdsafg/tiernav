# TierNav 实际模块与工作流文档

<!-- 根据 src/ 实际代码对照 docs/plans/ 中的 8 份设计/实现计划整理 -->
<!-- 生成日期: 2026-06-25 -->

---

## 1. 项目概览

TierNav 是一个层级化内存引导探索（HM-GE）导航代理，运行在 Habitat-sim 室内 3D 模拟器中，用于回答有关场景的具身问题（AEQA / GOATBench 基准测试）。

### 1.1 三种运行模式

| 模式 | 入口脚本 | 核心函数 | 状态 |
|------|----------|----------|------|
| **Legacy 单步评估** | `run_aeqa_evaluation.py` / `run_goatbench_evaluation.py` | 单步 VLM 查询循环 | 保留但不再维护 |
| **HM-GE 6 阶段代理** | `run_hmge_evaluation.py` | `agent_workflow.run_episode()` | 实现完成 |
| **Two-Tier Planner-Executor** | `run_two_tier_aeqa_evaluation.py` | `agent_workflow.run_episode_two_tier()` | 实现完成，支持多进程分片 |

---

## 2. 实际模块结构

```
src/
├── const.py                    # 环境变量常量（API key/url/model, GD路径）
├── agent_workflow.py           # ★ 核心：两套工作流 + VLM API客户端 + 阶段提示词
├── agent_planner.py            # ★ Two-Tier 上层Planner（mimo-v2.5 API, 4组件prompt）
├── agent_executor.py           # ★ Two-Tier 下层Executor（6工具 dispatch）
├── agent_tools.py              # 7个底层工具函数 + 静默感知 + 导航循环
├── agent_memory.py             # MemoryStore：CLIP快照存储与检索
├── agent_notebook.py           # ★ EvidenceNotebook：跨阶段证据追踪与循环检测
├── agent_evidence.py           # ★ TrajectoryEvidence：轨迹证据压缩为notebook条目
├── agent_context.py            # ContextManager：阶段间上下文桥接（HM-GE模式用）
├── agent_image_utils.py        # 图像工具：base64编码、马赛克拼接、matplotlib转换
├── seed_views.py               # SeedViewManager：种子视图懒更新管理
├── scene_aeqa.py               # AEQA场景管理 + GD导航链 + 质量过滤
├── scene_goatbench.py          # GOATBench场景管理
├── tsdf_base.py                # TSDF体素融合基础
├── tsdf_planner.py             # TSDF规划器 + 房间分割 + frontier提取 + agent_step
├── geom.py                     # 几何工具：Bresenham射线、碰撞检测、坐标转换
├── habitat.py                  # Habitat-sim包装器
├── utils.py                    # 通用工具
├── hierarchy_clustering.py     # 快照层次聚类（BisectingKmeans）
├── query_vlm_aeqa.py           # Legacy AEQA VLM查询
├── query_vlm_goatbench.py      # Legacy GOATBench VLM查询
├── eval_utils_gpt_aeqa.py      # Legacy AEQA GPT评估
├── eval_utils_gpt_goatbench.py # Legacy GOATBench GPT评估
├── goatbench_utils.py          # GOATBench数据准备
├── logger_aeqa.py              # AEQA实验日志
├── logger_goatbench.py         # GOATBench实验日志
└── conceptgraph/               # ConceptGraph SLAM子系统
    ├── slam/                   # 实时3D地图构建
    │   ├── mapping.py          # 物体级关联与合并
    │   ├── slam_classes.py     # DetectionList/MapObjectList等数据结构
    │   └── ...
    └── utils/                  # 几何、IoU、可视化、VLM工具
        ├── geometry.py, ious.py, vis.py, vlm.py, ...
        └── ...
```

---

## 3. 工作流详解

### 3.1 HM-GE 6 阶段工作流 (`run_episode`)

**入口:** `run_episode()` in `src/agent_workflow.py:409`

**状态机流程:**

```
Stage 1 (全景) → Stage 2 (方向决策) → Stage 2.5a (种子选择) → ...
                                          ↓                        ↓
                                     Stage 3 (物体选择)     Stage 6 (frontier选择)
                                          ↓                        ↓
                                     Stage 4 (GD导航)        Stage 5 (重决策)
                                          ↓                        ↓
                                     Stage 5 (重决策) ←───────────┘
                                          ↓
                               submit_answer → done
```

**各阶段详情:**

| 阶段 | VLM调用 | 输入 | 输出 | 代码位置 |
|------|---------|------|------|----------|
| Stage 1 | 无 | 初始位置 | 8视角全景mosaic + 房间分割 + SeedViewManager注册 | `agent_workflow.py:531-546` |
| Stage 2 | 第1次 | 全景mosaic + STAGE2_PROMPT | navigate_to_object(view_idx) 或 explore_other_room | `agent_workflow.py:559-599` |
| Stage 2.5a | 第2次 | 种子视图mosaic | explore_seed(seed_id) 或 explore_frontier | `agent_workflow.py:601-655` |
| Stage 3 | 第3次 | 选定视角大图 | 具体物体名称 | `agent_workflow.py:657-704` |
| Stage 4 | 无(GD代码) | 物体名称 + 视角参数 | 导航结果(pts, angle, success) | `agent_workflow.py:689-703` |
| Stage 5 | 第4次 | 3视角正面mosaic | submit_answer / navigate_to_object / explore_other_room | `agent_workflow.py:706-778` |
| Stage 6 | 第5次 | frontier列表 + 可视化图 | 选定frontier_id → 导航 → Stage 5 | `agent_workflow.py:780-824` |

**关键机制:**
- `_parse_vlm_response()` (`agent_workflow.py:982`)：强制要求 `reason` 字段，2次连续缺失则fallback
- `_is_valid_object_desc()` (`agent_workflow.py:1579`)：拒绝方向词/房间名/纯数字作为物体描述
- 连续missing_reason计数器：Stage 2/2.5a/3/5/6各有独立的2次连续fallback策略

### 3.2 Two-Tier Planner-Executor 工作流 (`run_episode_two_tier`)

**入口:** `run_episode_two_tier()` in `src/agent_workflow.py:1086`

**架构:**

```
Planner (agent_planner.py)
  ├── 4组件提示词构建: Question / History / Scene Analysis / Progress / Actions
  ├── 调用 mimo-v2.5 API (通过 call_vlm)
  ├── 解析 JSON → PlannerAction dataclass
  └── 关键词回退解析
       │
       ▼ PlannerAction
       │
Executor (agent_executor.py)
  ├── execute_action() 分发到6个工具
  │   ├── explore_panorama()
  │   ├── navigate_to_object(object_name, view_idx)
  │   ├── explore_seed(seed_id)
  │   ├── explore_frontier(frontier_id)
  │   └── submit_answer
  └── 每个工具返回 TrajectoryEvidence
       │
       ▼ TrajectoryEvidence
       │
EvidenceNotebook (agent_notebook.py)
  ├── 转换为 NotebookEntry (5种类型)
  ├── 种子/frontier访问计数
  ├── 循环检测 (3次同实体 = exhausted)
  └── 注入历史到 Planner prompt
```

**循环控制:**

```
for round in 1..max_planner_rounds:
    1. 构建4组件prompt (history/scene/progress/actions)
    2. 附加图片: 当前3视角 + topdown地图
    3. Planner.decide() → PlannerAction
    4. 守卫子句:
       - 连续2次 explore_panorama → 替换为未访问种子/frontier
       - explore_seed 的seed已被访问 → 替换
       - navigate_to_object 的对象名无效 → 替换
    5. submit_answer → 返回结果
    6. Executor.execute_action() → TrajectoryEvidence
    7. Notebook.update_from_evidence()
    8. 实体耗尽检测 → 下轮强制换策略
    9. 步数预算检查

fallback: submit_best_guess (预算耗尽时)
```

**6个工具及对应证据条目类型:**

| 工具 | task_mode | NotebookEntry.entry_type | 关键证据 |
|------|-----------|--------------------------|----------|
| explore_panorama | explore_panorama | room_explored | 房间ID + 附近物体 |
| navigate_to_object | navigate_to_object | object_observed | 物体名 + GD质量 + 附近物体 |
| explore_seed | explore_seed | seed_visited | 种子ID + 到达房间ID |
| explore_frontier | explore_frontier | frontier_visited | Frontier ID + 到达房间ID |
| -- (GD失败) | -- | hypothesis_rejected | 失败原因 + 附近物体 |
| submit_answer | submit_answer | (直接返回结果，不进入notebook) | -- |

---

## 4. 计划与实际实现对照

### 4.1 2026-06-21 计划 `agent-workflow-design` / `hmge-implementation`

| 计划任务 | 实际实现文件 | 状态 |
|----------|-------------|------|
| 6阶段状态机 | `agent_workflow.py:run_episode()` | ✅ 已实现 (但后来被导航修复修改) |
| 7个VLM工具 | `agent_tools.py` | ✅ 7个全部实现 |
| 房间分割移植到tsdf_planner | `tsdf_planner.py` | ✅ (RoomRegion + 16个方法) |
| GD导航链移植到scene_aeqa | `scene_aeqa.py:grounded_navigate_to_object()` | ✅ |
| 图像工具模块 | `agent_image_utils.py` | ✅ |
| 记忆模块 | `agent_memory.py` | ✅ (CLIP+文本过滤，2次查询配额) |
| 上下文管理 | `agent_context.py` | ✅ (在HM-GE模式中使用) |
| 工作流控制器 | `agent_workflow.py:run_episode()` | ✅ |
| HM-GE评估脚本 | `run_hmge_evaluation.py` | ✅ |

### 4.2 2026-06-23 计划 `navigation-fix-design` / `navigation-fix-implementation`

| 计划任务 | 实际实现位置 | 状态 |
|----------|-------------|------|
| bresenham_2d | `geom.py:657` | ✅ |
| check_ray_blocked | `geom.py:688` | ✅ |
| GD 3D反投影坐标修复 | `scene_aeqa.py:1035-1169` (Phase A) | ✅ (pose_habitat_to_tsdf + z_forward) |
| GD 迭代螺旋搜索 | `scene_aeqa.py:1171+` (Phase B) | ✅ |
| 8视角全景(原7→8) | `agent_tools.py:observe_panorama()` | ✅ (前/右前/右/右后/后/左后/左/左前) |
| 3×3马赛克+指南针 | `agent_tools.py:observe_panorama()` | ✅ |
| SeedViewManager | `seed_views.py` | ✅ (注册/更新/马赛克/射线遮挡检测) |
| reason字段强制 | `agent_workflow.py:_parse_vlm_response()` | ✅ |
| 6阶段prompt重写 | `agent_workflow.py:STAGE1-6_PROMPT` | ✅ |
| 阶段prompt JSON schema | `agent_workflow.py:SCHEMA_REQUIREMENT` | ✅ |

### 4.3 2026-06-23 计划 `visualization-logging-design` / `visualization-logging-implementation`

| 计划任务 | 实际实现 | 状态 |
|----------|---------|------|
| RunLogger模块 (`src/run_logger.py`) | **未创建** | ❌ 计划未执行 |
| trace.jsonl | **未实现** | ❌ |
| 各阶段日志子目录 | **未实现** | ❌ |
| 可视化配置 | **未添加** | ❌ |

### 4.4 2026-06-24 计划 `two-tier-refactor-design` / `two-tier-refactor-implementation`

| 计划任务 | 实际实现位置 | 状态 |
|----------|-------------|------|
| EvidenceNotebook | `agent_notebook.py` | ✅ (NotebookEntry + EvidenceNotebook + 循环检测) |
| TrajectoryEvidence | `agent_evidence.py` | ✅ (5种entry_type转换) |
| GD质量过滤 | `scene_aeqa.py:gd_quality_filter()` | ✅ (仅score过滤; 计划中的bbox_ratio>30%过滤被移除——当agent接近目标时大bbox是有效的) |
| converge_dist_voxels 5→12 | `agent_tools.py:navigate_to_object()` | ✅ (实际值是12) |
| Euclidean距离<1.5m到达验证 | **未独立实现** | ⚠️ (被agent_step的内置收敛逻辑替代) |
| 8视角全景改进 (200→400, 标签, cam_pose, YOLO标注) | `agent_tools.py:observe_panorama()` | ✅ (分辨率提升到400, 有方向标签和cam_pose; YOLO标注未独立实现，但静默感知已做检测) |
| Planner模块 | `agent_planner.py` | ✅ (PlannerAction + Planner + 4组件prompt) |
| Executor模块 | `agent_executor.py` | ✅ (6工具 dispatch) |
| Two-tier工作流 (`run_episode_two_tier`) | `agent_workflow.py:1086-1549` | ✅ |
| 守卫子句(去重/访问检测) | `agent_workflow.py:1493-1503` | ✅ |
| Planner API配置 | `const.py:QWN_PLANNER_*` | ✅ |
| 多进程分片评估 | `run_two_tier_aeqa_evaluation.py` | ✅ |
| 跨分片结果聚合 | `run_two_tier_aeqa_evaluation.py:_aggregate_all_results()` | ✅ |

---

## 5. 模块详细对照

### 5.1 `src/agent_workflow.py` (1739行)

| 函数/常量 | 计划来源 | 用途 |
|-----------|---------|------|
| `SYSTEM_PROMPT` | nav-fix-design §3.2 | HM-GE 6阶段系统提示词 |
| `SCHEMA_REQUIREMENT` | nav-fix-design §3.2 | 共享JSON schema要求 |
| `STAGE1_PROMPT` ~ `STAGE6_PROMPT` | nav-fix-design §3.2 | 各阶段提示词模板 |
| `call_vlm()` | hmge-impl Task 8 | mimo-v2.5 API调用 (代理支持) |
| `run_episode()` | hmge-impl Task 8 + nav-fix Tasks 10-14 | 6阶段HM-GE工作流 |
| `run_episode_two_tier()` | two-tier-impl Tasks 5-6 | Planner-Executor循环 |
| `_parse_vlm_response()` | nav-fix-design §3.3 | reason字段强制解析 |
| `_is_valid_object_desc()` | two-tier-design §3 (guard) | 物体描述有效性检测 |
| `_register_new_seeds()` | nav-fix-design Phase C | 扫描room_regions注册种子 |
| `_select_frontier_with_vlm()` | two-tier-impl (扩展) | Stage 6.5 frontier VLM精选 |
| `_first_available_action()` | two-tier-impl (guard) | 守卫fallback动作生成 |

### 5.2 `src/agent_tools.py` (732行)

| 函数 | 计划来源 | 用途 |
|------|---------|------|
| `silent_perception_step()` | hmge-impl Task 6 + nav-fix Part B | 每步静默感知(3视角YOLO/SAM/CLIP/TSDF+快照) |
| `_navigate_to_target_with_agent_step()` | nav-fix § 循环导航 | agent_step循环+卡死检测+种子视图更新 |
| `observe_panorama()` | nav-fix-design Phase C + two-tier-design §3 | 8视角全景+TSDF融合+房间分割触发 |
| `view_direction()` | hmge-impl Task 6 | 方向观察(前/后/左/右) |
| `navigate_to_object()` | hmge-impl Task 5 + nav-fix Phase B | GD导航(视角指定+converge_dist_voxels=12) |
| `navigate_to_seed()` | hmge-impl Task 6 | 种子导航(临时Frontier构造) |
| `navigate_to_frontier()` | hmge-impl Task 6 | Frontier导航 |
| `query_memory()` | hmge-impl Task 5 | CLIP记忆查询+马赛克返回 |
| `submit_answer()` | hmge-impl Task 6 | 答案提交 |
| `build_planner_topdown_map_b64()` | two-tier-design §3 | Planner用精简topdown地图 |
| `save_topdown_step_visualization()` | (nav-fix扩展) | 导航步可视化保存 |
| `record_topdown_position()` | (nav-fix扩展) | 轨迹记录 |

### 5.3 `src/scene_aeqa.py` (1351行) — 导航相关

| 函数 | 计划来源 | 用途 |
|------|---------|------|
| `grounded_navigate_to_object()` | nav-fix-design Phase B | 完整GD导航链：Phase A(检测+反投影) + Phase B(螺旋搜索) |
| `gd_quality_filter()` | two-tier-design §4 | GD检测质量过滤(仅score; bbox尺寸不限) |
| `_gd_detect()` | nav-fix-implementation §B | 单图GroundingDINO检测 |
| 坐标修复 `pose_habitat_to_tsdf` + `z_forward` | nav-fix-design §2 | 3D反投影坐标系统修复 |

### 5.4 `src/tsdf_planner.py` (1632行)

| 功能 | 计划来源 | 用途 |
|------|---------|------|
| `RoomRegion` dataclass | hmge-impl Task 2 | 房间分割数据结构 |
| `Frontier` dataclass (扩展) | hmge-impl Task 2 | Frontier数据结构 |
| `SnapShot` dataclass (扩展) | hmge-impl Task 2 | 快照数据结构 |
| `update_room_map()` | hmge-impl Task 2 | 房间分割(dilation-watershed) |
| `update_frontier_map()` | hmge-impl Task 2 | Frontier提取与更新 |
| `agent_step()` | hmge-impl Task 2 + nav-fix | 增量导航步 |
| `set_next_navigation_point()` | hmge-impl Task 2 | 导航目标设定 |
| `get_room_id_at()` | hmge-impl Task 2 | 查询体素所在房间 |

### 5.5 `src/geom.py` (761行)

| 函数 | 计划来源 | 用途 |
|------|---------|------|
| `bresenham_2d()` | nav-fix-implementation Task 1 | 2D Bresenham直线追踪 |
| `check_ray_blocked()` | nav-fix-implementation Task 2 | 高度感知射线遮挡检测 (1.2m阈值) |

### 5.6 新建模块 (Two-Tier Refactor)

| 模块 | 行数 | 计划来源 | 核心职责 |
|------|------|---------|---------|
| `agent_planner.py` | 228 | two-tier-impl Task 4 | Planner: Qwen3.6-Plus API调用 + 4组件prompt + JSON/keyword解析 |
| `agent_executor.py` | 319 | two-tier-impl Task 4 | Executor: 6工具dispatch + 状态管理 + TrajectoryEvidence生成 |
| `agent_notebook.py` | 133 | two-tier-impl Task 1 | EvidenceNotebook: 5种条目类型 + 循环检测(3次=耗尽) |
| `agent_evidence.py` | 84 | two-tier-impl Task 1 | TrajectoryEvidence: 执行结果→笔记本条目映射 |
| `seed_views.py` | 127 | nav-fix-implementation Task 7 | SeedViewManager: 懒更新 + 射线遮挡 + 马赛克生成 |

---

## 6. 测试覆盖

| 测试文件 | 测试对象 | 计划来源 |
|----------|---------|---------|
| `test_geom.py` | `bresenham_2d` (5个用例) | nav-fix-implementation Task 1 |
| `test_geom.py` | `check_ray_blocked` (隐式) | nav-fix-implementation Task 2 |
| `test_vlm_parse.py` | `_parse_vlm_response` (8个用例) | nav-fix-implementation Task 8 |
| `test_planner.py` | `PlannerAction` 解析 (5个用例) | two-tier-impl Task 5 |
| `test_notebook.py` | `EvidenceNotebook` + `TrajectoryEvidence` | two-tier-impl Task 1 |
| `test_seed_views.py` | `SeedViewManager` (4个用例) | nav-fix-implementation Task 7 |
| `test_panorama.py` | 全景配置 (8视图/分辨率400/方向标签) | two-tier-impl Task 3 |
| `test_gd_filter.py` | `gd_quality_filter` | two-tier-impl Task 2 |
| `test_e2e_smoke.py` | 端到端模块导入检查 | 集成验证 |

---

## 7. 数据流总览

```
                    ┌──────────────────────────────┐
                    │    run_hmge_evaluation.py     │  (HM-GE模式)
                    │  run_two_tier_aeqa_eval.py    │  (Two-Tier模式)
                    └──────────┬───────────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │     agent_workflow.py         │
                    │  run_episode() /              │
                    │  run_episode_two_tier()       │
                    └──────────┬───────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │   Planner    │  │   Executor   │  │   Notebook   │
    │ (Two-Tier)   │  │              │  │  + Evidence  │
    └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
           │                 │                  │
           │         ┌───────┼───────┐          │
           │         ▼       ▼       ▼          │
           │  ┌─────────┐ ┌──────┐ ┌────────┐   │
           │  │ tools   │ │scene │ │ tsdf   │   │
           │  │(7工具)  │ │_aeqa │ │_planner│   │
           │  └────┬────┘ └──┬───┘ └───┬────┘   │
           │       │         │         │         │
           │       ▼         ▼         ▼         │
           │  ┌─────────────────────────────┐    │
           │  │       Habitat-sim           │    │
           │  │   (3D场景模拟 + 渲染)       │    │
           │  └─────────────────────────────┘    │
           │                                     │
           │  ┌─────────────────────────────┐    │
           │  │       MemoryStore            │◄───┤
           │  │   (CLIP快照存储与检索)       │    │
           │  └─────────────────────────────┘    │
           │                                     │
           └────────── call_vlm() ───────────────┘
                      (mimo-v2.5 API)
```

---

## 8. 关键设计决策与偏差

| 设计 | 计划 | 实际 | 原因 |
|------|------|------|------|
| GD bbox尺寸过滤 | 拒绝 >30% 图像面积 | 移除该限制 | agent接近目标时大bbox是有效的，拒绝导致重复失败 |
| Planner模型 | Qwen3.6-Plus | mimo-v2.5 | API环境实际可用模型，Qwen常量和环境变量已配置 |
| 视觉日志模块 | `src/run_logger.py` | 未创建 | 计划2（visualization-logging）整体未执行 |
| 全景中YOLO对象标注 | 每视图附加YOLO列表 | 未单独实现 | 静默感知已做检测并存入scene.objects，在构建prompt时通过collect_nearby可用 |
| Euclidean到达验证 | 独立 <1.5m 检查 | agent_step内置收敛 | TSDFPlanner.agent_step已处理目标到达检测 |
| converged_dist_voxels | 5→12 (计划) | 12 (实际) | 与计划一致 |
