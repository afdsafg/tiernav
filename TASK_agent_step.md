调查 tsdf_planner.py 中 agent_step 函数产生 "Warning: next point is the same as the current point when determining the direction" 的根本原因。

工作目录: /home/afdsafg/下载/new/3D-Mem/.worktrees/refactor-two-tier

请读取: src/tsdf_planner.py:611-931 (agent_step 函数)

重点调查：
1. 什么条件下 max_point/target_point 被设置成与 agent 当前位置相同？
2. 当 target_point 与当前位置相同时，是否有 fallback 逻辑？
3. 这个 warning 出现后会怎样？agent 会强制移动还是原地踏步？
4. 查看 max_point 和 target_point 是如何被 set_next_navigation_point 设置的（line 459-609）

输出：
- 导致 same-point 的代码路径
- 具体行号和逻辑
- 修复建议
