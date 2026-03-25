from typing import List, Dict, Any, Optional
from synth_agent.llm.synth_LLM import SynthLLM
from synth_agent.flow.role import RoleType, get_all_roles_description
from synth_agent.flow.task import Task, TaskPlan
from synth_agent.agent.react_agent import ReActAgent
from synth_agent.tool.tool_registry import ToolRegistry
import json
from datetime import datetime
import uuid


PLANNER_SYSTEM_PROMPT = """你是一个任务规划专家，负责将复杂任务分解为可执行的子任务。

## 可用角色
{roles_description}

## 任务分解原则
1. 每个子任务应该清晰、独立、可执行
2. 合理设置任务间的依赖关系
3. 为每个任务分配合适的角色
4. 任务粒度适中，不要过细或过粗
5. 确保任务链条完整，能够达成最终目标

## 输出格式
请严格按照以下JSON格式输出，不要包含其他内容：
```json
{{
  "tasks": [
    {{
      "task_id": "task_1",
      "description": "任务描述",
      "role": "researcher",
      "depends_on": [],
      "expected_output": "期望输出描述"
    }}
  ]
}}
```

## 注意事项
- task_id 按顺序编号：task_1, task_2, ...
- role 必须是可用角色之一
- depends_on 是依赖的任务ID列表，无依赖则为空列表
- 最后一个任务通常由 manager 角色负责总结
- 直接输出JSON，不要有额外的解释
"""


class TaskPlanner:
    def __init__(self, llm: SynthLLM, tool_registry: Optional[ToolRegistry] = None):
        self.llm = llm
        self.tool_registry = tool_registry or ToolRegistry()
        self.planner_agent = self._create_planner_agent()

    def _create_planner_agent(self) -> ReActAgent:
        system_prompt = PLANNER_SYSTEM_PROMPT.format(
            roles_description=get_all_roles_description()
        )

        return ReActAgent(
            name="TaskPlanner",
            llm=self.llm,
            tool_registry=self.tool_registry,
            system_prompt=system_prompt,
            max_steps=5
        )

    def plan(self, goal: str, max_tasks: int = 10) -> TaskPlan:
        plan_id = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        prompt = f"""请将以下任务分解为子任务，直接输出JSON格式：

目标：{goal}

要求：
1. 分解为 {max_tasks} 个以内的子任务
2. 每个任务指定合适的角色
3. 设置合理的依赖关系
4. 直接输出JSON，不要有其他内容"""

        response = self.planner_agent.run(prompt)

        tasks = self._parse_response(response, max_tasks)

        plan = TaskPlan(
            plan_id=plan_id,
            goal=goal,
            tasks=tasks
        )

        self._validate_plan(plan)

        return plan

    def _parse_response(self, response, max_tasks: int) -> List[Task]:
        try:
            if isinstance(response, dict):
                response_text = response.get("content", "") or response.get("response", "")
                if not response_text:
                    response_text = str(response)
            else:
                response_text = str(response) if response else ""

            json_str = self._extract_json(response_text)
            data = json.loads(json_str)

            tasks = []
            for i, task_data in enumerate(data.get("tasks", [])[:max_tasks]):
                task = Task(
                    task_id=task_data.get("task_id", f"task_{i+1}"),
                    description=task_data.get("description", ""),
                    role=RoleType(task_data.get("role", "researcher")),
                    depends_on=task_data.get("depends_on", []),
                    expected_output=task_data.get("expected_output", "")
                )
                tasks.append(task)

            return tasks

        except json.JSONDecodeError as e:
            print(f"⚠️ JSON解析失败: {e}")
            return self._create_fallback_tasks(response)
        except Exception as e:
            print(f"⚠️ 任务解析失败: {e}")
            return self._create_fallback_tasks(response)

    def _extract_json(self, text: str) -> str:
        import re

        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
        if json_match:
            return json_match.group(1)

        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            return json_match.group(0)

        return text

    def _create_fallback_tasks(self, response) -> List[Task]:
        if isinstance(response, dict):
            response_text = response.get("content", "") or response.get("response", "")
        else:
            response_text = str(response) if response else ""

        return [
            Task(
                task_id="task_1",
                description=response_text[:200] if response_text else "处理用户请求",
                role=RoleType.MANAGER,
                depends_on=[],
                expected_output="完整的处理结果"
            )
        ]

    def _validate_plan(self, plan: TaskPlan):
        task_ids = {t.task_id for t in plan.tasks}

        for task in plan.tasks:
            for dep_id in task.depends_on:
                if dep_id not in task_ids:
                    print(f"⚠️ 警告: 任务 {task.task_id} 依赖不存在的任务 {dep_id}")

        if self._has_circular_dependency(plan):
            raise ValueError("检测到循环依赖，任务计划无效")

    def _has_circular_dependency(self, plan: TaskPlan) -> bool:
        task_dict = {t.task_id: t for t in plan.tasks}
        visited = set()
        rec_stack = set()

        def has_cycle(task_id: str) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)

            task = task_dict.get(task_id)
            if task:
                for dep_id in task.depends_on:
                    if dep_id not in visited:
                        if has_cycle(dep_id):
                            return True
                    elif dep_id in rec_stack:
                        return True

            rec_stack.remove(task_id)
            return False

        for task in plan.tasks:
            if task.task_id not in visited:
                if has_cycle(task.task_id):
                    return True

        return False
