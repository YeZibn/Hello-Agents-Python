import asyncio
from typing import Optional, Dict, Any
from synth_agent.llm.synth_LLM import SynthLLM
from synth_agent.tool.tool_registry import ToolRegistry
from synth_agent.flow.planner import TaskPlanner
from synth_agent.flow.scheduler import TaskScheduler, RetryPolicy
from synth_agent.flow.task import TaskPlan
from synth_agent.flow.task_persistence import TaskPersistence


class PlanFlow:
    def __init__(
        self,
        llm: SynthLLM,
        tool_registry: ToolRegistry,
        max_tasks: int = 10,
        max_concurrent: int = 5,
        retry_policy: Optional[RetryPolicy] = None,
        persistence: Optional[TaskPersistence] = None
    ):
        self.llm = llm
        self.tool_registry = tool_registry
        self.max_tasks = max_tasks
        self.max_concurrent = max_concurrent
        self.retry_policy = retry_policy or RetryPolicy(
            max_attempts=3,
            backoff_base=1.0,
            backoff_multiplier=2.0
        )
        self.persistence = persistence or TaskPersistence()

        self.planner = TaskPlanner(llm)
        self.scheduler = TaskScheduler(
            llm=llm,
            tool_registry=tool_registry,
            retry_policy=self.retry_policy,
            max_concurrent=self.max_concurrent,
            persistence=self.persistence
        )

        self.current_plan: Optional[TaskPlan] = None

    def run(self, goal: str) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"🚀 PlanFlow 启动")
        print(f"{'='*60}")

        print(f"\n📌 阶段1: 任务分解")
        print("-" * 40)
        self.current_plan = self.planner.plan(goal, self.max_tasks)

        print(f"\n📋 任务计划:")
        for task in self.current_plan.tasks:
            deps = f" (依赖: {task.depends_on})" if task.depends_on else ""
            print(f"  - {task.task_id}: [{task.role.value}] {task.description}{deps}")

        plan_dir = self.persistence.save_plan(self.current_plan)
        print(f"\n💾 任务计划已保存至: {plan_dir}")

        print(f"\n📌 阶段2: 任务执行")
        print("-" * 40)
        result = asyncio.run(self.scheduler.execute_plan(self.current_plan))

        print(f"\n📌 阶段3: 结果汇总")
        print("-" * 40)
        final_result = self._summarize_result(result)

        print(f"\n{'='*60}")
        print(f"✅ PlanFlow 完成")
        print(f"{'='*60}")

        return final_result

    def _summarize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        summary = result.get("summary", {})

        print(f"📊 执行统计:")
        print(f"  - 总任务数: {summary.get('total', 0)}")
        print(f"  - 完成数: {summary.get('completed', 0)}")
        print(f"  - 失败数: {summary.get('failed', 0)}")
        print(f"  - 跳过数: {summary.get('skipped', 0)}")

        if result.get("failed_tasks"):
            print(f"\n⚠️ 失败任务:")
            for task in result["failed_tasks"]:
                print(f"  - {task['task_id']}: {task['error']}")

        return result

    def get_plan(self) -> Optional[TaskPlan]:
        return self.current_plan

    def load_plan(self, plan_id: str) -> Optional[TaskPlan]:
        return self.persistence.load_plan(plan_id)

    def get_all_plans(self) -> list:
        return self.persistence.get_all_plans()

    def visualize_plan(self) -> str:
        if not self.current_plan:
            return "暂无任务计划"

        lines = ["\n📊 任务依赖图 (DAG):", "=" * 40]

        for task in self.current_plan.tasks:
            indent = "  " * len(task.depends_on) if task.depends_on else ""
            deps_str = f" ← [{', '.join(task.depends_on)}]" if task.depends_on else ""
            status_icon = {"completed": "✅", "failed": "❌", "running": "🔄", "pending": "⏳", "skipped": "⏭️"}.get(task.status.value, "❓")
            lines.append(f"{indent}└─ {status_icon} {task.task_id} ({task.role.value}){deps_str}")

        return "\n".join(lines)


def create_plan_flow(
    model: str,
    api_key: str,
    base_url: str = None,
    tools: list = None
) -> PlanFlow:
    from synth_agent.tool.tool_list.bash_tool import BashTool
    from synth_agent.tool.tool_list.read_tool import ReadTool
    from synth_agent.tool.tool_list.write_tool import WriteTool
    from synth_agent.tool.tool_list.web.baidu_search_tool import BaiduSearchTool
    from synth_agent.tool.tool_list.web.url_search_tool import UrlSearchTool

    llm = SynthLLM(model=model, api_key=api_key, base_url=base_url)

    tool_registry = ToolRegistry()

    default_tools = [BashTool(), ReadTool(), WriteTool(), BaiduSearchTool(), UrlSearchTool()]
    for tool in (tools or default_tools):
        tool_registry.register_tool(tool)

    return PlanFlow(llm=llm, tool_registry=tool_registry)
