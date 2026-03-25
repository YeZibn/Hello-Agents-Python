import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from synth_agent.flow.task import Task, TaskPlan, TaskStatus, AgentInput, AgentOutput
from synth_agent.flow.role import get_role, RoleType
from synth_agent.flow.task_persistence import TaskPersistence
from synth_agent.agent.react_agent import ReActAgent
from synth_agent.llm.synth_LLM import SynthLLM
from synth_agent.tool.tool_registry import ToolRegistry


class RetryPolicy:
    def __init__(
        self,
        max_attempts: int = 3,
        backoff_base: float = 1.0,
        backoff_multiplier: float = 2.0,
        max_backoff: float = 30.0
    ):
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.backoff_multiplier = backoff_multiplier
        self.max_backoff = max_backoff

    def get_delay(self, attempt: int) -> float:
        delay = self.backoff_base * (self.backoff_multiplier ** (attempt - 1))
        return min(delay, self.max_backoff)


class TaskScheduler:
    def __init__(
        self,
        llm: SynthLLM,
        tool_registry: ToolRegistry,
        retry_policy: Optional[RetryPolicy] = None,
        max_concurrent: int = 5,
        persistence: Optional[TaskPersistence] = None
    ):
        self.llm = llm
        self.tool_registry = tool_registry
        self.retry_policy = retry_policy or RetryPolicy()
        self.max_concurrent = max_concurrent
        self.persistence = persistence or TaskPersistence()
        self.agents: Dict[str, ReActAgent] = {}

    async def execute_plan(self, plan: TaskPlan) -> Dict[str, Any]:
        print(f"\n📋 开始执行计划: {plan.plan_id}")
        print(f"🎯 目标: {plan.goal}")
        print(f"📝 任务数量: {len(plan.tasks)}")

        results: Dict[str, AgentOutput] = {}
        completed_task_ids: List[str] = []
        failed_task_ids: List[str] = []

        semaphore = asyncio.Semaphore(self.max_concurrent)

        while not plan.all_tasks_completed():
            ready_tasks = plan.get_ready_tasks(completed_task_ids)

            if not ready_tasks:
                blocked_tasks = [t for t in plan.tasks if t.status == TaskStatus.PENDING]
                if blocked_tasks:
                    for task in blocked_tasks:
                        deps_failed = [d for d in task.depends_on if d in failed_task_ids]
                        if deps_failed:
                            task.mark_skipped(f"依赖任务失败: {deps_failed}")
                            self.persistence.update_task_status(plan.plan_id, task)
                            print(f"⏭️ 任务 {task.task_id} 被跳过，依赖任务失败")

                if not any(t.status == TaskStatus.RUNNING for t in plan.tasks):
                    break

                await asyncio.sleep(0.5)
                continue

            async def run_with_semaphore(task: Task):
                async with semaphore:
                    return await self._execute_task_with_retry(task, plan, results)

            coroutines = [run_with_semaphore(task) for task in ready_tasks]
            task_results = await asyncio.gather(*coroutines, return_exceptions=True)

            for task, result in zip(ready_tasks, task_results):
                if isinstance(result, Exception):
                    task.mark_failed(str(result))
                    self.persistence.update_task_status(plan.plan_id, task)
                    failed_task_ids.append(task.task_id)
                    print(f"❌ 任务 {task.task_id} 执行异常: {result}")
                elif result.status == "completed":
                    completed_task_ids.append(task.task_id)
                    results[task.task_id] = result
                    self.persistence.update_task_status(plan.plan_id, task)
                    print(f"✅ 任务 {task.task_id} 完成")
                else:
                    failed_task_ids.append(task.task_id)
                    self.persistence.update_task_status(plan.plan_id, task)
                    print(f"❌ 任务 {task.task_id} 失败: {result.error_message}")

        return self._build_final_result(plan, results)

    async def _execute_task_with_retry(
        self,
        task: Task,
        plan: TaskPlan,
        results: Dict[str, AgentOutput]
    ) -> AgentOutput:
        task.mark_running()
        self.persistence.update_task_status(plan.plan_id, task)

        while task.attempt <= task.max_attempts:
            try:
                agent_input = self._build_agent_input(task, plan, results)
                agent = self._get_or_create_agent(task.role)

                output = await self._run_agent_async(agent, agent_input)

                if output.status == "completed" and not output.needs_retry:
                    task.mark_completed(output.output, output.artifacts)
                    return output

                if task.attempt < task.max_attempts:
                    delay = self.retry_policy.get_delay(task.attempt)
                    print(f"🔄 任务 {task.task_id} 将在 {delay:.1f}秒后重试 (第{task.attempt + 1}次)")
                    await asyncio.sleep(delay)
                    task.attempt += 1
                    self.persistence.update_task_status(plan.plan_id, task)
                else:
                    task.mark_failed(output.error_message or "超过最大重试次数")
                    return output

            except Exception as e:
                if task.attempt < task.max_attempts:
                    delay = self.retry_policy.get_delay(task.attempt)
                    print(f"🔄 任务 {task.task_id} 异常，将在 {delay:.1f}秒后重试")
                    await asyncio.sleep(delay)
                    task.attempt += 1
                    self.persistence.update_task_status(plan.plan_id, task)
                else:
                    task.mark_failed(str(e))
                    return AgentOutput(
                        status="failed",
                        output="",
                        error_message=str(e)
                    )

        return AgentOutput(
            status="failed",
            output="",
            error_message="超过最大重试次数"
        )

    def _build_agent_input(
        self,
        task: Task,
        plan: TaskPlan,
        results: Dict[str, AgentOutput]
    ) -> AgentInput:
        dependencies_result = []
        for dep_id in task.depends_on:
            if dep_id in results:
                dep_output = results[dep_id]
                dependencies_result.append({
                    "task_id": dep_id,
                    "status": dep_output.status,
                    "output": dep_output.output,
                    "artifacts": dep_output.artifacts
                })

        retry_info = {
            "attempt": task.attempt,
            "max_attempts": task.max_attempts,
            "previous_error": task.error_message
        }

        context = {
            "original_goal": plan.goal,
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return AgentInput(
            task=task,
            context=context,
            dependencies_result=dependencies_result,
            retry_info=retry_info
        )

    def _get_or_create_agent(self, role_type: RoleType) -> ReActAgent:
        if role_type.value not in self.agents:
            role = get_role(role_type)

            filtered_registry = self._create_filtered_tool_registry(role.available_tools)

            agent = ReActAgent(
                name=f"{role.name}_{role_type.value}",
                llm=self.llm,
                tool_registry=filtered_registry,
                system_prompt=role.get_full_prompt(),
                max_steps=20
            )

            self.agents[role_type.value] = agent

        return self.agents[role_type.value]

    def _create_filtered_tool_registry(self, tool_names: List[str]) -> ToolRegistry:
        filtered_registry = ToolRegistry()

        for tool_name in tool_names:
            tool = self.tool_registry.get_tool(tool_name)
            if tool:
                filtered_registry.register_tool(tool)

        return filtered_registry

    async def _run_agent_async(self, agent: ReActAgent, agent_input: AgentInput) -> AgentOutput:
        loop = asyncio.get_event_loop()
        prompt = agent_input.to_prompt()

        result = await loop.run_in_executor(
            None,
            lambda: agent.run(prompt)
        )

        return AgentOutput(
            status="completed",
            output=result,
            artifacts={},
            confidence=1.0
        )

    def _build_final_result(
        self,
        plan: TaskPlan,
        results: Dict[str, AgentOutput]
    ) -> Dict[str, Any]:
        completed_tasks = [t for t in plan.tasks if t.status == TaskStatus.COMPLETED]
        failed_tasks = [t for t in plan.tasks if t.status == TaskStatus.FAILED]
        skipped_tasks = [t for t in plan.tasks if t.status == TaskStatus.SKIPPED]

        return {
            "plan_id": plan.plan_id,
            "goal": plan.goal,
            "status": "completed" if not failed_tasks else "partial",
            "summary": {
                "total": len(plan.tasks),
                "completed": len(completed_tasks),
                "failed": len(failed_tasks),
                "skipped": len(skipped_tasks)
            },
            "tasks_result": {
                task_id: {
                    "status": output.status,
                    "output": output.output,
                    "artifacts": output.artifacts
                }
                for task_id, output in results.items()
            },
            "failed_tasks": [
                {
                    "task_id": t.task_id,
                    "description": t.description,
                    "error": t.error_message
                }
                for t in failed_tasks
            ]
        }
