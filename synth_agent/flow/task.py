from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from synth_agent.flow.role import RoleType


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Task(BaseModel):
    task_id: str
    description: str
    role: RoleType
    depends_on: List[str] = Field(default_factory=list)
    expected_output: str = ""
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    attempt: int = 0
    max_attempts: int = 3
    error_message: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    def is_ready(self, completed_task_ids: List[str]) -> bool:
        return all(dep_id in completed_task_ids for dep_id in self.depends_on)

    def can_retry(self) -> bool:
        return self.attempt < self.max_attempts and self.status == TaskStatus.FAILED

    def mark_running(self):
        self.status = TaskStatus.RUNNING
        self.attempt += 1

    def mark_completed(self, result: str, artifacts: Dict[str, Any] = None):
        self.status = TaskStatus.COMPLETED
        self.result = result
        if artifacts:
            self.artifacts = artifacts
        self.completed_at = datetime.now().isoformat()

    def mark_failed(self, error: str):
        self.status = TaskStatus.FAILED
        self.error_message = error

    def mark_skipped(self, reason: str):
        self.status = TaskStatus.SKIPPED
        self.error_message = reason


class TaskPlan(BaseModel):
    plan_id: str
    goal: str
    tasks: List[Task]
    final_task: Optional[Task] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    def get_task(self, task_id: str) -> Optional[Task]:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        if self.final_task and self.final_task.task_id == task_id:
            return self.final_task
        return None

    def get_pending_tasks(self) -> List[Task]:
        return [t for t in self.tasks if t.status == TaskStatus.PENDING]

    def get_ready_tasks(self, completed_task_ids: List[str]) -> List[Task]:
        ready = []
        for task in self.tasks:
            if task.status == TaskStatus.PENDING and task.is_ready(completed_task_ids):
                ready.append(task)
        return ready

    def get_completed_task_ids(self) -> List[str]:
        completed = [t.task_id for t in self.tasks if t.status == TaskStatus.COMPLETED]
        return completed

    def all_tasks_completed(self) -> bool:
        return all(t.status in [TaskStatus.COMPLETED, TaskStatus.SKIPPED] for t in self.tasks)


class AgentInput(BaseModel):
    task: Task
    context: Dict[str, Any] = Field(default_factory=dict)
    dependencies_result: List[Dict[str, Any]] = Field(default_factory=list)
    retry_info: Dict[str, Any] = Field(default_factory=dict)

    def to_prompt(self) -> str:
        prompt_parts = [f"## 任务描述\n{self.task.description}"]

        if self.context.get("original_goal"):
            prompt_parts.append(f"\n## 总体目标\n{self.context['original_goal']}")

        if self.context.get("current_time"):
            prompt_parts.append(f"\n## 当前时间\n{self.context['current_time']}")

        if self.dependencies_result:
            prompt_parts.append("\n## 前置任务结果")
            for dep_result in self.dependencies_result:
                prompt_parts.append(f"\n### {dep_result['task_id']}")
                prompt_parts.append(f"状态: {dep_result['status']}")
                if dep_result.get("output"):
                    prompt_parts.append(f"结果:\n{dep_result['output']}")

        if self.task.expected_output:
            prompt_parts.append(f"\n## 期望输出\n{self.task.expected_output}")

        if self.retry_info.get("attempt", 1) > 1:
            prompt_parts.append(f"\n## 重试信息")
            prompt_parts.append(f"当前尝试: 第{self.retry_info['attempt']}次")
            if self.retry_info.get("previous_error"):
                prompt_parts.append(f"上次错误: {self.retry_info['previous_error']}")

        return "\n".join(prompt_parts)


class AgentOutput(BaseModel):
    status: str
    output: str
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    needs_retry: bool = False
    error_message: Optional[str] = None
