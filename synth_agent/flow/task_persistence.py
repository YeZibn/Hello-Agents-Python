import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from synth_agent.flow.task import TaskPlan, Task, TaskStatus


class TaskPersistence:
    def __init__(self, base_path: str = None):
        if base_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            base_path = os.path.join(project_root, "task")
        self.base_path = base_path
        self._ensure_dir()

    def _ensure_dir(self):
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

    def save_plan(self, plan: TaskPlan) -> str:
        plan_dir = os.path.join(self.base_path, plan.plan_id)
        if not os.path.exists(plan_dir):
            os.makedirs(plan_dir)

        plan_file = os.path.join(plan_dir, "plan.json")
        plan_data = {
            "plan_id": plan.plan_id,
            "goal": plan.goal,
            "created_at": plan.created_at,
            "tasks": [self._task_to_dict(t) for t in plan.tasks]
        }

        with open(plan_file, "w", encoding="utf-8") as f:
            json.dump(plan_data, f, ensure_ascii=False, indent=2)

        self._save_dependency_graph(plan, plan_dir)

        return plan_dir

    def _task_to_dict(self, task: Task) -> Dict[str, Any]:
        return {
            "task_id": task.task_id,
            "description": task.description,
            "role": task.role.value,
            "depends_on": task.depends_on,
            "expected_output": task.expected_output,
            "status": task.status.value,
            "result": task.result,
            "artifacts": task.artifacts,
            "attempt": task.attempt,
            "max_attempts": task.max_attempts,
            "error_message": task.error_message,
            "created_at": task.created_at,
            "completed_at": task.completed_at
        }

    def _save_dependency_graph(self, plan: TaskPlan, plan_dir: str):
        graph_file = os.path.join(plan_dir, "dependency_graph.json")
        
        nodes = []
        edges = []
        
        for task in plan.tasks:
            nodes.append({
                "id": task.task_id,
                "label": task.description[:50],
                "role": task.role.value,
                "status": task.status.value
            })
            
            for dep_id in task.depends_on:
                edges.append({
                    "from": dep_id,
                    "to": task.task_id
                })

        graph_data = {
            "nodes": nodes,
            "edges": edges
        }

        with open(graph_file, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

    def update_task_status(self, plan_id: str, task: Task):
        plan_dir = os.path.join(self.base_path, plan_id)
        status_file = os.path.join(plan_dir, f"{task.task_id}_status.json")

        status_data = {
            "task_id": task.task_id,
            "status": task.status.value,
            "attempt": task.attempt,
            "result": task.result,
            "error_message": task.error_message,
            "updated_at": datetime.now().isoformat(),
            "completed_at": task.completed_at
        }

        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(status_data, f, ensure_ascii=False, indent=2)

        self._update_plan_summary(plan_id, task)

    def _update_plan_summary(self, plan_id: str, updated_task: Task):
        plan_dir = os.path.join(self.base_path, plan_id)
        summary_file = os.path.join(plan_dir, "summary.json")

        plan_file = os.path.join(plan_dir, "plan.json")
        if os.path.exists(plan_file):
            with open(plan_file, "r", encoding="utf-8") as f:
                plan_data = json.load(f)

            for i, task_data in enumerate(plan_data["tasks"]):
                if task_data["task_id"] == updated_task.task_id:
                    plan_data["tasks"][i] = self._task_to_dict(updated_task)
                    break

            with open(plan_file, "w", encoding="utf-8") as f:
                json.dump(plan_data, f, ensure_ascii=False, indent=2)

        status_counts = {
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "skipped": 0
        }

        if os.path.exists(plan_file):
            with open(plan_file, "r", encoding="utf-8") as f:
                plan_data = json.load(f)
            
            for task_data in plan_data["tasks"]:
                status = task_data.get("status", "pending")
                if status in status_counts:
                    status_counts[status] += 1

        summary_data = {
            "plan_id": plan_id,
            "updated_at": datetime.now().isoformat(),
            "status_counts": status_counts,
            "total_tasks": sum(status_counts.values()),
            "is_completed": status_counts["pending"] == 0 and status_counts["running"] == 0
        }

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)

    def load_plan(self, plan_id: str) -> Optional[TaskPlan]:
        plan_dir = os.path.join(self.base_path, plan_id)
        plan_file = os.path.join(plan_dir, "plan.json")

        if not os.path.exists(plan_file):
            return None

        with open(plan_file, "r", encoding="utf-8") as f:
            plan_data = json.load(f)

        from synth_agent.flow.role import RoleType
        
        tasks = []
        for task_data in plan_data["tasks"]:
            task = Task(
                task_id=task_data["task_id"],
                description=task_data["description"],
                role=RoleType(task_data["role"]),
                depends_on=task_data["depends_on"],
                expected_output=task_data.get("expected_output", ""),
                status=TaskStatus(task_data.get("status", "pending")),
                result=task_data.get("result"),
                artifacts=task_data.get("artifacts", {}),
                attempt=task_data.get("attempt", 0),
                max_attempts=task_data.get("max_attempts", 3),
                error_message=task_data.get("error_message"),
                created_at=task_data.get("created_at", ""),
                completed_at=task_data.get("completed_at")
            )
            tasks.append(task)

        return TaskPlan(
            plan_id=plan_data["plan_id"],
            goal=plan_data["goal"],
            tasks=tasks,
            created_at=plan_data.get("created_at", "")
        )

    def get_all_plans(self) -> List[Dict[str, Any]]:
        plans = []
        
        if not os.path.exists(self.base_path):
            return plans

        for plan_id in os.listdir(self.base_path):
            plan_dir = os.path.join(self.base_path, plan_id)
            if os.path.isdir(plan_dir):
                summary_file = os.path.join(plan_dir, "summary.json")
                if os.path.exists(summary_file):
                    with open(summary_file, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                    plans.append(summary)

        return sorted(plans, key=lambda x: x.get("updated_at", ""), reverse=True)
