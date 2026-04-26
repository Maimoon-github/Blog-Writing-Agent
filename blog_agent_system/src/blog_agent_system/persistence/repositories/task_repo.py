# src/blog_agent_system/persistence/repositories/task_repo.py
from sqlalchemy.orm import Session
from blog_agent_system.persistence.models import Task
from typing import Optional

class TaskRepository:
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, task: Task) -> Task:
        self.db.add(task)
        self.db.commit()
        self.db.refresh(task)
        return task
    
    def get_by_thread(self, thread_id: str) -> Optional[Task]:
        return self.db.query(Task).filter(Task.thread_id == thread_id).first()
    
    def update_status(self, task_id: uuid.UUID, status: str):
        task = self.db.query(Task).filter(Task.id == task_id).first()
        task.status = status
        if status == "complete":
            task.completed_at = func.now()
        self.db.commit()