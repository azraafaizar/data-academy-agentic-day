import gradio as gr
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
from sample_tasks import TASK_DATA
from difflib import get_close_matches
from dateutil.parser import parse
import uuid

# Mock database models
class Team(BaseModel):
    team_id: str
    tasks: List[Dict]
    status: str
    deadline: datetime

class Reminder(BaseModel):
    reminder_id: str
    reminder_time: datetime
    name: str = "Unnamed Reminder"
    status: str = "pending"


# Mock team database
TEAMS_DB = {
    "TEAM-001": Team(
        team_id="TM-001",
        tasks=[{"task_name": "Team Meeting","priority": "Medium","days": 5}],
        status="pending",
        deadline=datetime.now() - timedelta(days=5)
    ),
    "TEAM-002": Team(
        team_id="TM-002",
        tasks=[{"task_name": "Complete tickets on hold","priority": "Low","days": 2}],
        status="pending",
        deadline=datetime.now() - timedelta(days=2)
    )
}

class ProductKnowledgeBase:
    def __init__(self, tasks: List[Dict]):
        self.tasks = tasks

    def search_tasks(self, query: str = "", status: Optional[str] = None,priority: Optional[str] = None,due_before: Optional[datetime] = None,due_after: Optional[datetime] = None,department: Optional[str] = None) -> List[Dict]:
        """Filter and search tasks with several criteria."""
        results = []

        for task in self.tasks:
            if department and task.get("department", "").lower() != department.lower():
                continue
            if status and task.get("status", "").lower() != status.lower():
                continue
            if priority and task.get("priority", "").lower() != priority.lower():
                continue

            due_date_str = task.get("due_date")
            due_date = parse(due_date_str) if due_date_str else None

            if due_before and due_date and due_date >= due_before:
                continue
            if due_after and due_date and due_date <= due_after:
                continue

            if query:
                name = task.get("task", "").lower()
                description = task.get("description", "").lower()
                match_found = any(
                    get_close_matches(query.lower(), [name, description], n=1, cutoff=0.7)
                )
                if not match_found:
                    continue

            results.append(task)
            if len(results) >= 5:
                break

        return results

    def get_task_by_id(self, task_id: str) -> Optional[Dict]:
        """Retrieve a task by its ID"""
        for task in self.tasks:
            if task['id'] == task_id:
                return task
        return None

kb = ProductKnowledgeBase(TASK_DATA)

REMINDERS_DB = {}

def filter_tasks_by_department(department: str, status: Optional[str] = None,priority: Optional[str] = None) -> List[Dict]:
        """Filter tasks based on department, status, and priority"""
        return kb.search_tasks(query="", department=department, status=status, priority=priority)

def check_task(task_id: str) -> Optional[Dict]:
        """Return task details by ID, or None if not found."""
        task = kb.get_task_by_id(task_id)
        if not task:
            return None
        return task

def update_description(task_id: str, new_description: str) -> bool:
    """Update the description of a task."""
    task = kb.get_task_by_id(task_id)
    if not task:
        return False
    task['description'] = new_description
    return True

def update_priority(task_id: str, new_priority: str) -> bool:
    """Update the priority of a task."""
    valid_priorities = {'low', 'medium', 'high'}
    if new_priority.lower() not in valid_priorities:
        return False
    task = kb.get_task_by_id(task_id)
    if not task:
        return False
    task['priority'] = new_priority.capitalize()
    return True

def update_status(task_id: str, new_status: str) -> bool:
    """Update the status of a task."""
    valid_statuses = {'pending', 'in progress', 'completed', 'on hold'}
    if new_status.lower() not in valid_statuses:
        return False
    task = kb.get_task_by_id(task_id)
    if not task:
        return False
    task['status'] = new_status.lower()
    return True

def delete_task(task_id: str) -> bool:
    """Delete a task by ID."""
    for i, task in enumerate(kb.tasks):
        if task['id'] == task_id:
            kb.tasks.pop(i)
            return True
    return False

def add_new_task(task_id: str, task_name: str, description: str, priority: str = "Medium",status: str = "pending",) -> bool:
    """Add a new task, returns False if task_id already exists or priority/status invalid."""
    if kb.get_task_by_id(task_id):
        return False  # Task ID already exists

    valid_priorities = {'low', 'medium', 'high'}
    valid_statuses = {'pending', 'in progress', 'completed', 'on hold'}
    if priority.lower() not in valid_priorities or status.lower() not in valid_statuses:
        return False

    new_task = {
        "id": task_id,
        "task": task_name,
        "description": description,
        "priority": priority.capitalize(),
        "status": status.lower(),
        "pending": status.lower() == "pending",  # maintain backward compatibility with your sample
    }
    kb.tasks.append(new_task)
    return True

def add_reminder(reminder_time_input: str, name: Optional[str] = None, status: str = "pending") -> str:
    """Add a new reminder and return its ID. Accepts different date inputs."""
    try:
        reminder_time = parse(reminder_time_input)
    except ValueError:
        raise ValueError("Could not parse the date and time. Please use a recognisable format.")

    reminder_id = str(uuid.uuid4())
    REMINDERS_DB[reminder_id] = Reminder(
        reminder_id=reminder_id,
        reminder_time=reminder_time,
        name=name or f"Reminder on {reminder_time.isoformat()}",
        status=status,
    )
    return reminder_id

def get_reminders(status: Optional[str] = None) -> List[str]:
    """Retrieve reminders with optional status filter."""
    reminders = list(REMINDERS_DB.values())
    if status:
        reminders = [r for r in reminders if r.status == status]

    return [f"{r.name} (ID: {r.reminder_id}) - {r.reminder_time.strftime('%Y-%m-%d %H:%M:%S')}" for r in reminders]


def delete_reminder(reminder_id: str) -> bool:
    """Delete a reminder by ID."""
    if reminder_id in REMINDERS_DB:
        del REMINDERS_DB[reminder_id]
        return True
    return False

# Enhanced system prompt for tool calling
TOOL_SYSTEM_PROMPT = """
You are an advanced Task Manager agent for TechMartCorporate with access to order management tools.

Mention how you can help employees by:
- Returning detailed info for a task if it exists
- Search tasks by department, status or priority 
- Updating task description
- Updating task priority to Low, Medium or High
- Updating task status to Pending, In progress, Completed, On hold
- Deleting tasks
- Cancelling orders
- Adding new tasks
- Adding reminders
- Retrieving reminders
- Delete reminders

When users request these actions, use the appropriate tools to help them. Always ask for necessary information like task ID and assigned department for verification.

For reminder requests, be empathetic.

Use the following rules for better conversations:
- When you receive user input involving tasks, determine the appropriate tool and parameters to call.
- Always validate inputs before performing updates or additions.
- If a task is not found or inputs are invalid, respond appropriately indicating the issue.
- Use task IDs as provided.
- Only use the tools as needed and return clear, user-friendly responses.
- Be concise and helpful. Always confirm actions taken using the tool responses.
- Users may specify reminder times in any format like "August 4th at 2pm", "2025-08-04T14:00", or "tomorrow at 3 PM".
- Users can optionally provide a name for a reminder. If no name is provided, generate a default using the date/time.
- Use the add_reminder tool with reminder_time_input and name.
- If a tool returns an error, explain why and what the user can do next.
"""

# Create agent with tools
provider = GoogleProvider(api_key='AIzaSyDwhZrg7OXq1RjXzsVz3grhr7Aa9XgI7-Q')
model = GoogleModel('gemini-2.5-flash', provider=provider)

agent = Agent(
    model=model,
    system_prompt=TOOL_SYSTEM_PROMPT,
    tools=[
        filter_tasks_by_department,
        check_task,
        update_description,
        update_priority,
        update_status,
        delete_task,
        add_new_task,
        add_reminder,
        get_reminders,
        delete_reminder
    ]
)
# --- Chat state ---
chat_history = []

# --- Chat Function ---
async def chat(user_message, history):
    history = history or []
    history.append({"role": "user", "content": user_message})  # use lowercase 'user'

    full_prompt = "\n".join(f"{item['role']}: {item['content']}" for item in history)

    # Call Gemini LLM using async Agent
    result = await agent.run(full_prompt)

    # Extract plain text from AgentRunResult
    response = result.output if hasattr(result, 'output') else str(result)

    history.append({"role": "assistant", "content": response})
    return history, history


# --- Gradio UI ---
with gr.Blocks(title="Gemini Chatbot") as demo:
    gr.Markdown("## Gemini Task Manager Chatbot 🤖")
    
    chatbot = gr.Chatbot(type='messages')
    msg = gr.Textbox(placeholder="Type your message here...", label="Your Message")
    clear_btn = gr.Button("Clear")

    state = gr.State([])

    msg.submit(chat, inputs=[msg, state], outputs=[chatbot, state])
    clear_btn.click(lambda: ([], []), None, [chatbot, state])

# --- Launch the app ---
demo.queue().launch()