import time
import uuid
import json
from threading import Lock
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Prompt Injection Defense Env", version="1.0.0")

# FIXED: thread-safe session store with TTL eviction
_sessions: dict = {}
_lock = Lock()
SESSION_TTL = 600  # seconds

def _evict_old_sessions():
    now = time.time()
    to_delete = [k for k, v in _sessions.items() if now - v["ts"] > SESSION_TTL]
    for k in to_delete:
        del _sessions[k]

with open("data/dataset_dev.json") as f:
    DEV_DATA = json.load(f)

# Pydantic models for typed responses (required by spec)
class ResetResponse(BaseModel):
    session_id: str
    task: str
    observation: dict
    max_steps: int

class StepRequest(BaseModel):
    session_id: str
    action: int

class StepResponse(BaseModel):
    session_id: str
    observation: dict
    reward: float
    terminated: bool
    truncated: bool
    info: dict

class StateResponse(BaseModel):
    session_id: str
    task: str
    step: int
    max_steps: int
    terminated: bool
    reward_so_far: float
    current_observation: dict

class HealthResponse(BaseModel):
    status: str
    name: str
    version: str
    tasks: list
    action_space: int

@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "name": "prompt-injection-defense-env",
        "version": "1.0.0",
        "tasks": ["easy", "medium", "hard"],
        "action_space": 4,
    }

@app.post("/reset", response_model=ResetResponse)
def reset(task: str = "easy"):
    # FIXED: normalize and validate input
    task = task.lower().strip()
    if task not in ("easy", "medium", "hard"):
        raise HTTPException(
            status_code=422,
            detail="task must be 'easy', 'medium', or 'hard'"
        )
    from env.environment import PromptInjectionEnv
    data = [r for r in DEV_DATA if r["task"] == task]
    if not data:
        raise HTTPException(status_code=500, detail="No data for task")

    env = PromptInjectionEnv(task=task, dataset=data, mock_mode=True)
    obs, _ = env.reset()
    sid = str(uuid.uuid4())

    with _lock:
        _evict_old_sessions()
        _sessions[sid] = {"env": env, "ts": time.time()}

    return {
        "session_id": sid,
        "task": task,
        "observation": obs,
        "max_steps": env.max_steps,
    }

@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    with _lock:
        session = _sessions.get(req.session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    env = session["env"]

    if req.action not in (0, 1, 2, 3):
        raise HTTPException(status_code=422, detail="action must be 0,1,2,3")

    obs, reward, terminated, truncated, info = env.step(req.action)

    # Update timestamp on activity
    with _lock:
        if req.session_id in _sessions:
            _sessions[req.session_id]["ts"] = time.time()

    return {
        "session_id": req.session_id,
        "observation": obs,
        "reward": round(reward, 4),
        "terminated": terminated,
        "truncated": truncated,
        "info": info,
    }

@app.get("/state", response_model=StateResponse)
def state(session_id: str):
    with _lock:
        session = _sessions.get(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    env = session["env"]
    state_dict = env.get_state_dict()

    return {"session_id": session_id, **state_dict}
