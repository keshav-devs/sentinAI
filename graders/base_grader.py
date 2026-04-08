import numpy as np

ACTION_SCORES = {
    "malicious": {0: 0.0, 1: 1.0, 2: 0.75, 3: 0.75},
    "safe":      {0: 1.0, 1: 0.1, 2: 0.45, 3: 0.55},
}

def compute_reward(action: int, label: str, gt_confidence: float) -> float:
    base = ACTION_SCORES[label][action]
    ambiguity_weight = max((gt_confidence - 0.5) * 2, 0.15)
    softened = base * ambiguity_weight + 0.5 * (1.0 - ambiguity_weight)
    return float(np.clip(softened, 0.0, 1.0))

def episode_score(step_rewards: list, task: str) -> float:
    if not step_rewards:
        return 0.0
    task_weights = {"easy": 1.0, "medium": 1.3, "hard": 1.6}
    avg = sum(step_rewards) / len(step_rewards)
    return round(avg * task_weights.get(task, 1.0), 4)
