import json
from env.environment import PromptInjectionEnv
from agent.gemini_agent import GeminiAgent
from eval.baseline_keyword import KeywordFilterBaseline
from graders.base_grader import episode_score

def eval_agent(agent, task, data):
    records = [r for r in data if r["task"] == task]
    env = PromptInjectionEnv(task=task, dataset=records, mock_mode=True)
    obs, _ = env.reset()
    rewards = []
    terminated = False
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    while not terminated:
        action = agent.act(obs)
        obs, reward, terminated, _, info = env.step(action)
        rewards.append(reward)
        action_counts[action] = action_counts.get(action, 0) + 1
    return episode_score(rewards, task), action_counts

if __name__ == "__main__":
    with open("data/dataset_eval.json") as f:
        data = json.load(f)

    agent    = GeminiAgent(mock_mode=True)
    baseline = KeywordFilterBaseline()

    results = {}
    print("\n=== Evaluation Results ===\n")
    header = f"{'Task':<10} | {'Agent':>10} | {'Baseline':>10} | {'Delta':>8}"
    print(header)
    print("-" * len(header))

    for task in ("easy", "medium", "hard"):
        a_score, a_counts = eval_agent(agent,    task, data)
        b_score, b_counts = eval_agent(baseline, task, data)
        delta = round((a_score - b_score) * 100, 1)
        sign  = "+" if delta >= 0 else ""
        results[task] = {"agent": a_score, "baseline": b_score, "delta": delta}
        print(f"{task:<10} | {a_score:>10} | {b_score:>10} | {sign}{delta:>6}%")
        print(f"           Agent actions: ALLOW={a_counts[0]} BLOCK={a_counts[1]} "
              f"SANITIZE={a_counts[2]} ESCALATE={a_counts[3]}")

    all_agent    = round(sum(r["agent"]    for r in results.values()) / 3, 4)
    all_baseline = round(sum(r["baseline"] for r in results.values()) / 3, 4)
    all_delta    = round((all_agent - all_baseline) * 100, 1)
    sign = "+" if all_delta >= 0 else ""
    print("-" * len(header))
    print(f"{'OVERALL':<10} | {all_agent:>10} | {all_baseline:>10} | {sign}{all_delta:>6}%")

    with open("eval/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to eval/results.json")
