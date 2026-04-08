import time
import json
import streamlit as st
from env.environment import PromptInjectionEnv
from agent.gemini_agent import GeminiAgent
from eval.baseline_keyword import KeywordFilterBaseline
from graders.base_grader import episode_score

st.set_page_config(
    page_title="Prompt Injection Defense Env",
    layout="wide"
)
st.title("Prompt Injection Defense Environment")
st.caption("Meta OpenEnv Hackathon 2026")

with open("data/dataset_dev.json") as f:
    DEV_DATA = json.load(f)

# Sidebar controls
with st.sidebar:
    st.header("Configuration")
    task     = st.selectbox("Task", ["easy", "medium", "hard"])
    ep_len   = st.slider("Episode length", 3, 20, 10)
    run_btn  = st.button("Run episode")
    reset_btn = st.button("Reset")

ACTION_LABELS = {0: "ALLOW", 1: "BLOCK", 2: "SANITIZE", 3: "ESCALATE"}
ACTION_COLORS = {0: "green",  1: "red",   2: "orange",   3: "blue"}

if reset_btn or "rewards" not in st.session_state:
    st.session_state.rewards = []
    st.session_state.log = []
    st.session_state.done = False

if run_btn:
    data = [r for r in DEV_DATA if r["task"] == task][:ep_len]
    env   = PromptInjectionEnv(task=task, dataset=data, mock_mode=True)
    agent = GeminiAgent(mock_mode=True)
    obs, _ = env.reset()
    st.session_state.rewards = []
    st.session_state.log = []

    left, right = st.columns(2)
    reward_placeholder = st.empty()
    terminated = False

    while not terminated:
        action = agent.act(obs)
        obs, reward, terminated, _, info = env.step(action)
        st.session_state.rewards.append(reward)
        st.session_state.log.append({
            "step":    info["step"],
            "prompt":  data[info["step"] - 1]["prompt"][:80],
            "context": data[info["step"] - 1]["system_context"][:60],
            "label":   info["label"],
            "action":  action,
            "reward":  round(reward, 3),
            "backend": info["mock_backend_response"],
        })

        with left:
            entry = st.session_state.log[-1]
            st.markdown(f"**Step {entry['step']}**")
            st.caption(entry["context"])
            st.info(entry["prompt"])

        with right:
            entry = st.session_state.log[-1]
            color = ACTION_COLORS[entry["action"]]
            label = ACTION_LABELS[entry["action"]]
            st.markdown(
                f":{color}[**{label}**]  —  reward: `{entry['reward']}`"
            )
            st.code(entry["backend"])

        reward_placeholder.bar_chart(st.session_state.rewards)
        time.sleep(0.4)

    score = episode_score(st.session_state.rewards, task)
    st.success(f"Episode complete — score: {score}")

    # Baseline comparison
    baseline = KeywordFilterBaseline()
    env2 = PromptInjectionEnv(task=task, dataset=data, mock_mode=True)
    obs2, _ = env2.reset()
    b_rewards = []
    term2 = False
    while not term2:
        a2 = baseline.act(obs2)
        obs2, r2, term2, _, _ = env2.step(a2)
        b_rewards.append(r2)
    b_score = episode_score(b_rewards, task)

    st.markdown("### Agent vs baseline")
    c1, c2, c3 = st.columns(3)
    c1.metric("Agent score",    score)
    c2.metric("Baseline score", b_score)
    c3.metric("Delta", round(score - b_score, 4))
