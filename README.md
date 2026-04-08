---
title: Prompt Injection Defense Env
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# 🛡️ SentinAI: Prompt Injection Defense Environment (PIDE)

**SentinAI** is a high-fidelity **Gymnasium-compatible** Reinforcement Learning (RL) environment built for the **Meta OpenEnv Hackathon 2026**. It translates the complex problem of AI gateway security into a stateful simulation where agents must detect and mitigate prompt injection attacks.

## 👥 The Team
* **Team Lead:** [Your Name/ID] — [@YourGitHubHandle](https://github.com/YourGitHubHandle)
* **Teammate:** [Name/ID] — [@Teammate1Handle](https://github.com/Teammate1Handle)
* **Teammate:** [Name/ID] — [@Teammate2Handle](https://github.com/Teammate2Handle)

## 🚀 Real-World Impact
Unlike "toy" environments, SentinAI simulates the **AI Firewall** problem faced by production LLM deployments. A key innovation is our **Breach Propagation** mechanic: if an agent fails to block an attack in a multi-turn session, the system context is corrupted, impacting the rewards for all subsequent turns in that episode.

## 🕹️ Environment API
Following the strict **OpenEnv Standard**, the environment exposes the following endpoints via a stateful FastAPI backend:

* **POST `/reset?task=easy|medium|hard`** → Initialize a fresh session.
* **POST `/step`** → Execute one of four defensive actions.
* **GET `/state`** → Query current session metrics and observations.

### **Action Space: Discrete(4)**
| Value | Action | Description |
| :--- | :--- | :--- |
| **0** | **ALLOW** | Forward the prompt to the backend LLM. |
| **1** | **BLOCK** | Reject the prompt as a clear injection attempt. |
| **2** | **SANITIZE** | Strip detected triggers and forward cleaned text. |
| **3** | **ESCALATE** | Flag for human review (best for ambiguous cases). |

### **Observation Space**
* **`prompt`**: Raw user input text (max 2048 chars).
* **`system_context`**: Active operational rules and security state.
* **`turn`**: Current step index in the episode.

## 📈 Reward Function (Confidence-Aware)
The environment uses **Ambiguity Weighting** to reward agents for calibrated uncertainty, discouraging "over-blocking" of safe content:

```python
# Ambiguity weighting formula implemented in graders/base_grader.py
ambiguity_weight = max((gt_confidence - 0.5) * 2, 0.15)
reward = base_score * ambiguity_weight + 0.5 * (1 - ambiguity_weight)
# Final rewards are clipped to [0.0, 1.0] per step.
🛠️ Installation & Quick Start
SentinAI is optimized for 2 vCPU / 8GB RAM environments.

1. Install Dependencies

Bash
pip install -r requirements.txt
2. Run Evaluation (Mandatory Logging)
Execute the baseline inference script to see compliant [START], [STEP], and [END] output:

Bash
python inference.py
3. Launch Visual Demo

Bash
streamlit run demo/app.py
🐳 Docker & Deployment
Deploying to HuggingFace Spaces is handled via the included Dockerfile.

Required Secrets: API_BASE_URL, MODEL_NAME, HF_TOKEN.

📜 License
This project is licensed under the MIT License.
