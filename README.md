---
title: Prompt Injection Defense Env
emoji: shield
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# Prompt Injection Defense Environment

This environment abstracts the core detection policy problem of AI gateway 
systems into a Gymnasium-compatible RL format, enabling systematic evaluation 
of prompt injection defense classifiers.

## Installation
```bash
pip install -r requirements.txt
```

## Quick start
```bash
python inference.py
```

## Demo
```bash
streamlit run demo/app.py
```

## Environment API
- POST /reset?task=easy|medium|hard  → ResetResponse
- POST /step  → StepResponse  
- GET  /state?session_id=xxx  → StateResponse
- GET  /health  → HealthResponse

## Action space
Discrete(4): 0=ALLOW, 1=BLOCK, 2=SANITIZE, 3=ESCALATE

## Observation space
- prompt: text (max 2048 chars)
- system_context: text (max 512 chars)
- turn: integer

## Reward function
```
ambiguity_weight = max((gt_confidence - 0.5) * 2, 0.15)
reward = base_score * ambiguity_weight + 0.5 * (1 - ambiguity_weight)
clipped to [0.0, 1.0] per step
```

## Dataset schema
id, task, system_context, prompt, label, injection_type, gt_confidence,
adversarial_system_context

## Known limitations
- Base64-encoded injection payloads not detected
- Unicode homoglyph substitution bypasses regex patterns
- Non-English injection strings not covered
- Code-comment-embedded instructions not parsed
