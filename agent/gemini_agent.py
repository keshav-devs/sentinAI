import os
import json
import hashlib
from openai import OpenAI

class GeminiAgent:
    def __init__(self, mock_mode: bool = True,
                 mock_responses_path: str = "agent/mock_responses.json"):
        self.mock_mode = mock_mode
        if mock_mode:
            with open(mock_responses_path) as f:
                self.mock_responses = json.load(f)
        else:
            api_base = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
            api_key  = os.environ.get("HF_TOKEN", "sk-placeholder")
            model    = os.environ.get("MODEL_NAME", "gpt-4o-mini")
            self.client = OpenAI(base_url=api_base, api_key=api_key)
            self.model = model

    def act(self, observation: dict) -> int:
        if self.mock_mode:
            prompt_text = observation.get("prompt", "")
            # FIXED: MD5 hash key — no collision from truncation
            key = hashlib.md5(prompt_text.encode()).hexdigest()
            return self.mock_responses.get(key, 1)
        return self._call_llm(observation)

    def _call_llm(self, observation: dict) -> int:
        prompt_text = observation.get("prompt", "")
        sys_ctx     = observation.get("system_context", "")
        history     = observation.get("conversation_history", [])

        msg = (
            f"Security classifier. System context: {sys_ctx}\n"
            f"History: {chr(10).join(history) if history else 'None'}\n"
            f"Prompt: {prompt_text}\n"
            f"0=ALLOW 1=BLOCK 2=SANITIZE 3=ESCALATE\n"
            f"Reply with one digit only."
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": msg}],
                max_tokens=5,
                temperature=0,
            )
            action = int(response.choices[0].message.content.strip()[0])
            return action if action in (0, 1, 2, 3) else 1
        except Exception:
            return 1
