class KeywordFilterBaseline:
    KEYWORDS = [
        "ignore", "jailbreak", "system:", "you are now", "forget your",
        "unrestricted", "DAN", "no restrictions", "reveal your", "bypass",
        "disregard", "override", "new task:", "new instruction:",
    ]

    def act(self, observation: dict) -> int:
        prompt = observation.get("prompt", "").lower()
        if any(kw.lower() in prompt for kw in self.KEYWORDS):
            return 1
        return 0
