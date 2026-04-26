class TokenBudgetManager:
    """Token budget management for Ollama context windows."""

    def __init__(self, max_tokens: int = 128000):
        self.max_tokens = max_tokens
        self.reserved_output = 4096
        self.available_input = max_tokens - self.reserved_output

    def truncate_context(self, messages: list, provider: "LLMProvider") -> list:
        """Truncate oldest non-system messages if over budget."""
        total = sum(provider.count_tokens(m["content"]) for m in messages)
        while total > self.available_input and len(messages) > 2:
            for i, msg in enumerate(messages):
                if msg.get("role") != "system":
                    total -= provider.count_tokens(msg["content"])
                    messages.pop(i)
                    break
        return messages