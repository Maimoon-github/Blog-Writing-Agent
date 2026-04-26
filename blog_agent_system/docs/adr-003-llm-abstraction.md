# ADR-003: LLM Abstraction — Provider-Agnostic Layer with Per-Agent Model Selection

**Status:** Accepted  
**Date:** 2026-04-26  
**Deciders:** Architecture Team  
**Affected:** `llm/`, `config/llm_config.py`, `agents/base.py`

---

## Context

The blog writing system relies on Large Language Models for diverse tasks with conflicting requirements:

- **Research:** Requires speed and low cost; creativity is undesirable (high temperature harmful)
- **Writing:** Requires long-context handling, narrative coherence, and stylistic nuance
- **Editing:** Requires precision, adherence to grammar rules, and minimal hallucination
- **Fact-checking:** Requires structured reasoning and deterministic output
- **Image generation:** Requires a completely different modality (diffusion models)

Additionally, production systems must mitigate vendor lock-in, handle provider outages, and optimize costs by routing tasks to the most cost-effective model capable of achieving the required quality.

We needed an abstraction that:
1. Allows different agents to use different models/providers without code changes
2. Enforces structured output schemas reliably across providers
3. Manages token budgets to prevent runaway costs
4. Provides automatic fallback when a provider is unavailable

---

## Decision

We will implement a **provider-agnostic LLM abstraction layer** with the following characteristics:

- **Unified Interface:** All LLM interactions go through `LLMProvider` abstract base class
- **Per-Agent Configuration:** Each agent role maps to a specific `(provider, model, temperature, max_tokens)` tuple via `AGENT_LLM_CONFIGS`
- **Structured Output Enforcement:** Pydantic schemas are injected into prompts with JSON mode constraints
- **Token Budget Management:** Pre-flight token counting and context truncation prevent overruns
- **Retry & Fallback:** Tenacity-based retries with exponential backoff; ultimate fallback to local Ollama models

---

## Consequences

### Positive

- **Cost optimization:** Research and SEO agents use cheap GPT-4o-mini; writing and editing use premium Claude 3.5 Sonnet only where justified.
- **Vendor independence:** Adding a new provider (e.g., Google Gemini, Mistral) requires only implementing the `LLMProvider` interface—no changes to agent logic.
- **Resilience:** If OpenAI API is rate-limited, the system falls back to Anthropic; if both cloud providers fail, it degrades to local Ollama (quality drops but service continues).
- **Quality control:** Per-agent temperature and token limits prevent inappropriate creativity (e.g., low temp for fact-checking) or truncation (e.g., high token limit for long-form writing).
- **Testability:** The abstraction allows injecting mock providers in unit tests, enabling deterministic agent testing without API calls.

### Negative

- **Lowest-common-denominator effect:** Some provider-specific features (e.g., Anthropic's prompt caching, OpenAI's seed parameter for reproducibility) are abstracted away unless explicitly exposed.
- **Maintenance overhead:** Each provider requires a separate SDK dependency and adapter implementation.
- **Schema drift risk:** Different providers interpret JSON schema instructions differently. We mitigate this with explicit schema injection and Pydantic validation of outputs.

---

## Alternatives Considered

### Direct SDK Calls in Agents

**Why rejected:** Hard-coding `openai.ChatCompletion.create()` inside each agent would create tight vendor coupling, prevent per-agent model selection, and make testing impossible without network mocks.

### LiteLLM Proxy

**Why rejected:** [LiteLLM](https://github.com/BerriAI/litellm) provides a unified API across providers and was seriously considered. It was rejected because:
- **Infrastructure complexity:** Running a proxy service adds a network hop and single point of failure.
- **Limited structured output control:** LiteLLM's abstraction makes it harder to enforce provider-specific JSON mode parameters.
- **Self-hosted requirement:** We would need to operate and monitor the proxy, conflicting with our goal of minimizing operational surface area.

LiteLLM remains an alternative if we scale beyond 3 providers or need unified usage tracking across teams.

### LangChain ChatModels as the Sole Abstraction

**Why rejected:** While LangChain provides `BaseChatModel` interfaces, we found its structured output mechanisms (e.g., `with_structured_output`) inconsistent across providers in earlier versions. Our custom `LLMProvider` + `structured_output.py` module provides stricter enforcement and clearer error messages for our specific use case.

*Note:* We still use LangChain for tool binding and message formatting, but the core generation loop is managed by our abstraction to ensure fallback logic and token management work uniformly.

### Single Model for All Agents

**Why rejected:** Using GPT-4o for every agent would simplify the architecture but increase costs by approximately 400% and reduce quality for specialized tasks (Claude 3.5 Sonnet outperforms on long-form writing coherence).

---

## Implementation Notes

### Provider Hierarchy (`llm/`)

```
llm/
├── provider.py           # Abstract base: generate(), stream(), count_tokens()
├── openai_provider.py    # GPT-4o, GPT-4o-mini, DALL-E 3
├── anthropic_provider.py # Claude 3.5 Sonnet, Claude 3 Haiku
├── ollama_provider.py    # Local Llama 3.1 for fallback
├── factory.py            # Instantiates provider based on AgentLLMConfig
├── token_manager.py      # Context truncation and budget enforcement
└── structured_output.py  # Pydantic schema injection and validation
```

### Configuration (`config/llm_config.py`)

- `AgentLLMConfig` Pydantic model defines per-agent parameters
- `AGENT_LLM_CONFIGS` dictionary maps role names to configurations
- Environment variables override defaults (e.g., `WRITER_MODEL=claude-3-5-sonnet-20241022`)

### Fallback Chain (`llm/factory.py`)

1. Primary provider (configured per agent)
2. Retry 3x with exponential backoff on rate limits
3. Secondary provider (cross-cloud fallback)
4. Tertiary: Ollama local model (`llama3.1:70b`)
5. Hard failure only if all providers exhaust their budgets

### Structured Output Enforcement

- Schema is appended to the system prompt as a JSON Schema string
- Provider is instructed to respond with valid JSON only
- `pydantic.BaseModel.model_validate_json()` validates the raw output
- Validation failures trigger a retry with the error message injected into the prompt (self-correction loop, max 2 attempts)

### Token Management

- `tiktoken` for OpenAI models; Anthropic provides native token counting via SDK
- Pre-flight check: if estimated input + output > model limit, truncate context by removing oldest non-system messages
- Reserved output budget: 4096 tokens always reserved for the response

---

## References

- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Anthropic Messages API](https://docs.anthropic.com/en/api/messages)
- [Ollama API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Pydantic Validation](https://docs.pydantic.dev/latest/concepts/validation/)
- [Tenacity Retry Library](https://github.com/jd/tenacity)
- [tiktoken Tokenizer](https://github.com/openai/tiktoken)
