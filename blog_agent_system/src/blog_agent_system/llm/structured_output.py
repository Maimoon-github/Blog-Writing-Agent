from pydantic import BaseModel, Field
from typing import Type

from blog_agent_system.llm.provider import LLMProvider


class OutlineOutput(BaseModel):
    title: str = Field(..., description="Compelling blog post title")
    sections: list[dict] = Field(..., description="List of sections with heading and key_points")
    estimated_read_time: int = Field(..., description="Estimated read time in minutes")


async def generate_structured(
    provider: LLMProvider,
    messages: list,
    output_schema: Type[BaseModel]
) -> BaseModel:
    """Generate structured JSON output using prompt injection + validation."""
    schema_json = output_schema.model_json_schema()
    messages[0]["content"] += (
        f"\n\nYou MUST respond with valid JSON matching this exact schema:\n"
        f"{schema_json}\n"
        f"Respond ONLY with the JSON object. No explanations."
    )

    raw = await provider.generate(messages, response_format=output_schema)

    try:
        return output_schema.model_validate_json(raw)
    except Exception as e:
        # Self-correction attempt
        messages.append({"role": "user", "content": f"Invalid JSON. Fix it: {str(e)}"})
        raw = await provider.generate(messages)
        return output_schema.model_validate_json(raw)