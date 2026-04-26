# src/blog_agent_system/llm/structured_output.py
from pydantic import BaseModel, Field
from typing import Type

class OutlineOutput(BaseModel):
    title: str = Field(..., description="Compelling blog post title")
    sections: list[dict] = Field(..., description="List of sections with heading and key_points")
    estimated_read_time: int = Field(..., description="Estimated read time in minutes")

async def generate_structured(
    provider: LLMProvider,
    messages: list,
    output_schema: Type[BaseModel]
) -> BaseModel:
    # Add schema instruction to system prompt
    schema_json = output_schema.model_json_schema()
    messages[0]["content"] += f"\n\nYou must respond with valid JSON matching this schema: {schema_json}"
    
    raw = await provider.generate(messages, response_format=output_schema)
    return output_schema.model_validate_json(raw)