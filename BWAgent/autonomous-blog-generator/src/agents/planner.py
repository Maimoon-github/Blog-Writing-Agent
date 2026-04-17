from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from src.models.blog_plan import BlogPlan, Section
import json
import re
from typing import Dict, Any

def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    topic = state.get("topic", "").strip()
    research_required = state.get("research_required", True)

    if not topic:
        return {"error": "No topic provided"}

    llm = ChatOllama(model="mistral", temperature=0.3)

    system_prompt = open("src/prompts/planner_prompt.txt").read()
    user_content = system_prompt.replace("{topic}", topic).replace("{research_required}", str(research_required))

    messages = [
        SystemMessage(content="You are a precise blog planner. Always respond with valid JSON only."),
        HumanMessage(content=user_content)
    ]

    try:
        # Primary: structured output (works well with recent Ollama + Mistral)
        structured_llm = llm.with_structured_output(BlogPlan, method="json_schema")
        blog_plan: BlogPlan = structured_llm.invoke(messages)
        print(f"✅ Planner (structured) → Title: {blog_plan.blog_title} | Sections: {len(blog_plan.sections)}")
        
    except Exception as e:
        print(f"⚠️ Structured output failed ({e}), using fallback parser...")
        # Fallback: raw invoke + parse (your style)
        raw_response = llm.invoke(messages)
        raw_content = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
        
        blog_plan = _fallback_parse(raw_content, topic, research_required)

    return {"plan": blog_plan}

def _fallback_parse(raw_content: str, topic: str, research_required: bool) -> BlogPlan:
    # Strategy 1: direct JSON
    try:
        data = json.loads(raw_content)
        return _dict_to_blog_plan(data, topic, research_required)
    except:
        pass

    # Strategy 2: extract JSON block
    match = re.search(r'\{.*\}', raw_content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return _dict_to_blog_plan(data, topic, research_required)
        except:
            pass

    # Final default (from your code style)
    print("⚠️ Using default plan")
    return _create_default_plan(topic, research_required)

# Reuse your excellent _dict_to_blog_plan and _create_default_plan logic here if you want
# For brevity, you can copy the functions from your planner.py into this file (or import if you refactor later)

def _dict_to_blog_plan(data: Dict, topic: str, research_required: bool) -> BlogPlan:
    # Simplified version - expand with your rich logic
    sections = []
    for i, sec in enumerate(data.get("sections", [])[:6], 1):
        sections.append(Section(
            id=f"section_{i}",
            title=sec.get("title", f"Section {i}"),
            description=sec.get("description", ""),
            word_count=sec.get("word_count", 400),
            search_query=sec.get("search_query") if research_required else None,
            image_prompt=sec.get("image_prompt", f"Illustration for {sec.get('title', 'section')}"),
        ))
    
    if not sections:
        sections = [Section(id="section_1", title="Introduction", description="", word_count=400, search_query=None, image_prompt="")]
    
    return BlogPlan(
        blog_title=data.get("blog_title", f"Blog about {topic}"),
        feature_image_prompt=data.get("feature_image_prompt", f"Hero image for {topic}"),
        sections=sections,
        research_required=research_required,
    )

def _create_default_plan(topic: str, research_required: bool) -> BlogPlan:
    # Paste your rich _create_default_plan from the provided file if desired
    # For now, a minimal working version:
    return BlogPlan(
        blog_title=f"Understanding {topic}",
        feature_image_prompt=f"Professional hero image representing {topic}, high quality, cinematic",
        research_required=research_required,
        sections=[
            Section(id="section_1", title="Introduction", description="Introduce the topic", word_count=350, search_query=None, image_prompt="Intro image"),
            Section(id="section_2", title="Key Concepts", description="Core ideas", word_count=400, search_query=None, image_prompt="Concepts image"),
            Section(id="section_3", title="Practical Applications", description="How to use it", word_count=400, search_query=None, image_prompt="Applications image"),
            Section(id="section_4", title="Future Outlook", description="What's next", word_count=350, search_query=None, image_prompt="Future image"),
        ]
    )