from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
import json
import re
from typing import Dict, Any

def router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    topic = state.get("topic", "").strip()
    if not topic:
        return {"research_required": True, "error": "No topic provided"}

    llm = ChatOllama(model="mistral", temperature=0.0)

    system_prompt = open("src/prompts/router_prompt.txt").read()

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Topic: {topic}")
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()
        
        # Try direct JSON
        try:
            parsed = json.loads(raw)
        except:
            # Regex fallback
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            parsed = json.loads(match.group()) if match else {}
        
        research_required = bool(parsed.get("research_required", True))
        safe = bool(parsed.get("safe", True))
        
        if not safe:
            return {"research_required": False, "error": "Topic rejected by safety filter"}
            
        print(f"🔀 Router → research_required: {research_required}")
        return {"research_required": research_required}
        
    except Exception as e:
        print(f"⚠️ Router fallback: {e}")
        return {"research_required": True}