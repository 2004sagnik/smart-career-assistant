import os
import re
import json

from dotenv import load_dotenv
load_dotenv()

from state import CapstoneState
from knowledge_base import retrieve
from tools import run_tool
from google import genai


# ── LLM helper ────────────────────────────────────────────────────────────────

def _llm(system: str, user: str, max_tokens: int = 800) -> str:
    provider = os.environ.get("LLM_PROVIDER", "gemini").strip().lower()

    # ── GROQ ────────────────────────────────────────────────────────────────
    if provider == "groq":
        try:
            from groq import Groq
            api_key = os.environ.get("GROQ_API_KEY", "").strip()

            if not api_key:
                return "Setup error: GROQ_API_KEY not set"

            client = Groq(api_key=api_key)

            resp = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0.1,
            )

            return resp.choices[0].message.content.strip()

        except Exception:
            return "I am currently unable to access the AI model. Please try again later."

    # ── GEMINI ──────────────────────────────────────────────────────────────
    else:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()

        if not api_key:
            return "Setup error: GEMINI_API_KEY not set"

        try:
            client = genai.Client(api_key=api_key)

            full_prompt = f"{system}\n\n{user}"

            response = client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=full_prompt,
            )

            return response.text.strip()

        except Exception:
            return "I am currently unable to access the AI model due to API limits. Please try again later."


# ── 1. memory_node ────────────────────────────────────────────────────────────

def memory_node(state: CapstoneState) -> CapstoneState:
    history = list(state.get("chat_history") or [])
    question = state.get("q_text", "").strip()

    if question:
        history.append({"role": "user", "content": question})
    history = history[-6:]

    user_name = state.get("user_name", "") or ""
    if not user_name:
        combined = " ".join(m.get("content", "") for m in history)
        m = re.search(r"(?:my name is|i am|i'm|call me)\s+([A-Z][a-z]+)", combined, re.IGNORECASE)
        if m:
            user_name = m.group(1).strip().title()

    user_goal = state.get("user_goal", "") or ""
    if not user_goal:
        text = " ".join(m.get("content", "") for m in history).lower()
        if "ssc" in text:
            user_goal = "SSC exam preparation"
        elif "bank" in text:
            user_goal = "Banking exam preparation"
        elif "placement" in text:
            user_goal = "Campus placement preparation"
        elif "dsa" in text:
            user_goal = "DSA preparation"

    return {
        **state,
        "chat_history": history,
        "user_name": user_name,
        "user_goal": user_goal,
    }


# ── 2. router_node ────────────────────────────────────────────────────────────

def router_node(state: CapstoneState) -> CapstoneState:
    question = state.get("q_text", "").lower()

    if any(k in question for k in ["date", "time", "calculate", "+", "-", "*", "/"]):
        route = "calc"
    elif any(k in question for k in ["hi", "hello", "thanks"]):
        route = "none"
    else:
        route = "kb"

    return {**state, "nav_route": route}


# ── 3. retrieval_node ─────────────────────────────────────────────────────────

def retrieval_node(state: CapstoneState) -> CapstoneState:
    question = state.get("q_text", "")
    kb_text, sources = retrieve(question, n_results=3)
    return {**state, "kb_context": kb_text, "kb_sources": sources}


# ── 4. skip_node ──────────────────────────────────────────────────────────────

def skip_node(state: CapstoneState) -> CapstoneState:
    return {**state, "kb_context": "", "kb_sources": [], "tool_output": ""}


# ── 5. tool_node ──────────────────────────────────────────────────────────────

def tool_node(state: CapstoneState) -> CapstoneState:
    question = state.get("q_text", "")

    if any(k in question.lower() for k in ["date", "time"]):
        result = run_tool("datetime", "")
    else:
        try:
            result = run_tool("calculator", question)
        except Exception:
            result = "Error in calculation"

    return {**state, "tool_output": result, "kb_context": "", "kb_sources": []}


# ── 6. answer_node ────────────────────────────────────────────────────────────

def answer_node(state: CapstoneState) -> CapstoneState:
    question   = state.get("q_text", "")
    kb_context = state.get("kb_context", "") or ""
    tool_out   = state.get("tool_output", "") or ""
    history    = list(state.get("chat_history") or [])
    user_name  = state.get("user_name", "") or ""

    context = kb_context or tool_out or ""

    system = "You are a helpful career assistant for B.Tech students."

    prompt = f"""
User: {question}

Context:
{context}

Answer clearly:
"""

    try:
        ai_resp = _llm(system, prompt)
    except Exception:
        ai_resp = "Something went wrong while generating the response."

    history.append({"role": "assistant", "content": ai_resp})
    history = history[-6:]

    return {
        **state,
        "ai_response": ai_resp,
        "chat_history": history,
    }


# ── 7. eval_node ──────────────────────────────────────────────────────────────

def eval_node(state: CapstoneState) -> CapstoneState:
    return {**state, "faith_score": 0.9}


# ── 8. save_node ──────────────────────────────────────────────────────────────

def save_node(state: CapstoneState) -> CapstoneState:
    return state