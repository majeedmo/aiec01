from __future__ import annotations

import os
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from app.graphs.simple_agent import graph as agent_graph
from app.models import get_chat_model

MAX_HELPFULNESS_ATTEMPTS = 3

JUDGE_PROMPT = """You are evaluating whether an AI assistant's response is helpful.

The assistant specializes in feline (cat) health. A response is helpful when it:
- Directly addresses the user's question
- Uses tools or retrieved information when needed for cat-health facts
- Cites sources when tool results inform the answer
- Stays relevant to cat health (or clearly explains when a question is out of scope)

User question:
{question}

Assistant response:
{answer}

Decide if the response is helpful. If not, provide actionable feedback for the assistant to improve on retry."""


class HelpfulnessVerdict(BaseModel):
    is_helpful: bool = Field(description="True if the response adequately helps the user.")
    reason: str = Field(description="Brief explanation of the verdict.")
    feedback: str = Field(
        description="If not helpful, specific guidance for improving the next attempt."
    )


class AgentState(MessagesState):
    helpfulness_attempts: int
    is_helpful: bool | None
    last_feedback: str | None


def get_judge_model():
    model_name = os.environ.get("OPENAI_JUDGE_MODEL")
    return get_chat_model(model_name).with_structured_output(HelpfulnessVerdict)


RETRY_FEEDBACK_PREFIX = "Your previous response was not helpful enough."


def get_original_user_question(messages: list) -> str | None:
    for message in messages:
        if isinstance(message, HumanMessage):
            content = str(message.content)
            if not content.startswith(RETRY_FEEDBACK_PREFIX):
                return content
    return None


def get_latest_final_ai_message(messages: list) -> str | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return str(message.content)
    return None


def call_agent(state: AgentState) -> dict:
    result = agent_graph.invoke({"messages": state["messages"]})
    return {"messages": result["messages"]}


def check_helpfulness(state: AgentState) -> dict:
    question = get_original_user_question(state["messages"]) or ""
    answer = get_latest_final_ai_message(state["messages"]) or ""

    judge = get_judge_model()
    verdict = judge.invoke(
        JUDGE_PROMPT.format(question=question, answer=answer),
    )

    updates: dict = {
        "is_helpful": verdict.is_helpful,
        "last_feedback": verdict.feedback if not verdict.is_helpful else None,
    }

    if verdict.is_helpful:
        return updates

    attempts = state.get("helpfulness_attempts", 0) + 1
    updates["helpfulness_attempts"] = attempts

    if attempts < MAX_HELPFULNESS_ATTEMPTS:
        updates["messages"] = [
            HumanMessage(
                content=(
                    f"{RETRY_FEEDBACK_PREFIX} "
                    f"Reason: {verdict.reason}. "
                    f"Please improve your answer: {verdict.feedback}"
                )
            )
        ]

    return updates


def route_after_helpfulness(
    state: AgentState,
) -> Literal["agent", "end"]:
    if state.get("is_helpful"):
        return "end"

    attempts = state.get("helpfulness_attempts", 0)
    if attempts >= MAX_HELPFULNESS_ATTEMPTS:
        return "end"

    return "agent"


builder = StateGraph(AgentState)
builder.add_node("agent", call_agent)
builder.add_node("check_helpfulness", check_helpfulness)
builder.add_edge(START, "agent")
builder.add_edge("agent", "check_helpfulness")
builder.add_conditional_edges(
    "check_helpfulness",
    route_after_helpfulness,
    {"agent": "agent", "end": END},
)

graph = builder.compile()
