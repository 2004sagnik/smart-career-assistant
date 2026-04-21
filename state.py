from typing import TypedDict, List


class CapstoneState(TypedDict):
    q_text:       str        # user question
    chat_history: List[dict] # conversation messages
    nav_route:    str        # routing decision
    kb_context:   str        # retrieved knowledge
    kb_sources:   List[str]  # source topics
    tool_output:  str        # tool result
    ai_response:  str        # final answer
    faith_score:  float      # faithfulness score
    retry_count:  int        # eval retry counter
    user_goal:    str        # detected user goal
    user_name:    str        # detected user name
