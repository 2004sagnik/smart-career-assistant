import math
from datetime import datetime


def datetime_tool(query: str = "") -> str:
    now = datetime.now()
    day_of_year = now.timetuple().tm_yday
    days_remaining = 365 - day_of_year
    return (
        f"Current Date   : {now.strftime('%d %B %Y')} ({now.strftime('%A')})\n"
        f"Current Time   : {now.strftime('%I:%M %p')}\n"
        f"Month          : {now.strftime('%B %Y')}\n"
        f"Quarter        : Q{(now.month - 1) // 3 + 1} of {now.year}\n"
        f"Day of Year    : {day_of_year} / 365\n"
        f"Days Left in Year : {days_remaining}"
    )


def calculator_tool(expression: str) -> str:
    allowed_chars = set("0123456789 +-*/().")
    if not all(c in allowed_chars for c in expression):
        return "Error: Only basic arithmetic is allowed (+, -, *, /, parentheses, decimals)."
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return f"Result of '{expression}' = {result}"
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        return f"Error evaluating '{expression}': {str(e)}"


def run_tool(tool_name: str, tool_input: str) -> str:
    if tool_name == "datetime":
        return datetime_tool(tool_input)
    elif tool_name == "calculator":
        return calculator_tool(tool_input)
    return f"Unknown tool: '{tool_name}'"
