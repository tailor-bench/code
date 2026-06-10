"""Prompt for Step 2: Generate unconventional tools"""

PROMPT = """You are given a task scenario with its original tool and required attributes. Your task is to identify unconventional tools that could potentially accomplish the task (tools not typically used but might work).

Task Information:
- Task Goal: {task_goal}
- Original Tool: {original_tool}
- Expected Outcome: {expected_outcome}
- Required Tool Attributes: {required_tool_attributes}

For unconventional_tools:
- Think of everyday objects that have some of the required attributes but aren't the conventional choice
- These should be objects that might work but are less ideal (e.g., using a coin instead of a screwdriver)
- Include 4-5 examples that match the required attributes

Output your response as a JSON object with the following structure:
{{
    "unconventional_tools": [
        "tool 1",
        "tool 2"
    ]
}}

Return ONLY the JSON object, no additional text."""
