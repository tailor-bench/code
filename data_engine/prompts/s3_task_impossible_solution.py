"""Prompt for Step 3: Generate opposite attributes and impossible tools"""

PROMPT = """You are given a task scenario with its required attributes. Your task is to:
1. Identify opposite tool attributes (attributes that would make the task difficult or impossible)
2. Identify tools that would be IMPOSSIBLE to use for this task

Task Information:
- Task Goal: {task_goal}
- Original Tool: {original_tool}
- Required Tool Attributes: {required_tool_attributes}

For opposite_tool_attributes:
- These are attributes that directly oppose or conflict with the required attributes
- Think about what would make the tool ineffective (e.g., if "rigid structure" is required, "soft material" would be opposite)
- Include 2-3 examples

For impossible_tools:
- Generate a list of tools/objects that would be impossible to use for this task because they:
  - Have the opposite attributes (e.g., soft, flexible, rounded when rigid, sharp is needed)
  - Lack any of the critical required attributes
  - Are fundamentally incompatible with the physics of the action
- These should be objects that clearly cannot accomplish the task.
- Include 4-5 examples

Output your response as a JSON object with the following structure:
{{
    "opposite_tool_attributes": [
        "opposite attribute 1",
        "opposite attribute 2"
    ],
    "impossible_tools": [
        "tool 1",
        "tool 2",
        "tool 3",
        "tool 4"
    ]
}}

Return ONLY the JSON object, no additional text."""
