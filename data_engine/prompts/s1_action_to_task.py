"""Prompt for Step 1: Convert action to task with original tool and required attributes"""

PROMPT = """You are given an action from an action ontology.
Your task is to generate 20 diverse, realistic task scenarios that require this action. 
Each task must include:
1. A specific task goal (what needs to be accomplished, please make it as common as possible)
2. An original/conventional tool that would be used for this task
3. The expected outcome of completing the task
4. Required tool attributes that enable the action (based on physics and affordance)

Action Information:
- Action Name: {action_name}
- Action Description: {action_description}
- Physics: {physics}
- Affordance: {affordance}

Task Scenario Requirements:
Each task scenario should be simple, direct, and visually clear enough that the entire process—from start to successful outcome—could be convincingly demonstrated in a 5 second video clip. Focus on actions that can be completed swiftly, show obvious visible change, and do not require prolonged, hidden, or ambiguous steps. Ensure that for every task, an observer could identify the action, tool, and result within a brief visual sequence.
Generate exactly 20 tasks. Be CREATIVE and DIVERSE across scenarios—vary contexts (home, kitchen, workshop, office, outdoor, etc.), materials, scales, and use cases. Avoid repetition. At the same time, keep each task REALISTIC: specific, concrete, and plausible in everyday life. Each task should clearly require the given action.

Output your response as a JSON object with the following structure:
{{
    "tasks": [
        {{
            "task_goal": "a specific task description (e.g., 'tighten a loose screw on a chair')",
            "original_tool": "the conventional tool name (e.g., 'screwdriver')",
            "expected_outcome": "what happens when the task is completed successfully",
            "required_tool_attributes": [
                "attribute 1 (e.g., 'narrow tip')",
                "attribute 2 (e.g., 'rigid structure')",
                "attribute 3 (e.g., 'torque transmission')"
            ]
        }},
        ... (20 tasks total)
    ]
}}

The required_tool_attributes for each task should be derived from the physics and affordance information. Think about what physical properties the tool must have to enable the action.

Return ONLY the JSON object, no additional text."""
