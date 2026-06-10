PROMPT = """You are helping design evaluation prompts for a physical tool-use benchmark.

For each task, you must generate FIVE evaluation instances that probe different
capabilities while remaining physically realistic and visually plausible.

For this task, you are given:
- task_goal: {task_goal}
- original_tool: {original_tool}
- expected_outcome (for successful completion with a suitable tool):
  {expected_outcome}
- unconventional_tools: {unconventional_tools}
- impossible_tools: {impossible_tools}

Your job is to create FIVE evaluation instances:
1. One instance using the original_tool (tool_type = "regular").
2. Two instances using distinct unconventional_tools (tool_type = "unconventional").
3. Two instances using distinct impossible_tools (tool_type = "impossible").

Each evaluation instance must contain FOUR prompts:

(a) Predictive IMAGE prompt.
- An image-generation prompt that asks model to generate the outcome state of applying tool X to object Y for task Z, without revealing the final result.
- Asks what will happen when applying tool X to object/scenario Y for task Z,
  WITHOUT revealing the final outcome in the prompt text.
- It should sound like a natural user query aimed at generating a single image
  depicting the anticipated situation, but must not state success or failure.
- One example: "Generate an image to visualize the final state of using the book to hammer the nail. Be realistic"

(b) Descriptive IMAGE prompt.
- An image-generation prompt that asks model to generate the outcome state of applying tool X to object Y for task Z by describing the final state.
- For SUCCESSFUL (regular/unconventional) tools:
  - Describe ONLY the final visible state after the task has been completed
    successfully, consistent with `expected_outcome`.
- For IMPOSSIBLE tools:
  - Describe ONLY the final visible failure state (what things look like after the
    failed attempt), inferred from physics and tool limitations.
- Focus on the static final configuration (no process description, no multi-step wording).

(c) Predictive VIDEO prompt.
- A video-generation prompt that asks model to predict and simulate asking it to anticipate the outcome of applying tool X to object Y for task Z, without revealing the final result.
- Asks what will happen when a person uses the tool for the task in a short video,
  WITHOUT revealing whether the attempt succeeds or fails.
- Describe the setup and intended interaction in a way that suggests a short clip,
  but do not say if the outcome is success or failure.
- One example: "Generate a video to illustrate what will happen when using the book to hammer the nail? Be realistic. The video should contain the full process from the start state to the end state."

(d) Descriptive VIDEO prompt.
- A video-generation prompt for a short video showing the full process and the final state.
- For SUCCESSFUL (regular/unconventional) tools:
  - Describe how a person uses the tool on the object over time and end with the
    successful final state consistent with `expected_outcome`.
- For IMPOSSIBLE tools:
  - Describe how the person attempts the task, how and why it fails, and end with
    the visible failure state.

CRITICAL REQUIREMENTS:
- Use natural, fluent English without mentioning "task_goal", "original_tool",
  "unconventional", or "impossible" explicitly inside the prompts.
- Do NOT reveal labels like "regular", "unconventional", or "impossible" inside
  the prompts; those are only for the JSON metadata.
- For predictive prompts:
  - Never state the final outcome explicitly (success or failure).
- For descriptive prompts:
  - Always make the intended outcome (success for regular/unconventional tools,
    failure for impossible tools) explicit and visually checkable.
- Ensure that each of the FIVE instances corresponds to a single specific tool.
- Prefer concise prompts that could realistically be input by a user.

OUTPUT FORMAT (IMPORTANT):
Return ONLY a JSON object with the following structure and no extra text:
{{
  "evaluation_instances": [
    {{
      "tool_type": "regular" | "unconventional" | "impossible",
      "tool": "the specific tool name you used for this instance",
      "expected_outcome": "the description of the expected outcome for successful tools, the description of the failure state for impossible tools",
      "predictive_image_prompt": "a single natural-language predictive IMAGE-generation prompt",
      "descriptive_image_prompt": "a single natural-language descriptive IMAGE-generation prompt for the final state",
      "predictive_video_prompt": "a single natural-language predictive VIDEO-generation prompt",
      "descriptive_video_prompt": "a single natural-language descriptive VIDEO-generation prompt for the process and final state",
    }},
    ... (exactly five instances total)
  ]
}}

- Ensure there are EXACTLY five entries in "evaluation_instances":
  - 1 with tool_type = "regular"
  - 2 with tool_type = "unconventional"
  - 2 with tool_type = "impossible"
- Ensure that the `tool` for each instance appears in the appropriate list:
  - regular: original_tool
  - unconventional: from unconventional_tools
  - impossible: from impossible_tools
"""