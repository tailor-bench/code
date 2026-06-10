"""Prompt for Step 4: Revise eval instances for visual generation models"""

FEW_SHOT_EXAMPLES = """
Example 1:
{
    "task_goal": "Squeeze juice from a lemon half",
    "action_type": "press",
    "original_tool": "handheld citrus squeezer",
    "expected_outcome": "Fresh lemon juice flows into a cup while the rind compresses",
    "required_tool_attributes": ["concave rigid press cups", "hinged lever arms for mechanical advantage", "corrosion-resistant material", "drain slots/orifice"],
    "unconventional_tools": ["potato ricer (stainless steel, perforated cup press)", "two spoons back-to-back as a manual press (squeeze lemon half)"],
    "opposite_tool_attributes": ["flat or convex, flexible pressing surfaces instead of concave rigid press cups", "no hinged lever action (one-piece, non-levered, floppy design) providing no mechanical advantage", "sealed pressing surfaces with no drain slots/orifice to direct juice out"],
    "impossible_tools": ["kitchen sponge", "feather duster"]
}

Example 2:
{
    "task_goal": "Shatter a car side window to escape during an emergency",
    "action_type": "break",
    "original_tool": "emergency window hammer",
    "expected_outcome": "The tempered glass fractures into small cubes, creating an exit opening",
    "required_tool_attributes": ["hardened pointed tip to focus impact", "sufficient head mass for impulse delivery", "ergonomic, non-slip grip", "spring-loaded striker or rigid hammer head for sudden force"],
    "unconventional_tools": ["car seat headrest metal prongs used as a puncture/impact point", "wrench"],
    "opposite_tool_attributes": ["soft, rounded blunt tip that disperses impact instead of focusing it", "very low-mass, compressible head that absorbs energy rather than delivering impulse", "slippery, flexible handle that reduces grip and bends instead of transferring force"],
    "impossible_tools": ["water bottle", "foam pool noodle"]
}

Example 3:
{
    "task_goal": "Fill a kitchen sink for soaking dishes",
    "action_type": "block",
    "original_tool": "sink stopper",
    "expected_outcome": "Water is retained in the basin with no drain flow.",
    "required_tool_attributes": ["impermeable material", "snug, tapered fit to drain opening", "structural stiffness to resist water pressure", "grippable edge for placement and removal"],
    "unconventional_tools": ["flat metal jar lid covered in plastic wrap and weighted by a mug", "mini cup plunger pressed firmly to create suction over the drain"],
    "opposite_tool_attributes": ["porous or absorbent material that allows water to pass through", "loose, non-tapered shape that cannot form a seal with the drain opening", "flimsy, collapsible structure that deforms under water pressure and won't hold position"],
    "impossible_tools": ["kitchen sponge", "paper towel sheet"]
}
"""

PROMPT = """You are refining task instances so they are suitable for image or video generation models. For each task you will:
1. Filter out tools that are not creative or not easily visualizable.
2. Select exactly 3 most diverse and realistic tools for unconventional_tools, and exactly 3 for impossible_tools (or fewer only if fewer than 3 pass the filter).
3. Rewrite tool names/descriptions to be concise but clear for a visual model.

Guidelines:
- Filter: Remove any tool from unconventional_tools and impossible_tools that is not creative or not easily visualizable (e.g. abstract, purely conceptual, or hard to depict in a single image or short clip).
- Unconventional tools: Make unconventional_tools more creative. Prefer surprising repurposing of everyday objects (e.g. using a potato ricer for lemon juice, headrest prongs as a window breaker), inventive or non-obvious choices that still satisfy the required attributes, rather than bland or obvious alternatives.
- Select: Choose exactly 3 tools for unconventional_tools and exactly 3 for impossible_tools when possible. Pick the 3 most diverse and realistic; prefer concrete, recognizable, and visually distinct tools. If fewer than 3 pass the filter, generate more to reach 3.
- Revise: Rewrite each kept tool to a concise but clear short phrase for a visual model—no long clauses; enough detail to unambiguously generate the object or action.

Output: Return a single JSON object with exactly these keys (preserve task_goal, original_tool, expected_outcome, required_tool_attributes, opposite_tool_attributes from the input; only refine unconventional_tools and impossible_tools; include action_type):
- action_type (string)
- task_goal (string)
- original_tool (string)
- expected_outcome (string)
- required_tool_attributes (array of strings)
- opposite_tool_attributes (array of strings)
- unconventional_tools (array of strings, exactly 3 when possible, else fewer)
- impossible_tools (array of strings, exactly 3 when possible, else fewer)

Examples of refined tasks (match this style and structure):
""" + FEW_SHOT_EXAMPLES + """

---

Task to refine (input; add action_type if missing, then refine as above):

{task_json}

---

Return ONLY the JSON object for the refined task, no additional text."""
