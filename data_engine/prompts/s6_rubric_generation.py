PROMPT = """You are designing checklist-based evaluation RUBRICS for a SINGLE
evaluation instance in a physical tool-use benchmark. For this one instance,
you must produce FOUR separate rubrics:
- one for the predictive IMAGE prompt,
- one for the descriptive IMAGE prompt,
- one for the predictive VIDEO prompt,
- one for the descriptive VIDEO prompt.

You are given:
- task_goal: {task_goal}
- action_type: {action_type}
- original_tool: {original_tool}
- expected_outcome_for_task: {expected_outcome}
- required_tool_attributes: {required_tool_attributes}
- unconventional_tools: {unconventional_tools}
- impossible_tools: {impossible_tools}

And for ONE specific evaluation instance you are also given:
- tool_type: {tool_type}   # \"regular\" | \"unconventional\" | \"impossible\"
- tool_name: {tool}
- instance_expected_outcome: {instance_expected_outcome}
- predictive_image_prompt: {predictive_image_prompt}
- descriptive_image_prompt: {descriptive_image_prompt}
- predictive_video_prompt: {predictive_video_prompt}
- descriptive_video_prompt: {descriptive_video_prompt}

Your job is to generate FOUR structured, checklist-based RUBRICS that can
later be used to score model-generated outputs for THIS instance:
- predictive_image_rubric   → for predictive_image_prompt (images only)
- descriptive_image_rubric  → for descriptive_image_prompt (images only)
- predictive_video_rubric   → for predictive_video_prompt (videos only)
- descriptive_video_rubric  → for descriptive_video_prompt (videos only)

The rubric has TWO top-level dimensions, each with THREE sub-dimensions:

1) Instruction Adherence (0–100%)
   Measures whether the generated scene correctly instantiates the entities and
   functional properties required to enable the intended interaction.

   It is decomposed into three sub-dimensions:

   (a) Entity Completeness
       The presence of all required entities.
       Example questions:
       - Is the specified tool present?
       - Is the target object instantiated?
       - Are required contextual elements (supporting surface, environment) included?

   (b) Attribute Fidelity
       Correct instantiation of required functional attributes.
       Example questions:
       - Does the tool exhibit the required functional property
         (e.g., rigidity or sharpness)?
       - Is the material consistent with intended physical behavior?
       - Are size and structural properties compatible with the task?

   (c) Scene Validity
       Spatial configuration and physically feasible arrangement.
       Example questions:
       - Is the tool positioned at the correct interaction region?
       - Is the relative scale between tool and object plausible?
       - Does the arrangement enable the intended interaction?

2) Interaction Accuracy (0–100%)
   Measures whether the interaction outcome and dynamics are correctly realized.

   It is decomposed into three sub-dimensions:

   (a) State Change Correctness
       Correctness of the physically correct final state (or predicted outcome).
       Example questions:
       - Does the object exhibit the correct resulting state
         (e.g., cracked, cut, bent)?
       - In impossible cases, is failure correctly depicted?
       - Is the final configuration consistent with the applied interaction?

   (b) Affordance Grounding
       Whether interaction behavior aligns with object affordances.
       Example questions:
       - Is force applied through a structurally appropriate part of the tool?
       - Is the interaction consistent with object geometry and material constraints?
       - Does the behavior reflect a physically plausible affordance?

   (c) Motion Plausibility  (ONLY for video generation)
       Temporal coherence and dynamical feasibility.
       Example questions:
       - Is motion trajectory continuous?
       - Are deformations temporally consistent?

For EACH of the four rubrics, you must create checklist questions that are:
- SPECIFIC to this task_goal, tool, and evaluation instance;
- Grounded in the provided prompts and expected outcomes;
- Physically meaningful and visually checkable.

Do NOT score anything yourself. Only define the questions.

-------------------------------------------------------------------------------
OUTPUT FORMAT (IMPORTANT)

Return ONLY a JSON object with this structure and no extra text:

{{
  "predictive_image_rubric": {{
    "instruction_adherence": {{
      "entity_completeness": {{
        "checklist_items": [
          {{
            "id": "short_snake_case_identifier",
            "question": "clear yes/no style question about entity completeness for the predictive IMAGE output"
          }}
          // ... more checklist items (3–6 total for this sub-dimension)
        ]
      }},
      "attribute_fidelity": {{
        "checklist_items": [
          {{
            "id": "short_snake_case_identifier",
            "question": "clear yes/no style question about attributes for the predictive IMAGE output"
          }}
          // ... more checklist items
        ]
      }},
      "scene_validity": {{
        "checklist_items": [
          {{
            "id": "short_snake_case_identifier",
            "question": "clear yes/no style question about spatial / scene validity for the predictive IMAGE output"
          }}
          // ... more checklist items
        ]
      }}
    }},
    "interaction_accuracy": {{
      "state_change_correctness": {{
        "checklist_items": [
          {{
            "id": "short_snake_case_identifier",
            "question": "clear yes/no style question about final state correctness for the predictive IMAGE output"
          }}
          // ... more checklist items
        ]
      }},
      "affordance_grounding": {{
        "checklist_items": [
          {{
            "id": "short_snake_case_identifier",
            "question": "clear yes/no style question about affordance-consistent usage for the predictive IMAGE output"
          }}
          // ... more checklist items
        ]
      }},
      "motion_plausibility": {{
        "checklist_items": [
          // empty list if for images, this can be an empty list or omitted questions; keep the key.
        ]
      }}
    }}
  }},
  "descriptive_image_rubric": {{
    "instruction_adherence": {{
      "entity_completeness": {{
        "checklist_items": [
          {{
            "id": "short_snake_case_identifier",
            "question": "clear yes/no style question about entity completeness for the descriptive IMAGE output"
          }}
        ]
      }},
      "attribute_fidelity": {{
        "checklist_items": [
          {{
            "id": "short_snake_case_identifier",
            "question": "clear yes/no style question about attributes for the descriptive IMAGE output"
          }}
        ]
      }},
      "scene_validity": {{
        "checklist_items": [
          {{
            "id": "short_snake_case_identifier",
            "question": "clear yes/no style question about spatial / scene validity for the descriptive IMAGE output"
          }}
        ]
      }}
    }},
    "interaction_accuracy": {{
      "state_change_correctness": {{
        "checklist_items": [
          {{
            "id": "short_snake_case_identifier",
            "question": "clear yes/no style question about final state correctness for the descriptive IMAGE output"
          }}
        ]
      }},
      "affordance_grounding": {{
        "checklist_items": [
          {{
            "id": "short_snake_case_identifier",
            "question": "clear yes/no style question about affordance-consistent usage for the descriptive IMAGE output"
          }}
        ]
      }},
      "motion_plausibility": {{
        "checklist_items": [
          // For images, this can be an empty list or omitted questions; keep the key.
        ]
      }}
    }}
  }},
  "predictive_video_rubric": {{
    "instruction_adherence": {{
      "entity_completeness": {{
        "checklist_items": [
          {{
            "id": "short_snake_case_identifier",
            "question": "clear yes/no style question about entity completeness for the predictive VIDEO output"
          }}
        ]
      }},
      "attribute_fidelity": {{
        "checklist_items": [
          {{
            "id": "short_snake_case_identifier",
            "question": "clear yes/no style question about attributes for the predictive VIDEO output"
          }}
        ]
      }},
      "scene_validity": {{
        "checklist_items": [
          {{
            "id": "short_snake_case_identifier",
            "question": "clear yes/no style question about spatial / scene validity for the predictive VIDEO output"
          }}
        ]
      }}
    }},
    "interaction_accuracy": {{
      "state_change_correctness": {{
        "checklist_items": [
          {{
            "id": "short_snake_case_identifier",
            "question": "clear yes/no style question about final state correctness for the predictive VIDEO output"
          }}
        ]
      }},
      "affordance_grounding": {{
        "checklist_items": [
          {{
            "id": "short_snake_case_identifier",
            "question": "clear yes/no style question about affordance-consistent usage for the predictive VIDEO output"
          }}
        ]
      }},
      "motion_plausibility": {{
        "checklist_items": [
          {{
            "id": "short_snake_case_identifier",
            "question": "clear yes/no style question about temporal / motion plausibility for the predictive VIDEO output"
          }}
        ]
      }}
    }}
  }},
  "descriptive_video_rubric": {{
    "instruction_adherence": {{
      "entity_completeness": {{
        "checklist_items": [
          {{
            "id": "short_snake_case_identifier",
            "question": "clear yes/no style question about entity completeness for the descriptive VIDEO output"
          }}
        ]
      }},
      "attribute_fidelity": {{
        "checklist_items": [
          {{
            "id": "short_snake_case_identifier",
            "question": "clear yes/no style question about attributes for the descriptive VIDEO output"
          }}
        ]
      }},
      "scene_validity": {{
        "checklist_items": [
          {{
            "id": "short_snake_case_identifier",
            "question": "clear yes/no style question about spatial / scene validity for the descriptive VIDEO output"
          }}
        ]
      }}
    }},
    "interaction_accuracy": {{
      "state_change_correctness": {{
        "checklist_items": [
          {{
            "id": "short_snake_case_identifier",
            "question": "clear yes/no style question about final state correctness for the descriptive VIDEO output"
          }}
        ]
      }},
      "affordance_grounding": {{
        "checklist_items": [
          {{
            "id": "short_snake_case_identifier",
            "question": "clear yes/no style question about affordance-consistent usage for the descriptive VIDEO output"
          }}
        ]
      }},
      "motion_plausibility": {{
        "checklist_items": [
          {{
            "id": "short_snake_case_identifier",
            "question": "clear yes/no style question about temporal / motion plausibility for the descriptive VIDEO output"
          }}
        ]
      }}
    }}
  }}
}}

DETAILED INSTRUCTIONS:
- The top-level JSON object MUST contain ALL FOUR keys:
  - "predictive_image_rubric"
  - "descriptive_image_rubric"
  - "predictive_video_rubric"
  - "descriptive_video_rubric"
- Each of those four rubric objects MUST have the SAME internal structure:
  - "instruction_adherence" with sub-dimensions "entity_completeness",
    "attribute_fidelity", "scene_validity", each with a "checklist_items" list.
  - "interaction_accuracy" with sub-dimensions "state_change_correctness",
    "affordance_grounding", "motion_plausibility", each with a "checklist_items" list
    (for image rubrics, "motion_plausibility.checklist_items" may be empty but must exist).
- For EVERY sub-dimension in EVERY rubric:
  - Use 3–6 checklist items when they are meaningful, except for cases where
    motion is not applicable to images (then you may use 0–2 highly specific items).
  - Each checklist item MUST include:
    - "id": a short, unique snake_case identifier (no spaces, no punctuation),
    - "question": a concise, self-contained question that can be judged from
      the generated media for that specific prompt type.
- Tailor the wording of each question to THIS specific task, tool, instance, and
  prompt type (predictive vs descriptive, image vs video).
- Reflect whether the instance is regular / unconventional / impossible when
  phrasing questions about success vs. failure states.
- Do NOT mention internal labels like "instruction adherence" or
  "interaction accuracy" inside the question text itself; those are implicit in
  the JSON structure.
- Do NOT include any comments or explanations outside the JSON. Only return
  the JSON object described above.
"""

