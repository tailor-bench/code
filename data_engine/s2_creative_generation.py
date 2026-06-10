"""Step 2: Generate unconventional tools"""
import json
import os
import sys
sys.path.append(os.path.dirname(__file__))
from generate import generate
from prompts.s2_task_creative_solution import PROMPT

def generate_creative_solutions(task_data, output_dir, action_name, save=True):
    """Generate unconventional tools"""
    prompt = PROMPT.format(
        task_goal=task_data["task_goal"],
        original_tool=task_data["original_tool"],
        expected_outcome=task_data["expected_outcome"],
        required_tool_attributes=json.dumps(task_data["required_tool_attributes"], indent=2)
    )
    
    response, _ = generate(prompt)
    
    # Parse JSON response
    try:
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            creative_data = json.loads(json_match.group())
        else:
            creative_data = json.loads(response)
        
        # Merge with existing task data
        task_data.update(creative_data)
        
        # Save updated task data (when not in batch mode)
        if save:
            output_file = os.path.join(output_dir, action_name, "s2_creative.json")
            with open(output_file, 'w') as f:
                json.dump(task_data, f, indent=4)
        
        return task_data
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON for {action_name}: {e}")
        print(f"Response was: {response}")
        return None

if __name__ == "__main__":
    # Load action ontology
    ontology_path = os.path.join(os.path.dirname(__file__), "..", "action_ontology.json")
    with open(ontology_path, 'r') as f:
        actions = json.load(f)
    
    # Output directory
    output_dir = os.path.join(os.path.dirname(__file__), "..", "tasks")
    
    # Process each action
    for action in actions:
        action_name = action["action_name"]
        s1_file = os.path.join(output_dir, action_name, "s1_task.json")
        
        if os.path.exists(s1_file):
            print(f"Generating creative solutions for: {action_name}")
            with open(s1_file, 'r') as f:
                s1_data = json.load(f)
            tasks = s1_data.get("tasks", [s1_data] if "task_goal" in s1_data else [])
            results = []
            for task_data in tasks:
                result = generate_creative_solutions(task_data, output_dir, action_name, save=False)
                if result:
                    results.append(result)
            if results:
                output_file = os.path.join(output_dir, action_name, "s2_creative.json")
                with open(output_file, 'w') as f:
                    json.dump({"tasks": results}, f, indent=4)
                print(f"  Processed {len(results)} tasks")
        else:
            print(f"Skipping {action_name}: s1_task.json not found")
