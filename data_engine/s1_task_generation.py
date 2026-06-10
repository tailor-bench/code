"""Step 1: Generate task from action ontology"""
import json
import os
import sys
sys.path.append(os.path.dirname(__file__))
from generate import generate
from prompts.s1_action_to_task import PROMPT

def generate_task_from_action(action_data, output_dir):
    """Generate 20 task JSONs from action data"""
    prompt = PROMPT.format(
        action_name=action_data["action_name"],
        action_description=action_data["action_description"],
        physics=action_data["physics"],
        affordance=action_data["affordance"]
    )
    
    response, _ = generate(prompt)
    
    # Parse JSON response
    try:
        # Extract JSON from response (in case there's extra text)
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(response)
        
        # Ensure we have a tasks array
        tasks = result.get("tasks", [result] if "task_goal" in result else [])
        
        # Save to file
        action_name = action_data["action_name"]
        output_file = os.path.join(output_dir, action_name, "s1_task.json")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump({"tasks": tasks}, f, indent=4)
        
        return {"tasks": tasks}
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON for {action_data['action_name']}: {e}")
        print(f"Response was: {response}")
        return None

if __name__ == "__main__":
    # Load action ontology
    ontology_path = os.path.join(os.path.dirname(__file__), "..", "action_ontology.json")
    with open(ontology_path, 'r') as f:
        actions = json.load(f)
    
    # Output directory
    output_dir = os.path.join(os.path.dirname(__file__), "..", "tasks")
    
    # Generate tasks for each action
    for action in actions:
        print(f"Generating task for action: {action['action_name']}")
        generate_task_from_action(action, output_dir)
