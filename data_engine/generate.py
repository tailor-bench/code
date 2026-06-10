import os
from openai import OpenAI

def generate(prompt, model_name="gpt-5", port=1206):
    if "gpt" in model_name.lower():
        return gpt_generate(prompt, model_name)
    elif "qwen" in model_name.lower():
        return qwen3_generate(prompt, model_name, port)
    else:
        raise ValueError(f"Model {model_name} not supported")

def gpt_generate(prompt, model_name="gpt-5"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    chat_response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}])
    response = chat_response.choices[0].message.content.strip()
    reasoning = ""
    return response, reasoning

def qwen3_generate(prompt, port=1206, model_name="Qwen/Qwen3-8B", temperature=1.0): 
    openai_api_key = "EMPTY"
    openai_api_base = f"http://localhost:{port}/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    chat_response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    response = chat_response.choices[0].message.content.strip()
    if hasattr(chat_response.choices[0].message, 'reasoning_content') and chat_response.choices[0].message.reasoning_content is not None:
        reasoning = chat_response.choices[0].message.reasoning_content.strip()
    else:
        reasoning = ""
    return response, reasoning