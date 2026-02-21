import os
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

client = OpenAI(api_key=api_key)


def extract_python(text: str) -> str:
    """Extract Python from text, handling markdown code blocks."""
    match = re.search(r"```(?:python)?\s*([\s\S]*?)", text, re.IGNORECASE)
    if match:
        code = match.group(1)
    else:
        code = text
    
    code = code.strip()

    lines = code.splitlines()
    if lines and lines[0].strip().lower() == "python":
        lines = lines[1:]
    
    return "\n".join(lines).strip()

def query_llm(prompt, system_prompt="You are a scientific assistant for molecular simulation and materials science.", model="gpt-4.1-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()
