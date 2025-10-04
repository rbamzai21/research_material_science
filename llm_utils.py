import re
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

# Initialize client with API key
client = OpenAI(api_key=api_key)


def query_llm(prompt, model="gpt-4.1-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You help write LAMMPS input files."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def extract_blocks(text):
    tcl_match = re.search(r"```tcl\s*([\s\S]*?)```", text)
    lammps_match = re.search(r"```lammps\s*([\s\S]*?)```", text)
    colvars = tcl_match.group(1).strip() if tcl_match else None
    in_lammps = lammps_match.group(1).strip() if lammps_match else None
    return colvars, in_lammps
