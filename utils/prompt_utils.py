from hydra.utils import to_absolute_path
import json


def load_prompts():
    prompt_path = to_absolute_path("prompts/pic_template.json")
    with open(prompt_path, "r") as f:
        return json.load(f)


def format_claims(claims):
    return "\n".join(f"- {item}" for item in claims)
