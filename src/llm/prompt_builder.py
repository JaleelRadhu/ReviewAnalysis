from typing import List, Dict
from jinja2 import Template
import os

from src.util.config import load_config

cfg = load_config()  # load config
PROMPT_DIR = cfg["paths"]["prompt_templates_dir"]  # get template folder from cfg


def _load_template(file_name: str) -> Template:
    """Load a Jinja2 template from the prompt_templates folder specified in cfg."""
    file_path = os.path.join(PROMPT_DIR, file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        return Template(f.read())


# Preload templates
_iteration_template = _load_template("iteration_template.md")
_demo_template = _load_template("demo_block.md")

def render_demos(demos: list[dict]) -> str: ## this handles zero-shot too
    """
    Render the demos block (few-shot examples).
    Each demo must be a dict with keys: demo_text, demo_label
    """
    if not demos:
        return ""  # zero-shot: nothing added
    return _demo_template.render(demos=demos)


def build_iteration_prompt(
    demos: List[Dict[str, str]],
    question: str,
    answer: str,
    aspect: str,
    opinion: str,
    current_label: str,
    candidate_set: List[str],
) -> str:
    """
    Render the iterative classification prompt.
    Args:
        demos: list of {"demo_text": ..., "demo_label": ...}
        test_text: the input to classify
        current_label: current parent label
        candidate_set: list of candidate labels (strings)
        instruction_header: optional instruction
    """
    # Render demo examples (zero-shot handled inside render_demos)
    demos_block = render_demos(demos)
    current_label = current_label.split("/")[-1]

    candidate_set = [label.split("/")[-1] for label in candidate_set]
    candidate_list = "\n".join(f"{i+1}) {label}" for i, label in enumerate(candidate_set))
    # Format candidate list as "1) A\n2) B..."
    

    # Render final iteration prompt
    return _iteration_template.render(
        question=question,
        answer=answer,
        aspect=aspect,
        opinion=opinion,
        current_label=current_label,
        candidate_list=candidate_list,
        demos_block=demos_block
    )