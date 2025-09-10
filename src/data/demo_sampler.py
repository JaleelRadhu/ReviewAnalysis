# src/data/demo_sampler.py
import random
from typing import List, Dict, Optional

class DemoSampler:
    def __init__(self, train_data: Optional[List[Dict]] = None, synth_func=None):
        """
        Args:
            train_data: list of dicts with keys:
                - "text": text of training example
                - "label_path": full path string (e.g., "Course Material > Lecture Slides")
            synth_func: optional callable for synthetic demo generation
                Takes label_path and k, returns list of {"demo_text", "demo_label"}
        """
        self.train_data = train_data or []
        self.synth_func = synth_func

    def select_demos(
        self, k: int, current_label = None, candidate_set = None,  ## PROPERLY WRITE THIS PART DEPENDING ON LATER IMPLEMENTATION DESIGN OF SAMPLER
    ) -> List[Dict[str, str]]:
        """
        Select K demonstrations for current label and candidate set.
        Returns list of dicts with keys:
            - demo_text
            - demo_label
            - current_label
        """
        if k == 0:
            return []  # zero-shot

        # 1. Filter train_data for examples whose next label is in candidate_set
        matching_examples = [
            ex for ex in self.train_data
            if ex.get("current_label") == current_label and ex.get("demo_label") in candidate_set
        ]

        # 2. If enough examples, sample up to k
        if matching_examples:
            return random.sample(matching_examples, min(k, len(matching_examples)))

        # 3. If no real examples, generate synthetic demos if synth_func is provided
        # if self.synth_func:
        #     return self.synth_func(current_label, candidate_set, k)

        # 4. Fallback: zero-shot
        return []
