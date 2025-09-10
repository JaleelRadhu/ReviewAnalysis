# src/model/iterative.py
import sys
sys.path.append("/home/abdullahm/jaleel/Review_analysis")
from typing import List, Iterable
from src.llm.client import LLMClient
from src.data.loader import LabelTree
from src.data.demo_sampler import DemoSampler # DemoSampler not fully implemented yet, thus right k is kept 0 in config
from src.llm.prompt_builder import build_iteration_prompt
# from src.model.fallback import fallback_choose # not implemented yet
import re
class IterativePredictor:
    """
    Iterative Hierarchical Text Classifier.

    Walks down the label tree from the root to leaf,
    selecting a child category at each level using an LLM.
    """

    def __init__(self, llm, tree, sampler, cfg):
        self.llm = llm
        self.tree = tree
        self.sampler = sampler
        self.cfg = cfg
        self.k_demos = self.cfg.get("predictor", {}).get("k_demos", 3)  # default 3

        
    def predict_one(self, text: str):
        current = self.tree.root()
        path = []
        max_depth = self.cfg.get("predictor", {}).get("max_depth", 6)
        # print(f"question: {text['question']} answer: {text['answer']} aspect: {text['aspect']} opinion: {text['opinion']}")
        for level in range(max_depth):
            candidate_set = self.tree.children(current)
            if not candidate_set:
                break

            # ← use cfg value for number of demos
            demos = self.sampler.select_demos(self.k_demos, current, candidate_set) # BE CAREFUL WHEN IMPLEMENTING FULLY, BECAUSE RIGHT NOW, LABELS HAVE FULL PATH

            prompt = build_iteration_prompt(
                demos=demos,
                question=text['question'],
                answer=text['answer'],
                aspect=text['aspect'],
                opinion=text['opinion'],
                current_label=current,
                candidate_set=candidate_set,
            )
            
            current_splitted = current.split("/")[-1]
            candidate_set_splitted = [label.split("/")[-1] for label in candidate_set]

            response = self.llm.generate(prompt, temperature=self.cfg["llm"]["temperature"])
            print("\n------------------- LEVEL DONE ------------------\n")
            print(f"Level {level} | Current: {current_splitted} | Candidates: {candidate_set_splitted} | Response: {response}")
            chosen = self._parse_response(response, candidate_set, text)
            print("the llm chose:", chosen)
            path.append(chosen.split("/")[-1])
            print("Path so far:", path)
            current = chosen
            if self.tree.is_leaf(current):
                print("Reached leaf node.")
                break
        print("\n********************************\n")
        print("Final predicted path:", path)
        return path

    def predict_batch(self, texts: Iterable[str]) -> List[List[str]]:
        """
        Predict multiple texts.
        """
        return [self.predict_one(t) for t in texts]

    def _parse_response(self, response: str, candidate_set: List[str], text: str) -> str:
        """
        Map LLM output to a candidate label, with fallback.
        Handles numeric or string outputs.
        """
        out = response.strip()
        out = response.strip()

        # Try to find a numeric index anywhere in the string
        match = re.search(r"\b(\d+)\b", out)         # provisional parsing logic
        if match:
            idx = int(match.group(1)) - 1  # assuming options are 1-indexed
            if 0 <= idx < len(candidate_set):
                return candidate_set[idx]

        # Numeric index case (1-based)
        if out.isdigit():
            idx = int(out) - 1
            if 0 <= idx < len(candidate_set):
                return candidate_set[idx]
        else:
            print(" !!! Non-numeric output", "out:", out)
        # Exact match case
        for c in candidate_set:
            if out.lower() == c.lower() or out.strip() == c:
                return c
        print(" !!! No exact match for", "out:", out)
        # Fallback
        # return fallback_choose(candidate_set, out, text) IMPLEMENT FALLBACK

        return "null"  # (TEMPORARY FALLBACK) default to first if all else fails 
    
    
if __name__ == "__main__":
    import sys
    sys.path.append("/home/abdullahm/jaleel/Review_analysis")   
    import json
    from tqdm import tqdm
    from src.util.config import load_config, set_seed
    from src.llm.client import get_llm_client
    from src.data.loader import LabelTree
    from src.data.demo_sampler import DemoSampler
    # from src.model.iterative_predictor import IterativePredictor  # Assuming your class is here

    cfg = load_config()
    set_seed(cfg.get("seed", 42))

    # Load dataset
    test_path = cfg["paths"]["test_json"]
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Load label tree
    tree_path = cfg["paths"]["category_tree"]
    label_tree = LabelTree(tree_path)

    # Initialize demo sampler
    sampler = DemoSampler()  # implement select_demos later

    # Initialize LLM client
    llm_client = get_llm_client(cfg)

    # Initialize predictor
    predictor = IterativePredictor(llm_client, label_tree, sampler, cfg)

    # Predict on test dataset
    for item in tqdm(test_data, desc="Predicting"):
        print("========================================")
        print("\n\n")
        print("=========================================")
        path = predictor.predict_one(item)
        # Join path into full category path string
        item["predicted_full_category_path"] = " > ".join(["root"] + path)

    # Save predictions
    output_path = test_path.replace(".json", "_predicted.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"Predictions saved to {output_path}")