import json
from typing import List, Dict, Any, Union


class LabelTree:
    """
    A wrapper around the category_tree.json that provides
    easy access to children, labels, and paths.
    Nodes are uniquely identified by their full path, e.g.:
        root/Structure/Difficulty
        root/Evaluation/Quality of Questions/Difficulty
    """

    def __init__(self, tree_json: Union[str, Dict[str, Any]]):
        """
        Args:
            tree_json: Either a file path to JSON, or parsed JSON object.
        """
        if isinstance(tree_json, str):  # assume it's a path
            with open(tree_json, "r", encoding="utf-8") as f:
                self.tree = json.load(f)
        else:
            self.tree = tree_json

        self.node_to_children: Dict[str, List[str]] = {}
        self.node_to_label: Dict[str, str] = {}

        # Build index recursively
        self._root_id = self._build_index(self.tree, parent_path=None)

    def _build_index(self, node: Union[Dict[str, Any], str], parent_path: str) -> str:
        """
        Recursively assign a full-path ID to each node.
        Returns the node_id created.
        """
        if isinstance(node, str):
            node_id = f"{parent_path}/{node}" if parent_path else node
            self.node_to_label[node_id] = node
            self.node_to_children[node_id] = []
            return node_id

        label = node["text"]
        node_id = f"{parent_path}/{label}" if parent_path else label
        self.node_to_label[node_id] = label

        child_ids = []
        for child in node.get("children", []):
            child_id = self._build_index(child, parent_path=node_id)
            child_ids.append(child_id)

        self.node_to_children[node_id] = child_ids
        return node_id

    # ---------------- Public API ---------------- #

    def root(self) -> str:
        """Return the root node ID (full path)."""
        return self._root_id

    def children(self, node_id: str) -> List[str]:
        """Return child node IDs of a given node ID."""
        return self.node_to_children.get(node_id, [])

    def label(self, node_id: str) -> str:
        """Return the human-readable label of a node ID."""
        return self.node_to_label[node_id]

    def is_leaf(self, node_id: str) -> bool:
        """Check if a node is a leaf (no children)."""
        return len(self.children(node_id)) == 0

    def all_nodes(self) -> List[str]:
        """Return all node IDs in the tree."""
        return list(self.node_to_children.keys())

    def get_path_labels(self, node_id: str) -> List[str]:
        """Return the list of labels from root to this node (path as labels)."""
        return node_id.split("/")

    def __repr__(self):
        return f"<LabelTree root={self._root_id} nodes={len(self.node_to_children)}>"



if __name__ == "__main__":
    import sys
    # print(sys.path)
    sys.path.append("/home/abdullahm/jaleel/Review_analysis")
    from src.util.config import load_config
    cfg = load_config()
    # Example usage
    tree = LabelTree(cfg["paths"]["category_tree"])
    print(tree)
    print("Root:", tree.root())
    for child in tree.children(tree.root()):
        print("Child:", child, "Label:", tree.label(child), "Is leaf:", tree.is_leaf(child))
        for grandchild in tree.children(child):
            print("  Grandchild:", grandchild, "Label:", tree.label(grandchild), "Is leaf:", tree.is_leaf(grandchild))  
    