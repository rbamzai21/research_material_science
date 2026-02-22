"""Persistent search state for the MCTS formula search."""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class FormulaNode:
    id: str
    parent_id: str | None
    code: str
    description: str
    formula: str = ""
    accuracy: float = 0.0
    metrics: dict = field(default_factory=dict)
    plot_path: str = ""
    visit_count: int = 0
    total_reward: float = 0.0
    children_ids: list[str] = field(default_factory=list)
    depth: int = 0


@dataclass
class SearchState:
    nodes: dict[str, FormulaNode] = field(default_factory=dict)
    root_children: list[str] = field(default_factory=list)
    budget_used: int = 0
    total_llm_calls: int = 0
    debug_calls: int = 0

    def add_node(self, node: FormulaNode) -> None:
        self.nodes[node.id] = node
        if node.parent_id is None:
            if node.id not in self.root_children:
                self.root_children.append(node.id)
        else:
            parent = self.nodes[node.parent_id]
            if node.id not in parent.children_ids:
                parent.children_ids.append(node.id)

    def recompute_ranks(self) -> dict[str, float]:
        sorted_nodes = sorted(self.nodes.items(), key=lambda x: x[1].accuracy, reverse=True)
        total = len(sorted_nodes)
        if total == 0:
            return {}
        return {nid: 1.0 - (rank / total) for rank, (nid, _node) in enumerate(sorted_nodes)}

    def top_k(self, k: int = 10) -> list[FormulaNode]:
        return sorted(self.nodes.values(), key=lambda n: n.accuracy, reverse=True)[:k]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": {nid: asdict(n) for nid, n in self.nodes.items()},
            "root_children": self.root_children,
            "budget_used": self.budget_used,
            "total_llm_calls": self.total_llm_calls,
            "debug_calls": self.debug_calls,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        log.info("State saved to %s (%d nodes)", path, len(self.nodes))

    @classmethod
    def load(cls, path: Path) -> "SearchState":
        with open(Path(path)) as f:
            data = json.load(f)

        state = cls()
        state.root_children = data["root_children"]
        state.budget_used = data["budget_used"]
        state.total_llm_calls = data["total_llm_calls"]
        state.debug_calls = data["debug_calls"]

        for nid, ndata in data["nodes"].items():
            state.nodes[nid] = FormulaNode(**ndata)

        log.info("State loaded from %s (%d nodes)", path, len(state.nodes))
        return state
