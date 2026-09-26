"""Guard the research-facing architecture rather than resurrecting the old Trainer API."""

import ast
from pathlib import Path

from query_expansion.rewards import sft_loss
from query_expansion.utils.backward import gradient

ROOT = Path(__file__).resolve().parents[1]


def test_entrypoints_keep_forward_loss_and_loops_visible():
    for name in ("sft", "rl"):
        tree = ast.parse((ROOT / "train" / f"{name}.py").read_text())
        assert any(isinstance(node, ast.For) for node in ast.walk(tree))
        assert any(
            isinstance(node, ast.FunctionDef) and node.name == "forward" for node in ast.walk(tree)
        )
        assert "criterion(" in ast.unparse(tree)
        assert "@gradient(engine)" in ast.unparse(tree)
        assert "Agents(" in ast.unparse(tree)
        assert "DatasetStream(" in ast.unparse(tree)
        assert "islice(" not in ast.unparse(tree)
        assert "engine.window(" not in ast.unparse(tree)


def test_domain_module_dependency_boundaries():
    root = ROOT / "src" / "query_expansion"
    modules = {"agents", "data_loader", "evaluation", "rewards", "retriever", "utils"}
    for owner in modules:
        for path in (root / owner).glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    parts = node.module.split(".")
                    if parts[0] == "query_expansion" and len(parts) > 1 and parts[1] in modules:
                        assert owner == parts[1] or (owner == "rewards" and parts[1] == "retriever")


def test_decorator_executes_exactly_user_forward(make_policy):
    agent, engine = make_policy()
    inputs, labels = agent.supervised([{"query": "car", "language": "en"}], ["battery"])
    events = []

    @gradient(engine)
    def research(inputs, labels):
        events.append("forward")
        return sft_loss(agent(inputs), labels) * 0.5

    with engine.window(2):
        research(inputs, labels)
        assert engine.global_step == 0
        assert any(p.grad is not None for p in agent.model.parameters())
        research(inputs, labels)
    assert engine.global_step == 1 and events == ["forward", "forward"]
