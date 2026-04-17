from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F


def orthogonal_rank_one_edit(
    weight: torch.Tensor,
    edit_key: torch.Tensor,
    desired_value: torch.Tensor,
    preserve_keys: torch.Tensor,
    ridge: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a local rank-one edit while nulling directions used by preserve_keys.

    A Transformer MLP output projection can be viewed as y = W h.  We want
    W' h_edit = desired_value, while W' h_i = W h_i for preserved prompts.
    Projecting the edit key away from preserved keys gives a direction that
    produces zero first-order change on the preserved set.
    """

    # Project edit_key to the orthogonal complement of the preserved subspace.
    # preserve_keys shape: [n_preserve, d_model].
    hp = preserve_keys.T
    gram = hp.T @ hp + ridge * torch.eye(hp.size(1), device=hp.device)
    projection = hp @ torch.linalg.solve(gram, hp.T @ edit_key)
    edit_dir = edit_key - projection
    denom = torch.dot(edit_dir, edit_key).clamp_min(1e-8)

    residual = desired_value - weight @ edit_key
    delta = torch.outer(residual, edit_dir) / denom
    return weight + delta, delta


def classify(weight: torch.Tensor, keys: torch.Tensor, labels: list[str]) -> list[str]:
    logits = keys @ weight.T
    return [labels[i] for i in logits.argmax(dim=-1).tolist()]


def main() -> None:
    torch.manual_seed(7)
    d_model = 32
    labels = ["Paris", "Berlin", "Rome", "Madrid", "Lyon"]
    facts = [
        "France capital",
        "Germany capital",
        "Italy capital",
        "Spain capital",
    ]
    true_classes = torch.tensor([0, 1, 2, 3])

    keys = F.normalize(torch.randn(len(facts), d_model), dim=-1)
    values = F.one_hot(true_classes, num_classes=len(labels)).float() * 6.0

    # Closed-form least-squares associative memory: W maps fact-key to answer logits.
    weight = values.T @ torch.linalg.pinv(keys.T)
    before = classify(weight, keys, labels)

    edit_idx = 0
    desired = F.one_hot(torch.tensor(4), num_classes=len(labels)).float() * 6.0
    preserve = keys[1:]
    edited_weight, delta = orthogonal_rank_one_edit(weight, keys[edit_idx], desired, preserve)
    after = classify(edited_weight, keys, labels)

    old_logits = keys @ weight.T
    new_logits = keys @ edited_weight.T
    drift = (new_logits[1:] - old_logits[1:]).abs().max().item()

    result = {
        "method": "LORO: Local Orthogonal Rank-One Edit",
        "edit": {"fact": facts[edit_idx], "old_answer": before[edit_idx], "new_answer": after[edit_idx]},
        "preserved_facts": [
            {"fact": fact, "before": b, "after": a}
            for fact, b, a in zip(facts[1:], before[1:], after[1:])
        ],
        "max_preserved_logit_drift": drift,
        "delta_rank_upper_bound": int(torch.linalg.matrix_rank(delta).item()),
    }

    out = Path("outputs/knowledge_edit_demo.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
