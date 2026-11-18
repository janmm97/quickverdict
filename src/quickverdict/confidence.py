"""Confidence formulas, identical to TypeSafe's published definitions (docs: /confidence)."""


def choice_confidence(probs: list[float]) -> float:
    n = len(probs)
    if n < 2:
        return 1.0
    return max(0.0, min(1.0, (max(probs) - 1 / n) / (1 - 1 / n)))


def score_confidence(probs: list[float]) -> float:
    n = len(probs)
    if n < 2:
        return 1.0
    m = probs.index(max(probs))
    spread = sum(p * abs(i - m) for i, p in enumerate(probs))
    even_spread = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1 - spread / even_spread)
