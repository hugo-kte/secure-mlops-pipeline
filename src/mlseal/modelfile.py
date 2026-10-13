"""Safe model format: plain JSON parameters, a strict loader and a numpy-only runtime.

Loading a model never executes code. The file is data (numbers and short strings), parsed
by `json` with duplicate keys, NaN and Infinity rejected, then checked against a closed
schema: exact key sets, bounded sizes, finite values, feature list identical to the
contract, and tree structures that provably terminate (every child index is greater than
its parent's). Anything unexpected is refused rather than ignored.

The serving path imports this module, `numpy` and the standard library; never pickle,
joblib, skops or scikit-learn (a test enforces it).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from mlseal.schema import FEATURE_NAMES, N_FEATURES

FORMAT = "mlseal.model/v1"
MAX_MODEL_BYTES = 16 * 1024 * 1024
MAX_TREES = 2000
MAX_NODES = 4096

_TOP_KEYS = {"format", "name", "version", "kind", "features", "positive_class", "threshold"}
_TREE_KEYS = {"feature", "threshold", "left", "right", "value"}


class ModelFormatError(ValueError):
    """The bytes are not a valid model file; nothing was loaded."""


@dataclass(frozen=True)
class Tree:
    feature: np.ndarray  # int (n_nodes,), -1 on leaves
    threshold: np.ndarray  # float (n_nodes,)
    left: np.ndarray  # int (n_nodes,), -1 on leaves
    right: np.ndarray  # int (n_nodes,), -1 on leaves
    value: np.ndarray  # float (n_nodes,), used on leaves
    depth: int

    def predict(self, X32: np.ndarray) -> np.ndarray:
        rows = np.arange(X32.shape[0])
        node = np.zeros(X32.shape[0], dtype=np.intp)
        for _ in range(self.depth):
            internal = self.left[node] >= 0
            feat = np.where(internal, self.feature[node], 0)
            go_left = X32[rows, feat] <= self.threshold[node]
            node = np.where(internal, np.where(go_left, self.left[node], self.right[node]), node)
        return self.value[node]


@dataclass(frozen=True)
class Model:
    name: str
    version: str
    kind: str
    threshold: float
    params: dict[str, Any]
    trees: tuple[Tree, ...] = ()

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != N_FEATURES:
            raise ValueError(f"expected an array of shape (n, {N_FEATURES})")
        if self.kind == "logistic_regression":
            return X @ np.asarray(self.params["coef"]) + self.params["intercept"]
        # Same order of operations as scikit-learn: init, then one stage at a time, with the
        # features compared in float32 like its tree code.
        X32 = X.astype(np.float32)
        raw = np.full(X.shape[0], self.params["init"], dtype=np.float64)
        lr = self.params["learning_rate"]
        for tree in self.trees:
            raw += lr * tree.predict(X32)
        return raw

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability of the positive class (phishing)."""
        return 1.0 / (1.0 + np.exp(-self.decision_function(X)))


# --- parsing --------------------------------------------------------------------------------


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise ModelFormatError(f"duplicate key {key!r}")
        obj[key] = value
    return obj


def _reject_constant(name: str) -> None:
    raise ModelFormatError(f"non-finite number {name} is not allowed")


def _expect_keys(obj: Any, keys: set[str], where: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ModelFormatError(f"{where}: expected an object")
    if set(obj) != keys:
        raise ModelFormatError(
            f"{where}: unexpected keys {sorted(set(obj) - keys)}, missing {sorted(keys - set(obj))}"
        )
    return obj


def _number(value: Any, where: str) -> float:
    try:
        ok = not isinstance(value, bool) and isinstance(value, int | float)
        ok = ok and math.isfinite(value)
    except OverflowError:  # an integer too large for a float
        ok = False
    if not ok:
        raise ModelFormatError(f"{where}: expected a finite number")
    return float(value)


def _int_list(value: Any, n: int, where: str) -> np.ndarray:
    if (
        not isinstance(value, list)
        or len(value) != n
        or not all(isinstance(v, int) and not isinstance(v, bool) for v in value)
        or not all(-1 <= v <= MAX_NODES for v in value)
    ):
        raise ModelFormatError(f"{where}: expected {n} integers within -1..{MAX_NODES}")
    return np.asarray(value, dtype=np.intp)


def _float_list(value: Any, n: int, where: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != n:
        raise ModelFormatError(f"{where}: expected {n} numbers")
    return np.asarray([_number(v, where) for v in value], dtype=np.float64)


def _string(value: Any, where: str, max_len: int = 64) -> str:
    if not isinstance(value, str) or not 0 < len(value) <= max_len or not value.isprintable():
        raise ModelFormatError(f"{where}: expected a short printable string")
    return value


def _tree(obj: Any, where: str) -> Tree:
    obj = _expect_keys(obj, _TREE_KEYS, where)
    n = len(obj["left"]) if isinstance(obj["left"], list) else 0
    if not 1 <= n <= MAX_NODES:
        raise ModelFormatError(f"{where}: node count must be within 1..{MAX_NODES}")
    feature = _int_list(obj["feature"], n, f"{where}.feature")
    left = _int_list(obj["left"], n, f"{where}.left")
    right = _int_list(obj["right"], n, f"{where}.right")
    threshold = _float_list(obj["threshold"], n, f"{where}.threshold")
    value = _float_list(obj["value"], n, f"{where}.value")
    depth = np.zeros(n, dtype=np.intp)
    parents = np.zeros(n, dtype=np.intp)
    for i in range(n):
        if left[i] == -1 and right[i] == -1:
            if feature[i] != -1:
                raise ModelFormatError(f"{where}: leaf {i} must have feature -1")
            continue
        if not 0 <= feature[i] < N_FEATURES:
            raise ModelFormatError(f"{where}: node {i} has feature index {feature[i]}")
        # Children strictly after the parent: no cycles, so traversal always terminates.
        if not (i < left[i] < n and i < right[i] < n and left[i] != right[i]):
            raise ModelFormatError(f"{where}: node {i} has invalid children")
        parents[left[i]] += 1
        parents[right[i]] += 1
        depth[left[i]] = depth[right[i]] = depth[i] + 1
    # A proper tree: every node but the root has exactly one parent, so `depth` is exact
    # and bounds the number of traversal steps.
    if parents[0] != 0 or (n > 1 and not np.all(parents[1:] == 1)):
        raise ModelFormatError(f"{where}: nodes must form a single tree")
    return Tree(feature, threshold, left, right, value, depth=int(depth.max()))


def loads(data: bytes) -> Model:
    """Parse and validate a model file. Raises ModelFormatError on any deviation."""
    if len(data) > MAX_MODEL_BYTES:
        raise ModelFormatError(f"model file larger than {MAX_MODEL_BYTES} bytes")
    try:
        obj = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ModelFormatError(f"not a JSON model file: {exc}") from exc

    if not isinstance(obj, dict) or obj.get("format") != FORMAT:
        raise ModelFormatError(f"unsupported format (expected {FORMAT!r})")
    kind = obj.get("kind")
    if kind not in ("logistic_regression", "gradient_boosting"):
        raise ModelFormatError(f"unsupported model kind {kind!r}")
    obj = _expect_keys(obj, _TOP_KEYS | {"params"}, "model")
    if obj["features"] != list(FEATURE_NAMES):
        raise ModelFormatError("feature list does not match the feature contract")
    if obj["positive_class"] != "phishing":
        raise ModelFormatError("positive class must be 'phishing'")
    threshold = _number(obj["threshold"], "threshold")
    if not 0.0 < threshold < 1.0:
        raise ModelFormatError("threshold must be within (0, 1)")
    name = _string(obj["name"], "name")
    version = _string(obj["version"], "version")

    if kind == "logistic_regression":
        p = _expect_keys(obj["params"], {"coef", "intercept"}, "params")
        params = {
            "coef": _float_list(p["coef"], N_FEATURES, "params.coef").tolist(),
            "intercept": _number(p["intercept"], "params.intercept"),
        }
        return Model(name, version, kind, threshold, params)

    p = _expect_keys(obj["params"], {"init", "learning_rate", "trees"}, "params")
    trees_raw = p["trees"]
    if not isinstance(trees_raw, list) or not 1 <= len(trees_raw) <= MAX_TREES:
        raise ModelFormatError(f"params.trees: expected 1..{MAX_TREES} trees")
    trees = tuple(_tree(t, f"trees[{i}]") for i, t in enumerate(trees_raw))
    params = {
        "init": _number(p["init"], "params.init"),
        "learning_rate": _number(p["learning_rate"], "params.learning_rate"),
    }
    return Model(name, version, kind, threshold, params, trees)


def dumps(model: dict[str, Any]) -> bytes:
    """Canonical encoding: sorted keys, no whitespace, no NaN; round-trips through `loads`."""
    text = json.dumps(model, sort_keys=True, separators=(",", ":"), allow_nan=False)
    loads(text.encode("utf-8"))  # never write a file the loader would refuse
    return text.encode("utf-8") + b"\n"
