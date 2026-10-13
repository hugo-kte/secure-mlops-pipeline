"""The JSON model format: exact parity with scikit-learn, and a loader that refuses junk."""

import copy
import json

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from conftest import DATA
from mlseal.data import deduplicate, read_arff
from mlseal.export import ExportError, export, to_model_dict
from mlseal.modelfile import MAX_MODEL_BYTES, MAX_NODES, ModelFormatError, dumps, loads
from mlseal.schema import FEATURE_NAMES


@pytest.fixture(scope="module")
def xy():
    ds = read_arff(DATA)
    X, y, _ = deduplicate(ds.X, ds.y)
    return X, y


@pytest.fixture(scope="module")
def gbm(xy):
    X, y = xy
    return GradientBoostingClassifier(n_estimators=60, max_depth=4, random_state=0).fit(X, y)


@pytest.fixture(scope="module")
def model_dict(gbm, xy):
    return to_model_dict(gbm, name="m", version="1.0.0", threshold=0.5, X_check=xy[0][:200])


def test_boosting_export_matches_scikit_learn(gbm, xy):
    data, diff = export(gbm, name="m", version="1.0.0", threshold=0.5, X_check=xy[0])
    assert diff < 1e-12
    model = loads(data)
    assert np.array_equal(model.predict_proba(xy[0]) >= 0.5, gbm.predict_proba(xy[0])[:, 1] >= 0.5)


def test_logistic_export_matches_scikit_learn(xy):
    X, y = xy
    est = LogisticRegression(max_iter=2000).fit(X, y)
    data, diff = export(est, name="lr", version="1.0.0", threshold=0.5, X_check=X)
    assert diff < 1e-12
    assert loads(data).kind == "logistic_regression"


def test_unsupported_estimator_is_refused(xy):
    from sklearn.tree import DecisionTreeClassifier

    est = DecisionTreeClassifier(max_depth=2).fit(*xy)
    with pytest.raises(ExportError, match="no exporter"):
        to_model_dict(est, name="t", version="1.0.0", threshold=0.5, X_check=xy[0])


def test_dumps_is_canonical_and_deterministic(model_dict):
    assert dumps(model_dict) == dumps(copy.deepcopy(model_dict))
    assert dumps(model_dict).endswith(b"\n")


def _broken(model_dict, mutate):
    d = copy.deepcopy(model_dict)
    mutate(d)
    return json.dumps(d).encode()


def _tree(d):
    return d["params"]["trees"][0]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(format="pickle"), "unsupported format"),
        (lambda d: d.update(kind="neural_net"), "unsupported model kind"),
        (lambda d: d.update(extra="x"), "unexpected keys"),
        (lambda d: d.pop("threshold"), "missing"),
        (lambda d: d.update(features=list(reversed(FEATURE_NAMES))), "feature list"),
        (lambda d: d.update(positive_class="legitimate"), "positive class"),
        (lambda d: d.update(threshold=1.5), "threshold"),
        (lambda d: d.update(name="x" * 500), "name"),
        (lambda d: d["params"].update(init=True), "finite number"),
        (lambda d: d["params"].update(init=10**400), "finite number"),
        (lambda d: d["params"].update(trees=[]), "trees"),
        (lambda d: _tree(d)["feature"].__setitem__(0, 99), "feature index"),
        (lambda d: _tree(d)["left"].__setitem__(0, 0), "invalid children"),  # self-loop
        (lambda d: _tree(d)["right"].__setitem__(0, _tree(d)["left"][0]), "invalid children"),
        (lambda d: _tree(d)["left"].__setitem__(0, 10**9), "integers within"),
        (lambda d: _tree(d)["threshold"].pop(), "numbers"),
        (lambda d: _tree(d)["value"].__setitem__(-1, "0.1"), "finite number"),
        (lambda d: _tree(d).update(extra=[]), "unexpected keys"),
    ],
)
def test_loader_refuses_malformed_models(model_dict, mutate, message):
    with pytest.raises(ModelFormatError, match=message):
        loads(_broken(model_dict, mutate))


def _with_tree(model_dict, **tree):
    d = copy.deepcopy(model_dict)
    d["params"]["trees"] = [tree]
    return json.dumps(d).encode()


def test_loader_refuses_shared_children(model_dict):
    """Nodes 3 and 4 have two parents: a DAG, not a tree. Its depth bound would be wrong."""
    raw = _with_tree(
        model_dict,
        feature=[0, 1, 2, -1, -1],
        threshold=[0.5, 0.5, 0.5, 0.0, 0.0],
        left=[1, 3, 3, -1, -1],
        right=[2, 4, 4, -1, -1],
        value=[0.0, 0.0, 0.0, 0.1, -0.1],
    )
    with pytest.raises(ModelFormatError, match="single tree"):
        loads(raw)


def test_loader_refuses_unreachable_nodes(model_dict):
    raw = _with_tree(
        model_dict, feature=[-1, -1], threshold=[0.0, 0.0], left=[-1, -1], right=[-1, -1],
        value=[0.1, 0.2],
    )  # fmt: skip
    with pytest.raises(ModelFormatError, match="single tree"):
        loads(raw)


def test_minimal_valid_tree_loads_and_predicts(model_dict):
    raw = _with_tree(
        model_dict, feature=[7, -1, -1], threshold=[0.5, 0.0, 0.0], left=[1, -1, -1],
        right=[2, -1, -1], value=[0.0, -1.0, 1.0],
    )  # fmt: skip
    model = loads(raw)
    X = np.zeros((2, 30))
    X[1, 7] = 1  # SSLfinal_State above the threshold goes right
    raw_scores = model.decision_function(X) - model.params["init"]
    assert raw_scores.tolist() == [-model.params["learning_rate"], model.params["learning_rate"]]


@pytest.mark.parametrize(
    "raw",
    [
        b'{"format": "mlseal.model/v1", "format": "x"}',  # duplicate key
        b'{"format": "mlseal.model/v1", "threshold": NaN}',
        b'{"format": "mlseal.model/v1", "threshold": Infinity}',
        b"\x80\x04\x95\x00",  # a pickle header is just invalid JSON here
        b"[" * 100_000,
        "﻿{}".encode(),
    ],
)
def test_loader_refuses_non_json_or_ambiguous_json(raw):
    with pytest.raises(ModelFormatError):
        loads(raw)


def test_loader_refuses_oversized_files():
    with pytest.raises(ModelFormatError, match="larger than"):
        loads(b" " * (MAX_MODEL_BYTES + 1))


def test_loader_refuses_too_many_nodes(model_dict):
    def grow(d):
        n = MAX_NODES + 1
        _tree(d).update(feature=[-1] * n, threshold=[0.0] * n, left=[-1] * n, right=[-1] * n)
        _tree(d)["value"] = [0.0] * n

    with pytest.raises(ModelFormatError, match="node count"):
        loads(_broken(model_dict, grow))


def test_runtime_rejects_wrong_input_shape(model_dict):
    with pytest.raises(ValueError, match="shape"):
        loads(dumps(model_dict)).predict_proba(np.zeros((2, 29)))
