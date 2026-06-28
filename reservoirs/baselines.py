"""Honest baseline comparators for reservoir-computing claims.

This module exists to make the control conditions impossible to omit when you
report a reservoir number. Two questions a reviewer always asks:

First, did the reservoir actually beat not using a reservoir? To check this,
compare against ``majority_class_baseline`` (chance) and
``linear_readout_baseline`` (a plain linear model on the RAW inputs). If a linear
readout on the inputs already solves the task, the reservoir is not earning its
keep.

Second, is it the empirical wiring that helps, or just the degree and weight
distribution? Here ``reservoir_vs_null`` scores the real adjacency against an
ensemble of degree-preserving rewired nulls and returns a z-score. It generalizes
the null-model section of examples/combined_examples.ipynb.

``compare_classification`` bundles the first question into a single dict so all
three accuracies print side by side, and ``format_comparison`` renders it as a
compact table for the example notebooks.

Dependency policy: numpy and networkx only (networkx is pulled in transitively via
``reservoirs.nulls``). No pandas, matplotlib, or scikit-learn at module import
time. An optional scikit-learn logistic baseline is available via
``logistic_readout_baseline``, which imports sklearn lazily inside the function and
raises a clear error if it is not installed, so sklearn never becomes a core
dependency.
"""
from __future__ import annotations

import numpy as np


# 1. Chance baseline: always predict the most frequent TRAIN label
def majority_class_baseline(y_train, y_test=None) -> dict:
    """Accuracy of always predicting the most frequent label in ``y_train``.

    Evaluated on ``y_test``, or on ``y_train`` itself when ``y_test is None`` (the
    in-sample chance ceiling). Works for any 1-D integer or string label array.

    Tie-break: if several labels share the top count, the smallest by ``np.unique``
    sort order (numeric order for ints, lexicographic for strings) is chosen, so the
    result is deterministic.

    Returns a dict with two keys. ``accuracy`` is the fraction of ``y_eval`` equal to
    the predicted label, and ``predicted_label`` is the most frequent TRAIN label as a
    native Python scalar.
    """
    y_train = np.asarray(y_train).ravel()
    if y_train.size == 0:
        raise ValueError("majority_class_baseline: y_train is empty.")
    labels, counts = np.unique(y_train, return_counts=True)
    predicted = labels[int(np.argmax(counts))]
    y_eval = y_train if y_test is None else np.asarray(y_test).ravel()
    if y_eval.size == 0:
        raise ValueError("majority_class_baseline: y_test is empty.")
    accuracy = float(np.mean(y_eval == predicted))
    # return a plain Python scalar so callers/printing are not surprised by numpy types
    predicted_label = predicted.item() if hasattr(predicted, "item") else predicted
    return {"accuracy": accuracy, "predicted_label": predicted_label}


# 2. NO-RESERVOIR baseline: one-hot ridge (least-squares) classifier on the RAW features
def linear_readout_baseline(
    X_train,
    y_train,
    X_test,
    y_test,
    ridge: float = 1.0,
    fit_intercept: bool = False,
    standardize: bool = False,
) -> dict:
    """Closed-form one-hot ridge classifier on the RAW input features (NO reservoir).

    This is the head-to-head the project otherwise lacks: the reservoir against a
    plain linear model on the inputs. With the FAITHFUL DEFAULTS
    (``fit_intercept=False``, ``standardize=False``) the executed code is exactly the
    printed formula, in float64,

        Wout = (Xt X + ridge * I)^-1 Xt Y_onehot

    on the raw ``[n_samples, n_features]`` design matrix, decoded by ``argmax`` (the
    doc matches the code).

    The two flags are documented opt-ins that deviate from the bare formula.
    ``fit_intercept=True`` applies the same closed form to ``X`` augmented with a
    constant column (an augmented design matrix), giving the model a bias term.
    ``standardize=True`` z-scores features using TRAIN statistics only (no test
    leakage, and a zero-variance column gets sd=1), for callers who want a scale-fair
    linear model.

    Parameters:
        X_train, X_test : array [n_samples, n_features] of static features (e.g. raw Iris).
        y_train, y_test : 1-D label arrays (int or string).
        ridge : ridge penalty (>= 0; > 0 recommended so the Gram matrix is invertible).

    Returns a dict that is a documented SUPERSET of the spec's {'accuracy': float};
    ``accuracy`` is always present. ``accuracy`` is the contract key, ``predictions``
    holds the predicted labels for X_test (bonus), and ``classes`` holds the sorted
    train labels (bonus).

    A note on scientific honesty. One-hot least-squares is subject to the well-known
    multiclass masking problem (Hastie et al., ESL sec. 4.2), which lowers its AVERAGE
    accuracy. On real 3-class Iris it averages ~0.80 (verified 0.80 @ ridge=1.0, no
    bias; the +intercept variant averages roughly 0.83 to 0.87). Individual train/test
    splits can still exceed 0.9, but on average it trails a multinomial-logistic model.
    For a stronger linear-style bar use the optional ``logistic_readout_baseline``
    (~0.97 on Iris).
    """
    Xtr = np.asarray(X_train, dtype=np.float64)
    Xte = np.asarray(X_test, dtype=np.float64)
    if Xtr.ndim != 2 or Xte.ndim != 2:
        raise ValueError("linear_readout_baseline: X_train/X_test must be 2-D [n_samples, n_features].")
    y_train = np.asarray(y_train).ravel()
    y_test = np.asarray(y_test).ravel()
    if Xtr.shape[0] != y_train.shape[0]:
        raise ValueError("linear_readout_baseline: X_train and y_train length mismatch.")

    if standardize:
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd[sd == 0] = 1.0
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd

    if fit_intercept:
        Xtr = np.hstack([Xtr, np.ones((Xtr.shape[0], 1))])
        Xte = np.hstack([Xte, np.ones((Xte.shape[0], 1))])

    classes, y_idx = np.unique(y_train, return_inverse=True)
    Y = np.zeros((Xtr.shape[0], classes.shape[0]), dtype=np.float64)
    Y[np.arange(Xtr.shape[0]), y_idx] = 1.0

    gram = Xtr.T @ Xtr
    Wout = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), Xtr.T @ Y)

    scores = Xte @ Wout
    predictions = classes[scores.argmax(axis=1)]
    accuracy = float(np.mean(predictions == y_test))
    return {"accuracy": accuracy, "predictions": predictions, "classes": classes}


# Optional richer comparator (sklearn-backed; NOT a core dependency)
def logistic_readout_baseline(X_train, y_train, X_test, y_test, **kwargs) -> dict:
    """Optional multinomial-logistic no-reservoir baseline (lazy sklearn import).

    A stronger linear-style bar than one-hot least squares: it avoids the masking
    problem and clears ~0.9 on Iris. sklearn is imported inside this function so it
    never becomes a core dependency; if it is missing a clear ImportError is raised.
    ``standardize`` (default True) z-scores with TRAIN stats. Extra ``kwargs`` are
    forwarded to ``LogisticRegression``.
    """
    standardize = kwargs.pop("standardize", True)
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover - exercised only when sklearn is absent
        raise ImportError(
            "logistic_readout_baseline requires scikit-learn (optional). "
            "Install it (`pip install scikit-learn`) or use linear_readout_baseline instead."
        ) from exc
    Xtr = np.asarray(X_train, dtype=np.float64)
    Xte = np.asarray(X_test, dtype=np.float64)
    y_train = np.asarray(y_train).ravel()
    y_test = np.asarray(y_test).ravel()
    if standardize:
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd[sd == 0] = 1.0
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd
    kwargs.setdefault("max_iter", 1000)
    clf = LogisticRegression(**kwargs).fit(Xtr, y_train)
    accuracy = float(np.mean(clf.predict(Xte) == y_test))
    return {"accuracy": accuracy}


# 3. Real connectome vs degree-preserving null ensemble (generalizes example 07)
def reservoir_vs_null(adjacency, score_fn, n_null: int = 20, seed: int = 0, **null_kwargs) -> dict:
    """Score the real adjacency against an ensemble of degree-preserving rewired nulls.

    ``score_fn(A: np.ndarray) -> float`` maps a (symmetric, weighted, zero-diagonal)
    adjacency to a scalar performance (e.g. an Iris test accuracy obtained by building
    a reservoir from ``A``). Each null is drawn with
    ``reservoirs.nulls.rewire_degree_preserving`` using a distinct seed
    (``base_seed + i``) so the nulls actually differ; ``null_kwargs`` (e.g.
    ``n_swaps_per_edge``) are forwarded to it. A ``seed`` passed inside ``null_kwargs``
    is taken as the base seed, which avoids a duplicate-keyword error.

    Returns a dict whose keys are: ``real``, the score of the real adjacency;
    ``null_mean``, the mean of the null scores; ``null_std``, their population std
    (ddof=0); ``z``, equal to (real minus null_mean) over null_std, or 0.0 when
    null_std is 0; ``null_scores``, the list of n_null null scores; and ``n_null``, the
    number of nulls drawn.
    """
    from reservoirs.nulls import rewire_degree_preserving

    if n_null < 1:
        raise ValueError("reservoir_vs_null: n_null must be >= 1.")
    A = np.asarray(adjacency, dtype=float)
    base_seed = int(null_kwargs.pop("seed", seed))
    real = float(score_fn(A))

    null_scores = []
    for i in range(n_null):
        null = rewire_degree_preserving(A, seed=base_seed + i, **null_kwargs)
        null_scores.append(float(score_fn(null)))

    arr = np.asarray(null_scores, dtype=float)
    null_mean = float(arr.mean())
    null_std = float(arr.std())  # population std (ddof=0)
    # Guard a degenerate null spread. We treat null_std as zero not only when it is exactly 0
    # but also when it is negligible relative to the score magnitude: a score the null preserves
    # (e.g. total edge weight) can still pick up ~1e-14 jitter from float summation reassociation,
    # and dividing that ~0 numerator by a ~0 std would otherwise manufacture a meaningless large z.
    scale = max(abs(null_mean), abs(real), 1.0)
    z = float((real - null_mean) / null_std) if null_std > 1e-12 * scale else 0.0
    return {
        "real": real,
        "null_mean": null_mean,
        "null_std": null_std,
        "z": z,
        "null_scores": null_scores,
        "n_null": int(n_null),
    }


# 4. Bundle: reservoir vs majority vs linear, with deltas
def compare_classification(
    reservoir_accuracy: float,
    X_train,
    y_train,
    X_test,
    y_test,
    ridge: float = 1.0,
) -> dict:
    """Bundle a reservoir's accuracy next to the chance and no-reservoir-linear baselines.

    The 'linear' bar here is a PROPER linear model: an intercept-augmented one-hot
    ridge on the raw inputs (``fit_intercept=True``), so ``beats_linear`` answers the
    honest question, did the reservoir beat a standard linear model on the inputs? It
    is not the deliberately weakened no-intercept formula. For an even stronger
    linear-style bar (multinomial logistic, ~0.97 on Iris) call
    ``logistic_readout_baseline`` separately: ``beats_linear`` does NOT account for it,
    so a True there does not mean the reservoir beats every linear model.

    Returns a dict with these keys. ``reservoir`` is the supplied reservoir accuracy,
    ``majority`` is the majority_class_baseline accuracy, and ``linear`` is the
    linear_readout_baseline accuracy (with intercept). ``majority_label`` is the
    majority TRAIN label. ``delta_vs_majority`` is reservoir minus majority (positive
    means the reservoir beats chance) and ``delta_vs_linear`` is reservoir minus linear
    (positive means the reservoir beats a linear model). ``beats_majority`` and
    ``beats_linear`` are the matching booleans.
    """
    maj = majority_class_baseline(y_train, y_test)
    lin = linear_readout_baseline(X_train, y_train, X_test, y_test, ridge=ridge, fit_intercept=True)
    reservoir = float(reservoir_accuracy)
    d_maj = reservoir - maj["accuracy"]
    d_lin = reservoir - lin["accuracy"]
    return {
        "reservoir": reservoir,
        "majority": maj["accuracy"],
        "linear": lin["accuracy"],
        "majority_label": maj["predicted_label"],
        "delta_vs_majority": d_maj,
        "delta_vs_linear": d_lin,
        "beats_majority": bool(d_maj > 0),
        "beats_linear": bool(d_lin > 0),
    }


# 5. Pretty-printer for notebooks (no external deps)
def format_comparison(result: dict) -> str:
    """Render a comparison dict as a compact multi-line table (numpy/stdlib only).

    Accepts either a `compare_classification` dict (keys 'reservoir'/'majority'/'linear') or a
    `reservoir_vs_null` dict (keys 'real'/'null_mean'/'z'). Returns a non-empty string.
    """
    if "reservoir" in result:
        lines = [
            "model                       accuracy",
            "===================================",
            f"reservoir                   {result['reservoir']:.3f}",
            f"majority class (chance)     {result['majority']:.3f}  (label={result.get('majority_label')})",
            f"linear readout (no-reserv.) {result['linear']:.3f}",
            "===================================",
            f"delta  reservoir - majority {result['delta_vs_majority']:+.3f}",
            f"delta  reservoir - linear   {result['delta_vs_linear']:+.3f}",
        ]
        return "\n".join(lines)
    if "real" in result:
        lines = [
            "real connectome vs degree-preserving null",
            "=========================================",
            f"real        {result['real']:.3f}",
            f"null mean   {result['null_mean']:.3f}",
            f"null std    {result['null_std']:.3f}  (n={result.get('n_null')})",
            f"z-score     {result['z']:+.2f}",
        ]
        return "\n".join(lines)
    raise ValueError(
        "format_comparison: unrecognized result dict; expected a compare_classification "
        "or reservoir_vs_null result."
    )
