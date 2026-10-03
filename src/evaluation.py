"""Plots and metrics used by the original churn experiments."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score


def plot_confusion_matrix_tp_labels(
    y_test,
    y_pred,
    labels=(0, 1),
    class_names=None,
    title="Confusion Matrix",
    figsize=(7.5, 7.5),
    cmap=plt.cm.Reds,
):
    """
    Plots a 2x2 confusion matrix with TN/FP/FN/TP in each cell.

    Assumes sklearn's confusion_matrix layout (for labels ordered as [0, 1]):
        [[TN, FP],
         [FN, TP]]
    """
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_test, y_pred, labels=list(labels))

    if cm.shape != (2, 2):
        raise ValueError(
            f"Expected a 2x2 confusion matrix (binary classification), got shape {cm.shape}."
        )

    if class_names is None:
        class_names = [str(labels[0]), str(labels[1])]

    cell_labels = np.array([["TN", "FP"], ["FN", "TP"]])

    fig, ax = plt.subplots(figsize=figsize)
    ax.matshow(cm, cmap=cmap, alpha=0.3)

    for i in range(2):
        for j in range(2):
            ax.text(
                x=j,
                y=i,
                s=f"{cell_labels[i, j]}\n{cm[i, j]}",
                va="center",
                ha="center",
                fontsize=18,
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(
        [f"Predicted {class_names[0]}", f"Predicted {class_names[1]}"], fontsize=14
    )
    ax.set_yticklabels(
        [f"Actual {class_names[0]}", f"Actual {class_names[1]}"], fontsize=14
    )

    ax.set_xlabel("Predictions", fontsize=18)
    ax.set_ylabel("Actuals", fontsize=18)
    ax.set_title(title, fontsize=18)

    plt.tight_layout()
    plt.show()
    return cm


def plot_roc_curve(
    y_test,
    y_score=None,
    model=None,
    X_test=None,
    pos_label=1,
    title="ROC Curve",
    figsize=(7.5, 7.5),
):
    """
    Plot ROC curve for binary classification.

    Provide ONE of:
      - y_score: probability for the positive class OR decision scores (shape [n_samples])
      - model + X_test: will compute scores from model
    """
    y_test = np.asarray(y_test)

    if y_score is None:
        if model is None or X_test is None:
            raise ValueError("Provide either y_score, or (model and X_test).")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            if proba.shape[1] < 2:
                raise ValueError(
                    "predict_proba returned <2 columns; need binary probs with two columns."
                )
            y_score = proba[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            raise ValueError(
                "Model must implement predict_proba or decision_function, or pass y_score directly."
            )

    y_score = np.asarray(y_score)

    fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.set_title(f"{title} (AUC = {roc_auc:.3f})", fontsize=16)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

    return {"fpr": fpr, "tpr": tpr, "auc": roc_auc}


def plot_feature_importance(
    model,
    feature_names=None,
    top_n=20,
    title="Feature Importance",
    figsize=(10, 6),
    class_idx=0,
):
    """
    Plot feature importance for models that expose:
      - model.feature_importances_ (tree-based)
      - model.coef_ (linear models)

    For multiclass linear models (coef_.shape = [n_classes, n_features]),
    selects class_idx by default.
    """
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)

    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim == 2:  # multiclass or binary with shape (1, n_features)
            if coef.shape[0] == 1:
                coef = coef[0]
            else:
                if not (0 <= class_idx < coef.shape[0]):
                    raise ValueError(
                        f"class_idx must be in [0, {coef.shape[0]-1}] for this model."
                    )
                coef = coef[class_idx]
        importances = np.abs(coef)

    else:
        raise ValueError(
            "This model has neither feature_importances_ nor coef_. "
            "Use permutation importance (model-agnostic) or SHAP."
        )

    n_features = importances.shape[0]

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]
    else:
        if len(feature_names) != n_features:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) must match n_features ({n_features})."
            )

    idx = np.argsort(importances)[::-1][:top_n]
    top_importances = importances[idx]
    top_names = np.array(feature_names)[idx]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_importances)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()
    ax.set_xlabel("Importance", fontsize=14)
    ax.set_title(title, fontsize=16)

    plt.tight_layout()
    plt.show()

    return {"indices": idx, "importances": top_importances, "names": top_names.tolist()}


def evaluate_model(
    model,
    model_name,
    X_train,
    y_train,
    X_test,
    y_test,
    labels=(0, 1),
    class_names=("No", "Yes"),
    top_n=20,
    class_idx=0,
):
    # Train preds
    y_pred_train = model.predict(X_train)
    acc_train = accuracy_score(y_train, y_pred_train) * 100
    print(f"{model_name} - Training accuracy: {acc_train:.2f}%")

    # Test preds
    y_pred_test = model.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred_test) * 100
    print(f"{model_name} - Test accuracy: {acc_test:.2f}%")

    conf_matrix = confusion_matrix(y_test, y_pred_test, labels=list(labels))
    print(conf_matrix)

    plot_confusion_matrix_tp_labels(
        y_test,
        y_pred_test,
        labels=labels,
        class_names=class_names,
        title=f"Confusion Matrix ({model_name})",
    )

    roc_out = plot_roc_curve(
        y_test,
        model=model,
        X_test=X_test,
        pos_label=labels[1],  # assumes labels[1] is the "positive" class
        title=f"ROC Curve ({model_name})",
    )

    # Record the same AUC displayed on the ROC curve.
    auc_value = roc_out["auc"]
    print(f"{model_name} - AUC: {auc_value:.3f}")

    # Feature importance (skip if not supported)
    try:
        fe_out = plot_feature_importance(
            model,
            feature_names=getattr(X_train, "columns", None),
            top_n=top_n,
            title=f"Feature Importance ({model_name})",
            class_idx=class_idx,
        )
    except ValueError as e:
        fe_out = None
        print(f"{model_name} - Feature importance skipped: {e}")

    return {
        "model": model,
        "name": model_name,
        "confusion_matrix": conf_matrix,
        "accuracy_train": acc_train,
        "accuracy_test": acc_test,
        "auc": auc_value,
        "roc_curve": roc_out,
        "feature_importance": fe_out,
        "y_pred_test": y_pred_test,
    }
