from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch

from .utils import get_logger


LOGGER = get_logger(__name__)


def _predict_proba(model: torch.nn.Module, x_np: np.ndarray) -> np.ndarray:
    """Predict class probabilities for LIME/SHAP interfaces."""
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_np, dtype=torch.float32)
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


def generate_shap_explanation(
    model: torch.nn.Module,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    output_dir: str | Path,
) -> Dict[str, Any]:
    """Generate SHAP values, summary plot, and waterfall plot for tabular model."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train = np.array(X_train[:200], dtype=np.float32)
    X_test_sample = np.array(X_test[:100], dtype=np.float32)

    try:
        explainer = shap.KernelExplainer(lambda x: _predict_proba(model, x), X_train)
        shap_values = explainer.shap_values(X_test_sample, nsamples=100)

        # Prepare values for plots
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            plot_values_summary = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
            val_1d = shap_values[0, :, 1] if shap_values.shape[2] > 1 else shap_values[0, :, 0]
            base_val = np.mean(shap_values[:, :, 1] if shap_values.shape[2] > 1 else shap_values)
        elif isinstance(shap_values, list):
            plot_values_summary = shap_values
            val_1d = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            base_val = np.mean(shap_values[1] if len(shap_values) > 1 else shap_values[0])
        else:
            plot_values_summary = shap_values
            val_1d = shap_values[0]
            base_val = np.mean(shap_values)

        summary_path = out_dir / "shap_summary.png"
        waterfall_path = out_dir / "shap_waterfall.png"

        plt.figure(figsize=(10, 6))
        shap.summary_plot(plot_values_summary, X_test_sample, feature_names=feature_names, show=False, plot_type="bar")
        plt.tight_layout()
        plt.savefig(summary_path, dpi=200, bbox_inches="tight")
        plt.close()

        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # Multi-class: average absolute SHAP values across samples and classes
            importance = np.abs(shap_values).mean(axis=(0, 2))
        elif isinstance(shap_values, list):
            importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            # Single-class or 2D: average across samples
            importance = np.abs(shap_values).mean(axis=0)

        top_idx = np.argsort(importance)[::-1][:5]
        top_features = [feature_names[i] if i < len(feature_names) else f"feature_{i}" for i in top_idx]

        try:
            explanation = shap.Explanation(
                values=val_1d,
                base_values=base_val,
                data=X_test_sample[0],
                feature_names=feature_names,
            )
            plt.figure(figsize=(10, 5))
            shap.plots.waterfall(explanation, show=False)
            plt.tight_layout()
            plt.savefig(waterfall_path, dpi=200, bbox_inches="tight")
            plt.close()
        except Exception as plot_e:
            LOGGER.warning("SHAP waterfall plot failed: %s", plot_e)

        return {
            "shap_values": plot_values_summary,
            "feature_names": feature_names,
            "top_features": top_features,
            "plot_paths": {
                "summary": str(summary_path),
                "waterfall": str(waterfall_path),
            },
        }

    except Exception as e:  # pylint: disable=broad-except
        LOGGER.exception("SHAP explanation failed: %s", e)
        return {
            "shap_values": None,
            "feature_names": feature_names,
            "top_features": [],
            "plot_paths": {},
            "error": str(e),
        }


def generate_lime_explanation(
    model: torch.nn.Module,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    output_dir: str | Path,
    instance_idx: int = 0,
) -> Dict[str, Any]:
    """Generate LIME explanation for one prediction and save HTML."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)

    try:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            mode="classification",
            discretize_continuous=True,
        )

        instance_idx = min(max(instance_idx, 0), len(X_test) - 1)
        exp = explainer.explain_instance(
            X_test[instance_idx],
            lambda x: _predict_proba(model, x),
            num_features=min(10, len(feature_names)),
        )

        html_path = out_dir / "lime_explanation.html"
        exp.save_to_file(str(html_path))

        return {
            "explanation": exp,
            "html_path": str(html_path),
            "summary": exp.as_list(),
        }
    except Exception as e:  # pylint: disable=broad-except
        LOGGER.exception("LIME explanation failed: %s", e)
        return {"explanation": None, "html_path": "", "summary": [], "error": str(e)}


def generate_xai_results(
    model: torch.nn.Module,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    output_dir: str | Path,
) -> Dict[str, Any]:
    """Generate combined SHAP and LIME explainability outputs."""
    shap_results = generate_shap_explanation(model, X_train, X_test, feature_names, output_dir)
    lime_results = generate_lime_explanation(model, X_train, X_test, feature_names, output_dir)
    return {"shap": shap_results, "lime": lime_results}
