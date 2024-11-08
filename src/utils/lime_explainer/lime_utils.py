"""Module to generate LIME explanations for model predictions on a sample.

This module provides functions to create LIME plots that explain feature contributions
for a given sample using a pre-trained model. The explanations help in understanding
the model's predictions by highlighting the impact of each feature.

Functions:
    explain_plot(model, input_data, sample): Generates a feature contribution plot.
    explain_conditions(model, input_data, sample): Generates a condition-based explanation plot.
"""

import sys
import logging
import pandas as pd
import numpy as np
from lime import lime_tabular
import matplotlib.pyplot as plt
import tempfile
import os
import cv2
from typing import Any, Union
from src.utils.common.functions import predict_with_proba

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def explain_plot(
    model: Any,
    input_data: Union[pd.DataFrame, np.ndarray],
    sample: np.ndarray,
) -> np.ndarray:
    """Generate a plot explaining feature contributions for a sample.

    This function generates a LIME plot showing the contributions of each feature in the sample
    for class predictions. It saves the plot temporarily and loads it as a NumPy array.

    Args:
        model (Any): The trained model to be explained.
        input_data (Union[pd.DataFrame, np.ndarray]): Background data used to initialize the explainer.
        sample (np.ndarray): The sample to be explained.

    Returns:
        np.ndarray: The generated plot as a NumPy array.

    Raises:
        SystemExit: If an error occurs during the explanation process.
    """
    try:
        sample_inp: np.ndarray = np.array(sample).reshape(1, -1)
        proba_imposter: float
        proba_legit: float
        proba_imposter, proba_legit = predict_with_proba(model, sample_inp)[0]

        explainer = lime_tabular.LimeTabularExplainer(
            np.asarray(input_data),
            mode="classification",
            training_labels=model.classes_,
            kernel_width=1,
        )
        explanation = explainer.explain_instance(
            np.asarray(sample),
            model.predict_proba,
            num_features=model.n_features_in_,
        )

        feature_contributions = explanation.local_exp[1]

        # Sort and categorize feature contributions
        sorted_contributions = sorted(feature_contributions, key=lambda x: x[0])
        positive_contributions = [
            (idx, weight) for idx, weight in sorted_contributions if weight > 0
        ]
        negative_contributions = [
            (idx, weight) for idx, weight in sorted_contributions if weight < 0
        ]

        # Calculate total contributions
        total_contribution_class_legit: float = sum(
            weight for _, weight in positive_contributions
        )
        total_contribution_class_imposter: float = sum(
            weight for _, weight in negative_contributions
        )

        # Prepare data for plotting
        feature_indices_pos = [idx for idx, _ in positive_contributions]
        weights_pos = [weight for _, weight in positive_contributions]
        feature_indices_neg = [idx for idx, _ in negative_contributions]
        weights_neg = [weight for _, weight in negative_contributions]

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.bar(
            feature_indices_pos,
            weights_pos,
            color="blue",
            label="Supports Class Legit (Positive)",
        )
        ax.bar(
            feature_indices_neg,
            weights_neg,
            color="red",
            label="Supports Class Imposter (Negative)",
        )

        # Annotations
        ax.axhline(
            y=total_contribution_class_legit,
            color="blue",
            linestyle="--",
            label=f"Total Contribution Class Legit: {total_contribution_class_legit:.2f}",
        )
        ax.axhline(
            y=total_contribution_class_imposter,
            color="red",
            linestyle="--",
            label=f"Total Contribution Class Imposter: {total_contribution_class_imposter:.2f}",
        )

        prob_text = (
            f"Probability Class Imposter: {proba_imposter:.5f}\n"
            f"Probability Class Legit: {proba_legit:.5f}"
        )
        props = dict(
            boxstyle="round,pad=0.5", edgecolor="gray", facecolor="lightgray"
        )
        ax.text(
            -0.15,
            1.05,
            prob_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=props,
            ha="left",
        )

        ax.set_xlabel("Feature Index")
        ax.set_ylabel("Contribution (Weight)")
        ax.set_title("LIME Feature Contributions for Each Class")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.set_xticks(range(len(sorted_contributions)))

        # Save plot temporarily and load as NumPy array
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            temp_filename = tmpfile.name
            fig.savefig(temp_filename)

        img_array: np.ndarray = cv2.imread(temp_filename)
        os.remove(temp_filename)
        plt.close(fig)

        return img_array

    except Exception as e:
        logger.error(f"An error occurred during LIME feature plot generation: {e}")
        sys.exit(1)


def explain_conditions(
    model: Any,
    input_data: Union[pd.DataFrame, np.ndarray],
    sample: np.ndarray,
) -> np.ndarray:
    """Generate a condition-based LIME explanation plot.

    This function generates a LIME plot showing contributions based on feature conditions
    for a specific sample, saves the plot temporarily, and returns it as a NumPy array.

    Args:
        model (Any): The trained model to be explained.
        input_data (Union[pd.DataFrame, np.ndarray]): Background data used to initialize the explainer.
        sample (np.ndarray): The sample to be explained.

    Returns:
        np.ndarray: The generated plot as a NumPy array.

    Raises:
        SystemExit: If an error occurs during the explanation process.
    """
    try:
        sample_inp: np.ndarray = np.array(sample).reshape(1, -1)
        proba_imposter: float
        proba_legit: float
        proba_imposter, proba_legit = predict_with_proba(model, sample_inp)[0]

        explainer = lime_tabular.LimeTabularExplainer(
            np.asarray(input_data),
            mode="classification",
            training_labels=model.classes_,
        )
        explanation = explainer.explain_instance(
            np.asarray(sample),
            model.predict_proba,
            num_features=model.n_features_in_,
        )
        contributions_with_conditions = explanation.as_list()

        # Separate and calculate contributions
        positive_contributions = [
            (cond, weight)
            for cond, weight in contributions_with_conditions
            if weight > 0
        ]
        negative_contributions = [
            (cond, weight)
            for cond, weight in contributions_with_conditions
            if weight < 0
        ]

        total_contribution_class_legit: float = sum(
            weight for _, weight in positive_contributions
        )
        total_contribution_class_imposter: float = sum(
            weight for _, weight in negative_contributions
        )

        # Prepare data for plotting
        conditions_pos = [cond for cond, _ in positive_contributions]
        weights_pos = [weight for _, weight in positive_contributions]
        conditions_neg = [cond for cond, _ in negative_contributions]
        weights_neg = [weight for _, weight in negative_contributions]

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.barh(
            conditions_pos,
            weights_pos,
            color="blue",
            label="Supports Class Legit (Positive)",
        )
        ax.barh(
            conditions_neg,
            weights_neg,
            color="red",
            label="Supports Class Imposter (Negative)",
        )

        # Annotations
        ax.axvline(
            x=total_contribution_class_legit,
            color="blue",
            linestyle="--",
            label=f"Total Contribution Class Legit: {total_contribution_class_legit:.2f}",
        )
        ax.axvline(
            x=total_contribution_class_imposter,
            color="red",
            linestyle="--",
            label=f"Total Contribution Class Imposter: {total_contribution_class_imposter:.2f}",
        )

        prob_text = (
            f"Probability Class Imposter: {proba_imposter:.5f}\n"
            f"Probability Class Legit: {proba_legit:.5f}"
        )
        props = dict(
            boxstyle="round,pad=0.5", edgecolor="gray", facecolor="lightgray"
        )
        ax.text(
            -0.15,
            1.05,
            prob_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=props,
            ha="left",
        )

        ax.set_xlabel("Contribution (Weight)")
        ax.set_ylabel("Feature Condition")
        ax.set_title("LIME Feature Contributions with Conditions for Each Class")
        ax.legend()
        ax.grid(axis="x", linestyle="--", alpha=0.7)

        # Save plot temporarily and load as NumPy array
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            temp_filename = tmpfile.name
            fig.savefig(temp_filename)

        img_array: np.ndarray = cv2.imread(temp_filename)
        os.remove(temp_filename)
        plt.close(fig)

        return img_array

    except Exception as e:
        logger.error(f"An error occurred during LIME condition plot generation: {e}")
        sys.exit(1)
