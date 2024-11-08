"""Module to perform SHAP analysis and generate contribution plots for model predictions.

This module provides functions to report SHAP values and to plot feature contributions using SHAP values.
It helps in understanding the model's predictions by highlighting the impact of each feature.

Functions:
    report_shap_values(model, training_data, sample): Reports SHAP values for a given sample.
    plot_shap_contributions(model, training_data, sample): Plots SHAP feature contributions for a given sample.
"""

import sys
import logging
import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2  # OpenCV library
import shap
from typing import Any
from src.utils.common.functions import predict_with_proba, predict_class

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def report_shap_values(
    model: Any, training_data: pd.DataFrame, sample: np.ndarray
) -> None:
    """Report SHAP values for a given sample using a pre-trained model.

    This function calculates SHAP values for a sample and reports the positive and negative contributions
    of each feature to the model's prediction.

    Args:
        model (Any): The trained model used for SHAP value computation.
        training_data (pd.DataFrame): The training data used to initialize the SHAP explainer.
        sample (np.ndarray): The sample for which SHAP values are calculated.

    Returns:
        None

    Raises:
        Exception: If an error occurs during SHAP value computation.
    """
    try:
        # Prepare feature names
        feature_names = list(training_data.columns)

        # Initialize SHAP explainer
        explainer = shap.Explainer(model, feature_names=feature_names)

        # Calculate SHAP values for the sample
        shap_values_sample = explainer(np.array(sample).reshape(1, -1))

        # Extract SHAP values and feature names
        shap_values = shap_values_sample.values[0]  # SHAP values (contributions)
        feature_values = sample
        feature_details = [
            (feature, value, contribution)
            for feature, value, contribution in zip(
                feature_names, feature_values, shap_values
            )
        ]

        # Separate positive and negative contributions
        positive_contributions = [
            (feature, value, contribution)
            for feature, value, contribution in feature_details
            if contribution > 0
        ]
        negative_contributions = [
            (feature, value, contribution)
            for feature, value, contribution in feature_details
            if contribution < 0
        ]

        # Calculate total contributions
        total_positive = sum(
            contribution for _, _, contribution in positive_contributions
        )
        total_negative = sum(
            contribution for _, _, contribution in negative_contributions
        )

        # Log the results
        logger.info("Positive Contributions:")
        for feature, value, contribution in positive_contributions:
            logger.info(f"{feature} = {value}: {contribution}")
        logger.info(f"Total Positive Contributions: {total_positive}")

        logger.info("\nNegative Contributions:")
        for feature, value, contribution in negative_contributions:
            logger.info(f"{feature} = {value}: {contribution}")
        logger.info(f"Total Negative Contributions: {total_negative}")

    except Exception as e:
        logger.error(f"An error occurred during SHAP value reporting: {e}")
        sys.exit(1)


def plot_shap_contributions(
    model: Any, training_data: pd.DataFrame, sample: np.ndarray
) -> np.ndarray:
    """Plot SHAP feature contributions for a given sample using a pre-trained model.

    This function generates a bar plot of SHAP values for the features of the sample, indicating their
    contribution to the model's prediction. The plot is saved temporarily and returned as a NumPy array.

    Args:
        model (Any): The trained model used for SHAP value computation.
        training_data (pd.DataFrame): The training data used to initialize the SHAP explainer.
        sample (np.ndarray): The sample for which SHAP values are calculated.

    Returns:
        np.ndarray: The plot image as a NumPy array.

    Raises:
        Exception: If an error occurs during SHAP contribution plotting.
    """
    try:
        # Prepare the sample input
        sample_inp = np.array(sample).reshape(1, -1)

        # Predict class and probabilities for the sample
        class_predicted = predict_class(model=model, sample=sample_inp)[0]
        proba_imposter, proba_legit = predict_with_proba(
            model=model, sample=sample_inp
        )[0]
        difference = np.abs(proba_imposter - proba_legit)

        # Prepare feature names
        feature_names = list(training_data.columns)

        # Initialize SHAP explainer
        explainer = shap.Explainer(model, feature_names=feature_names)

        # Calculate SHAP values for the sample
        shap_values_sample = explainer(sample_inp)
        shap_values = shap_values_sample.values[0]
        feature_values = sample_inp[0]

        # Combine feature names, values, and contributions
        feature_details = [
            (feature, value, contribution)
            for feature, value, contribution in zip(
                feature_names, feature_values, shap_values
            )
        ]

        # Sort features by absolute SHAP value
        feature_details.sort(key=lambda x: x[2], reverse=True)

        # Prepare data for plotting
        feature_names_plot = [f"{feat[0]} = {feat[1]}" for feat in feature_details]
        shap_values_plot = [feat[2] for feat in feature_details]
        colors = ["green" if val > 0 else "red" for val in shap_values_plot]

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 8))
        y_pos = np.arange(len(feature_details))
        ax.barh(y_pos, shap_values_plot, align="center", color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names_plot)
        ax.set_xlabel("SHAP Value (Impact on Model Output)")
        ax.set_title("Feature Contributions to Prediction")

        # Invert y-axis to have the highest values on top
        ax.invert_yaxis()

        # Add vertical line at zero
        ax.axvline(0, color="black", linewidth=0.8)

        # Add text box with probabilities and total contributions
        total_positive = sum(val for val in shap_values if val > 0)
        total_negative = sum(val for val in shap_values if val < 0)
        textstr = (
            f"Class Predicted: {class_predicted}\n"
            f"Probability Imposter: {proba_imposter:.6f}\n"
            f"Probability Legit: {proba_legit:.6f}\n"
            f"Difference: {difference:.6f}\n"
            f"Total Positive Contributions (support 'legit'): {total_positive:.6f}\n"
            f"Total Negative Contributions (support 'imposter'): {total_negative:.6f}"
        )

        # Define properties of the text box
        props = dict(boxstyle="round", facecolor="white", alpha=0.5)

        # Place the text box in the plot
        fig.text(
            0.68, 0.95, textstr, fontsize=8, verticalalignment="top", bbox=props
        )

        # Adjust layout to make room for the text box
        plt.tight_layout(rect=[0, 0, 0.65, 1])

        # Save the plot to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            temp_filename = tmpfile.name
            plt.savefig(temp_filename, format="png", bbox_inches="tight")
        plt.close(fig)  # Close the figure to free memory

        # Load the image using OpenCV
        image_cv = cv2.imread(temp_filename)

        # Remove the temporary file
        os.unlink(temp_filename)

        return image_cv

    except Exception as e:
        logger.error(f"An error occurred during SHAP contribution plotting: {e}")
        sys.exit(1)
