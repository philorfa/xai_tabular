"""Module to perform SHAP analysis on a sample using a pre-trained model and training data.

This module provides functionality to load a pre-trained model and training data,
predict the class and probabilities of a sample, and perform SHAP analysis to explain
the model's predictions.

Functions:
    analyze_sample_shap(model_dir, training_data_dir, sample_dir): Performs SHAP analysis on a sample.
"""

import sys
import logging
import pickle
import pandas as pd
import numpy as np
from typing import Any
from src.utils.logger import log
from src.utils.common.functions import (
    process_data,
    data_informations,
    model_informations,
    predict_class,
    predict_with_proba,
    show_img,
)
from src.utils.shap_explainer.shap_utils import (
    report_shap_values,
    plot_shap_contributions,
)


def analyze_sample_shap(
    model_dir: str, training_data_dir: str, sample_dir: str
) -> None:
    """Performs SHAP analysis on a given sample using a pre-trained model and training data.

    Args:
        model_dir (str): Path to the serialized model file.
        training_data_dir (str): Path to the training data used to train the model.
        sample_dir (str): Path to the sample data to analyze with SHAP.

    Returns:
        None

    Raises:
        Exception: If there is an error during the SHAP analysis process.
    """
    try:
        # Load and process training data
        with open(training_data_dir, 'rb') as f:
            training_data = pickle.load(f)
        training_data = pd.DataFrame(training_data)
        processed_data = process_data(training_data)

        # Optional: Uncomment to log detailed data information
        # data_informations(processed_data)

        # Load model
        with open(model_dir, 'rb') as f:
            model = pickle.load(f)
        # Optional: Uncomment to log detailed model information
        # model_informations(model)

        # Load sample data
        with open(sample_dir, 'rb') as f:
            sample = pickle.load(f)
        sample_inp = np.array(sample).reshape(1, -1)

        # Predict class and probabilities for the sample
        class_predicted = predict_class(model=model, sample=sample_inp)[0]
        proba_imposter, proba_legit = predict_with_proba(
            model=model, sample=sample_inp
        )[0]

        # Log prediction results
        log.info(
            "\n"
            "-------------------------------------\n"
            f"Class Predicted:  {class_predicted}\n"
            f"Probability Imposter: {proba_imposter}\n"
            f"Probability Legit: {proba_legit}\n"
            f"Difference: {abs(proba_imposter - proba_legit)}\n"
            "-------------------------------------\n"
        )

        # Perform SHAP analysis
        # report_shap_values(model, training_data, sample)
        plot = plot_shap_contributions(model, training_data, sample)
        show_img(plot)

    except Exception as e:
        log.error(f"An error occurred during SHAP analysis: {e}")
        sys.exit(1)
