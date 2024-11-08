"""Module containing utility functions for data processing and model predictions.

This module provides functions to process data, display images, print data and model information,
and make predictions using a trained model. The functions are designed to be reusable across
different parts of the project.
"""

import sys
import logging
import pandas as pd
import numpy as np
import cv2

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def show_img(image: np.ndarray) -> None:
    """Display an image in a window using OpenCV.

    This function opens a window displaying the provided image. The window
    will wait for any key press to close.

    Args:
        image (np.ndarray): The image to be displayed.

    Raises:
        Exception: If an error occurs while displaying the image.
    """
    try:
        cv2.imshow("image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()  # Close the window after a key press
    except Exception as e:
        logger.error(f"An error occurred while displaying the image: {e}")
        sys.exit(1)


def process_data(input_data: pd.DataFrame) -> pd.DataFrame:
    """Process and convert data to the required formats.

    This function converts specified columns in the input DataFrame to appropriate
    types (e.g., boolean and numeric). Boolean columns are further converted to integers
    for compatibility with downstream models.

    Args:
        input_data (pd.DataFrame): The raw input data to be processed.

    Returns:
        pd.DataFrame: The processed data with standardized column types.

    Raises:
        Exception: If an error occurs during data processing.
    """
    try:
        input_df = pd.DataFrame(input_data)

        # Convert columns to required data types
        input_df[0] = input_df[0].astype(bool)
        numeric_cols = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12,
                        14, 15, 16, 17, 18, 20, 21, 22, 23, 24]
        boolean_cols = [6, 7, 13, 19]

        for col in numeric_cols:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        for col in boolean_cols:
            input_df[col] = input_df[col].astype(bool).astype(int)

        return input_df
    except Exception as e:
        logger.error(f"An error occurred during data processing: {e}")
        sys.exit(1)


def data_informations(input_data: pd.DataFrame) -> None:
    """Print summary information about a DataFrame.

    This function outputs the DataFrame's structure and statistical summary to
    help understand the dataset's structure and characteristics.

    Args:
        input_data (pd.DataFrame): The DataFrame for which information is displayed.

    Raises:
        Exception: If an error occurs while retrieving data information.
    """
    try:
        print(input_data.info())
        print(input_data.describe())
    except Exception as e:
        logger.error(f"An error occurred while displaying data information: {e}")
        sys.exit(1)


def model_informations(model) -> None:
    """Print model information, including features, classes, and parameters.

    This function displays the essential characteristics of the provided model,
    including the number of features, classes, and the model's parameters.

    Args:
        model: The trained model for which information is displayed.

    Raises:
        Exception: If an error occurs while retrieving model information.
    """
    try:
        print("Model's Features:", model.n_features_in_)
        print("Model's Classes:", model.classes_)
        print("Model's Parameters:", model.get_params())
    except Exception as e:
        logger.error(f"An error occurred while displaying model information: {e}")
        sys.exit(1)


def predict_class(model, sample: np.ndarray) -> np.ndarray:
    """Predict the class label for a given sample.

    This function uses the model to predict the class label of the provided sample.

    Args:
        model: The trained model used for prediction.
        sample (np.ndarray): The sample for which the class label is predicted.

    Returns:
        np.ndarray: The predicted class label.

    Raises:
        Exception: If an error occurs during prediction.
    """
    try:
        return model.predict(sample)
    except Exception as e:
        logger.error(f"An error occurred during class prediction: {e}")
        sys.exit(1)


def predict_with_proba(model, sample: np.ndarray) -> np.ndarray:
    """Predict the class probabilities for a given sample.

    This function uses the model to predict the probabilities for each class
    of the provided sample.

    Args:
        model: The trained model used for probability prediction.
        sample (np.ndarray): The sample for which class probabilities are predicted.

    Returns:
        np.ndarray: The predicted probabilities for each class.

    Raises:
        Exception: If an error occurs during probability prediction.
    """
    try:
        return model.predict_proba(sample)
    except Exception as e:
        logger.error(f"An error occurred during probability prediction: {e}")
        sys.exit(1)
