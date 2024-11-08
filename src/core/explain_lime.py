"""Script to perform LIME analysis on a sample using a pre-trained model and training data.

This script parses command-line arguments to specify the paths to the model, training data,
and sample data, and then performs LIME analysis on the sample.

Usage:
    python script_name.py --model_dir /path/to/model.pkl --training_dir /path/to/training_data.pkl --sample_dir /path/to/sample.pkl
"""

import sys
import logging
from argparse import ArgumentParser, Namespace
from src.utils.lime_explainer.call_lime import analyze_sample_lime


def main() -> None:
    """Main function to parse arguments and perform LIME analysis on a sample.

    This function parses command-line arguments for the model directory, training data directory,
    and sample directory, and calls the `analyze_sample_lime` function to perform LIME analysis
    on the sample.

    Returns:
        None

    Raises:
        SystemExit: If an error occurs during LIME analysis.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger: logging.Logger = logging.getLogger(__name__)

    # Initialize the argument parser
    parser: ArgumentParser = ArgumentParser(description="LIME sample analysis")
    # Required arguments
    parser.add_argument(
        "-mod_dir",
        "--model_dir",
        type=str,
        default="./model/GradientBoostingClassifier.pkl",
        help="Path to the serialized model file.",
    )
    parser.add_argument(
        "-tr_dir",
        "--training_dir",
        type=str,
        default="./data/training_data.pkl",
        help="Path to the training data used to train the model.",
    )
    parser.add_argument(
        "-s_dir",
        "--sample_dir",
        type=str,
        default="./data/imposter_sample.pkl",
        help="Path to the sample data to analyze with LIME.",
    )

    # Parse arguments
    args: Namespace = parser.parse_args()

    try:
        logger.info("Starting LIME analysis")
        # Perform LIME analysis on the sample
        analyze_sample_lime(
            model_dir=args.model_dir,
            training_data_dir=args.training_dir,
            sample_dir=args.sample_dir,
        )
        logger.info("LIME analysis completed successfully")
    except Exception as e:
        logger.error(f"An error occurred during LIME analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
