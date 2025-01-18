import random

from datasets import Dataset
from loguru import logger


def create_and_push_dataset():
    """
    Remember to create the dataset before hand in huggingface to get the url right.
    Creates a simple dataset with 100 random numbers and pushes it to Hugging Face Hub.
    """
    # Load environment variables from .env file

    dataset_id = "danivpv/testing"

    # Generate 100 random numbers
    data = {"number": [random.randint(1, 1000) for _ in range(100)]}
    logger.info("Generated 100 random numbers.")

    # Create a Hugging Face Dataset
    dataset = Dataset.from_dict(data)
    logger.info("Created Hugging Face dataset.")

    try:
        # Push the dataset to Hugging Face Hub
        dataset.push_to_hub(dataset_id)
        logger.success(f"Dataset successfully pushed to Hugging Face Hub as '{dataset_id}'.")
    except Exception as e:
        logger.error(f"Failed to push dataset to Hugging Face Hub: {e}")
        logger.error("use bash command 'huggingface-cli login' to set up credentials")


if __name__ == "__main__":
    create_and_push_dataset()
