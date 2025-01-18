# ruff: noqa: T201
import numpy as np
from datasets import load_dataset


def print_dataset_stats(split_data, split_name):
    """Print useful statistics for a dataset split."""
    print(f"\n=== {split_name} Split Statistics ===")
    print(f"Number of samples: {len(split_data)}")

    # Get numerical statistics for prompt, rejected, and chosen lengths
    prompt_lengths = [len(str(text)) for text in split_data["prompt"]]
    rejected_lengths = [len(str(text)) for text in split_data["rejected"]]
    chosen_lengths = [len(str(text)) for text in split_data["chosen"]]

    print("\nPrompt Length Statistics:")
    print(f"Mean length: {np.mean(prompt_lengths):.2f}")
    print(f"Median length: {np.median(prompt_lengths):.2f}")
    print(f"Min length: {min(prompt_lengths)}")
    print(f"Max length: {max(prompt_lengths)}")

    print("\nRejected Response Length Statistics:")
    print(f"Mean length: {np.mean(rejected_lengths):.2f}")
    print(f"Median length: {np.median(rejected_lengths):.2f}")
    print(f"Min length: {min(rejected_lengths)}")
    print(f"Max length: {max(rejected_lengths)}")

    print("\nChosen Response Length Statistics:")
    print(f"Mean length: {np.mean(chosen_lengths):.2f}")
    print(f"Median length: {np.median(chosen_lengths):.2f}")
    print(f"Min length: {min(chosen_lengths)}")
    print(f"Max length: {max(chosen_lengths)}")


def main():
    # Load dataset from Hugging Face
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("danivpv/ml-arxiv-dpo")

    # Print overall dataset info
    print("\nDataset Info:")
    print(dataset)

    # Print statistics for each split
    for split_name, split_data in dataset.items():
        print_dataset_stats(split_data, split_name)

    # Print first few examples
    print("\nFirst example from training split:")
    if "train" in dataset:
        print(dataset["train"][0])


if __name__ == "__main__":
    main()
