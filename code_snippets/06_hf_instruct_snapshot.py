# ruff: noqa: T201
import numpy as np
import pandas as pd
from datasets import load_dataset


def print_dataset_stats(split_data, split_name):
    """Print useful statistics for a dataset split."""
    print(f"\n=== {split_name} Split Statistics ===")
    print(f"Number of samples: {len(split_data)}")

    # Get numerical statistics for instruction and output lengths
    instruction_lengths = [len(str(text)) for text in split_data["instruction"]]
    output_lengths = [len(str(text)) for text in split_data["output"]]

    print("\nInstruction Length Statistics:")
    print(f"Mean length: {np.mean(instruction_lengths):.2f}")
    print(f"Median length: {np.median(instruction_lengths):.2f}")
    print(f"Min length: {min(instruction_lengths)}")
    print(f"Max length: {max(instruction_lengths)}")

    print("\nOutput Length Statistics:")
    print(f"Mean length: {np.mean(output_lengths):.2f}")
    print(f"Median length: {np.median(output_lengths):.2f}")
    print(f"Min length: {min(output_lengths)}")
    print(f"Max length: {max(output_lengths)}")

    # Print sample distribution if labels exist
    if "label" in split_data.features:
        print("\nLabel Distribution:")
        label_dist = pd.Series(split_data["label"]).value_counts()
        for label, count in label_dist.items():
            print(f"Label {label}: {count} samples ({count/len(split_data)*100:.2f}%)")


def main():
    # Load dataset from Hugging Face
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("danivpv/ml-arxiv-instruct")

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
