#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Convert custom WAV dataset to Omnilingual ASR parquet format.

This script converts a custom audio dataset with CSV annotations into the
parquet format required by Omnilingual ASR training pipeline.

Example usage:
    python custom_dataset_to_parquet.py \\
        --base_dir="/path/to/dataset" \\
        --audio_dir="Train" \\
        --annotation_dir="Train_annotation" \\
        --output_dir="/output/parquet" \\
        --corpus_name="my_corpus" \\
        --language_code="ben_Beng" \\
        --regions="Region1,Region2,Region3"
"""

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from audio_tools import AudioTableProcessor, map_to_target_schema
from text_tools import text_normalize
from tqdm import tqdm


class CustomDatasetConverter:
    """Converts custom dataset format to parquet format for Omnilingual ASR."""

    def __init__(
        self,
        base_dir: str,
        audio_dir: str,
        annotation_dir: str,
        output_dir: str,
        corpus_name: str,
        language_code: str,
        regions: List[str],
        split_ratio: float = 0.95,
        audio_column: str = "audio",
        text_column: str = "text",
        sample_rate: int = 16000,
        audio_format: str = "ogg",
        version: int = 0,
    ):
        """
        Initialize the converter.

        Args:
            base_dir: Root directory containing the dataset
            audio_dir: Subdirectory containing audio files (relative to base_dir)
            annotation_dir: Subdirectory containing CSV files (relative to base_dir)
            output_dir: Directory where parquet files will be saved
            corpus_name: Name of the corpus (e.g., "shobdotori", "my_dataset")
            language_code: Language code in format xxx_Xxxx (e.g., "ben_Beng")
            regions: List of region names (matching CSV filenames without extension)
            split_ratio: Ratio of train/validation split (default: 0.95)
            audio_column: Name of column in CSV containing audio filenames
            text_column: Name of column in CSV containing transcriptions
            sample_rate: Target sample rate for audio (default: 16000)
            audio_format: Output audio format, "ogg" or "flac" (default: "ogg")
            version: Dataset version number (default: 0)
        """
        self.base_dir = Path(base_dir)
        self.audio_base_dir = self.base_dir / audio_dir
        self.annotation_dir = self.base_dir / annotation_dir
        self.output_dir = Path(output_dir)
        self.corpus_name = corpus_name
        self.language_code = language_code
        self.regions = regions
        self.split_ratio = split_ratio
        self.audio_column = audio_column
        self.text_column = text_column
        self.sample_rate = sample_rate
        self.audio_format = audio_format
        self.version = version

        # Initialize audio processor
        self.audio_processor = AudioTableProcessor(
            sample_rate=sample_rate,
            audio_column="audio_bytes",
            audio_format=audio_format,
        )

        # Extract ISO code from language code (first 2-3 letters before underscore)
        self.iso_code = language_code.split("_")[0][:2]

    def load_data(self) -> List[dict]:
        """
        Load all data from CSV files.

        Returns:
            List of dictionaries with 'audio_path' and 'text' keys
        """
        all_data = []

        print(f"📊 Loading data from {len(self.regions)} regions...")
        for region in tqdm(self.regions, desc="Loading regions"):
            csv_path = self.annotation_dir / f"{region}.csv"

            if not csv_path.exists():
                print(f"⚠️  Warning: CSV file not found: {csv_path}")
                continue

            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"⚠️  Warning: Failed to read {csv_path}: {e}")
                continue

            if self.audio_column not in df.columns:
                print(
                    f"⚠️  Warning: '{self.audio_column}' column not found in {csv_path}"
                )
                continue

            if self.text_column not in df.columns:
                print(
                    f"⚠️  Warning: '{self.text_column}' column not found in {csv_path}"
                )
                continue

            for _, row in df.iterrows():
                audio_filename = row[self.audio_column]
                text = row[self.text_column]

                # Construct full audio path
                audio_path = self.audio_base_dir / region / audio_filename

                if not audio_path.exists():
                    continue

                all_data.append({"audio_path": str(audio_path), "text": str(text)})

        print(f"✓ Loaded {len(all_data)} samples")
        return all_data

    def split_data(self, all_data: List[dict]) -> Tuple[List[dict], List[dict]]:
        """
        Split data into train and validation sets.

        Args:
            all_data: List of all data samples

        Returns:
            Tuple of (train_data, val_data)
        """
        # Shuffle data
        random.seed(42)
        random.shuffle(all_data)

        # Split
        split_idx = int(len(all_data) * self.split_ratio)
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]

        print(
            f"📊 Split: {len(train_data)} train samples, {len(val_data)} validation samples"
        )
        return train_data, val_data

    def process_split(
        self, data: List[dict], split_name: str, batch_size: int = 100
    ) -> int:
        """
        Process a data split and save to parquet.

        Args:
            data: List of data samples
            split_name: Name of the split ("train" or "dev")
            batch_size: Number of samples per parquet row group

        Returns:
            Total audio duration in seconds
        """
        if len(data) == 0:
            print(f"⚠️  Warning: No data to process for {split_name} split")
            return 0

        # Create output directory
        output_path = (
            self.output_dir
            / f"version={self.version}"
            / f"corpus={self.corpus_name}"
            / f"split={split_name}"
            / f"language={self.language_code}"
        )
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"⚙️  Processing {split_name} split ({len(data)} samples)...")

        # Process in batches
        all_batches = []
        total_duration = 0

        for i in tqdm(range(0, len(data), batch_size), desc=f"Processing {split_name}"):
            batch_data = data[i : i + batch_size]

            # Create lists for batch
            audio_paths = []
            texts = []

            for item in batch_data:
                audio_paths.append(item["audio_path"])
                texts.append(item["text"])

            # Normalize texts
            normalized_texts = []
            for text in texts:
                try:
                    normalized_text = text_normalize(text, iso_code=self.iso_code)
                    normalized_texts.append(normalized_text)
                except Exception as e:
                    print(f"⚠️  Warning: Text normalization failed: {e}")
                    normalized_texts.append(text)

            # Create PyArrow table
            batch_table = pa.table(
                {
                    "audio_path": pa.array(audio_paths, type=pa.string()),
                    "transcription": pa.array(normalized_texts, type=pa.string()),
                }
            )

            # Process audio
            try:
                batch_table = self.audio_processor(batch_table)
            except Exception as e:
                print(f"⚠️  Warning: Audio processing failed for batch: {e}")
                continue

            # Calculate duration
            if "audio_size" in batch_table.column_names:
                batch_duration = sum(batch_table["audio_size"].to_pylist())
                total_duration += batch_duration

            # Add language column
            batch_table = batch_table.append_column(
                "language",
                pa.array([self.language_code] * len(batch_table), type=pa.string()),
            )

            # Map to target schema
            batch_table = map_to_target_schema(
                batch_table, split=split_name, corpus=self.corpus_name
            )

            all_batches.append(batch_table)

        if len(all_batches) == 0:
            print(f"⚠️  Warning: No batches processed for {split_name}")
            return 0

        # Concatenate all batches
        final_table = pa.concat_tables(all_batches)

        # Write to parquet
        output_file = output_path / "part-00000.parquet"
        pq.write_table(
            final_table,
            output_file,
            row_group_size=100,  # Match the expected format
            compression="snappy",
        )

        duration_seconds = total_duration / self.sample_rate
        print(
            f"✓ Saved {len(final_table)} samples to {output_file} ({duration_seconds:.2f} seconds of audio)"
        )

        return duration_seconds

    def create_statistics_file(self, train_duration: float, val_duration: float):
        """
        Create dataset statistics TSV file.

        Args:
            train_duration: Total training audio duration in seconds
            val_duration: Total validation audio duration in seconds
        """
        stats_path = self.output_dir / f"language_distribution_{self.version}.tsv"

        # Create statistics DataFrame
        stats_data = {
            "corpus": [self.corpus_name, self.corpus_name],
            "language": [self.language_code, self.language_code],
            "split": ["train", "dev"],
            "duration_seconds": [train_duration, val_duration],
        }

        df = pd.DataFrame(stats_data)
        df.to_csv(stats_path, sep="\t", index=False)

        print(f"✓ Created statistics file: {stats_path}")
        print(f"  Train: {train_duration:.2f} seconds")
        print(f"  Dev: {val_duration:.2f} seconds")
        print(f"  Total: {train_duration + val_duration:.2f} seconds")

    def convert(self):
        """Run the full conversion pipeline."""
        print("=" * 80)
        print("Custom Dataset to Parquet Converter")
        print("=" * 80)
        print(f"Base directory: {self.base_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Corpus name: {self.corpus_name}")
        print(f"Language code: {self.language_code}")
        print(f"Regions: {', '.join(self.regions)}")
        print("=" * 80)

        # Load data
        all_data = self.load_data()

        if len(all_data) == 0:
            print("❌ Error: No data loaded. Please check your input paths.")
            return

        # Split data
        train_data, val_data = self.split_data(all_data)

        # Process splits
        train_duration = self.process_split(train_data, "train")
        val_duration = self.process_split(val_data, "dev")

        # Create statistics file
        self.create_statistics_file(train_duration, val_duration)

        print("=" * 80)
        print("✅ Conversion complete!")
        print("=" * 80)
        print(f"Output directory: {self.output_dir / f'version={self.version}'}")
        print(
            f"Statistics file: {self.output_dir / f'language_distribution_{self.version}.tsv'}"
        )
        print()
        print("Next steps:")
        print("1. Create a dataset asset card in src/omnilingual_asr/cards/datasets/")
        print("2. Verify the dataset with dataloader_example.py")
        print("3. Create a training configuration YAML file")
        print("4. Start training with the recipe")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Convert custom WAV dataset to Omnilingual ASR parquet format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python custom_dataset_to_parquet.py \\
      --base_dir="/kaggle/input/shobdotori" \\
      --audio_dir="Train" \\
      --annotation_dir="Train_annotation" \\
      --output_dir="/output/parquet" \\
      --corpus_name="shobdotori" \\
      --language_code="ben_Beng" \\
      --regions="Rajshahi,Kushtia,Sylhet,Dhaka"

For more information, see CUSTOM_DATASET_GUIDE.md
        """,
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Root directory containing the dataset",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Subdirectory containing audio files (relative to base_dir)",
    )
    parser.add_argument(
        "--annotation_dir",
        type=str,
        required=True,
        help="Subdirectory containing CSV annotation files (relative to base_dir)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where parquet files will be saved",
    )
    parser.add_argument(
        "--corpus_name",
        type=str,
        required=True,
        help="Name of the corpus (e.g., 'shobdotori', 'my_dataset')",
    )
    parser.add_argument(
        "--language_code",
        type=str,
        required=True,
        help="Language code in format xxx_Xxxx (e.g., 'ben_Beng' for Bengali)",
    )
    parser.add_argument(
        "--regions",
        type=str,
        required=True,
        help="Comma-separated list of region names (matching CSV filenames)",
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.95,
        help="Train/validation split ratio (default: 0.95)",
    )
    parser.add_argument(
        "--audio_column",
        type=str,
        default="audio",
        help="Name of column in CSV containing audio filenames (default: 'audio')",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Name of column in CSV containing transcriptions (default: 'text')",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Target sample rate for audio (default: 16000)",
    )
    parser.add_argument(
        "--audio_format",
        type=str,
        default="ogg",
        choices=["ogg", "flac"],
        help="Output audio format (default: 'ogg')",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=0,
        help="Dataset version number (default: 0)",
    )

    args = parser.parse_args()

    # Parse regions
    regions = [r.strip() for r in args.regions.split(",")]

    # Create converter
    converter = CustomDatasetConverter(
        base_dir=args.base_dir,
        audio_dir=args.audio_dir,
        annotation_dir=args.annotation_dir,
        output_dir=args.output_dir,
        corpus_name=args.corpus_name,
        language_code=args.language_code,
        regions=regions,
        split_ratio=args.split_ratio,
        audio_column=args.audio_column,
        text_column=args.text_column,
        sample_rate=args.sample_rate,
        audio_format=args.audio_format,
        version=args.version,
    )

    # Run conversion
    converter.convert()


if __name__ == "__main__":
    main()
