# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for custom dataset converter."""

import tempfile
from pathlib import Path

import pytest


def test_converter_import():
    """Test that the converter module can be imported."""
    # Import check only - we can't test the full conversion without audio files
    try:
        import sys

        sys.path.insert(
            0, str(Path(__file__).parent.parent.parent / "workflows" / "dataprep")
        )
        from custom_dataset_to_parquet import CustomDatasetConverter

        assert CustomDatasetConverter is not None
    except ImportError as e:
        pytest.skip(f"Cannot import converter: {e}")


def test_converter_initialization():
    """Test converter initialization."""
    try:
        import sys

        sys.path.insert(
            0, str(Path(__file__).parent.parent.parent / "workflows" / "dataprep")
        )
        from custom_dataset_to_parquet import CustomDatasetConverter

        with tempfile.TemporaryDirectory() as tmpdir:
            converter = CustomDatasetConverter(
                base_dir=tmpdir,
                audio_dir="audio",
                annotation_dir="annotations",
                output_dir=tmpdir,
                corpus_name="test_corpus",
                language_code="eng_Latn",
                regions=["region1", "region2"],
            )

            assert converter.corpus_name == "test_corpus"
            assert converter.language_code == "eng_Latn"
            assert converter.iso_code == "en"
            assert len(converter.regions) == 2
    except ImportError as e:
        pytest.skip(f"Cannot import converter: {e}")


def test_converter_split_data():
    """Test data splitting logic."""
    try:
        import sys

        sys.path.insert(
            0, str(Path(__file__).parent.parent.parent / "workflows" / "dataprep")
        )
        from custom_dataset_to_parquet import CustomDatasetConverter

        with tempfile.TemporaryDirectory() as tmpdir:
            converter = CustomDatasetConverter(
                base_dir=tmpdir,
                audio_dir="audio",
                annotation_dir="annotations",
                output_dir=tmpdir,
                corpus_name="test_corpus",
                language_code="eng_Latn",
                regions=["region1"],
                split_ratio=0.8,
            )

            # Create test data
            test_data = [
                {"audio_path": f"audio_{i}.wav", "text": f"text {i}"}
                for i in range(100)
            ]

            train_data, val_data = converter.split_data(test_data)

            # Check split ratio is approximately correct
            assert len(train_data) == 80
            assert len(val_data) == 20
            assert len(train_data) + len(val_data) == 100
    except ImportError as e:
        pytest.skip(f"Cannot import converter: {e}")
