#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tuning Omnilingual ASR for Bengali Regional Dialects with Submission Generation

This script combines:
1. Data loading from regional CSV files + WAV audio
2. Dataset conversion to parquet format (Omnilingual ASR compatible)
3. Fine-tuning using Omnilingual ASR training pipeline
4. Inference and submission generation for Kaggle

Based on the original Whisper/Unsloth approach but adapted for Omnilingual ASR.
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# SECTION 1: INSTALLATION
# ============================================================

def install_dependencies():
    """Install Omnilingual ASR and dependencies"""
    print("📦 Installing Omnilingual ASR and dependencies...")
    
    # Install libsndfile first (required by fairseq2)
    subprocess.run(["apt-get", "update", "-qq"], check=False)
    subprocess.run(["apt-get", "install", "-y", "libsndfile1"], check=False)
    
    # Install Omnilingual ASR with data extras
    subprocess.run([sys.executable, "-m", "pip", "install", "omnilingual-asr[data]"], check=True)
    
    print("✓ Installation complete!")

# Uncomment to run installation
# install_dependencies()

# ============================================================
# SECTION 2: CONFIGURATION
# ============================================================

# Data paths
BASE_DIR = "/kaggle/input/shobdotori" if os.path.exists("/kaggle/input") else "."
TRAIN_DIR = os.path.join(BASE_DIR, "Train")
ANNOTATION_DIR = os.path.join(BASE_DIR, "Train_annotation")
TEST_DIR = os.path.join(BASE_DIR, "Test") if os.path.exists(os.path.join(BASE_DIR, "Test")) else "./Test"

# Output paths
OUTPUT_BASE = "./omniasr_output"
PARQUET_DIR = os.path.join(OUTPUT_BASE, "parquet_data")
TRAINING_DIR = os.path.join(OUTPUT_BASE, "training")
ASSET_CARD_PATH = "./src/omnilingual_asr/cards/datasets/shobdotori.yaml"

# Training configuration
CORPUS_NAME = "shobdotori"
LANGUAGE_CODE = "ben_Beng"  # Bengali in Bengali script
MODEL_NAME = "omniASR_CTC_300M"  # Can be changed to 1B, 3B, or LLM variants
TOKENIZER_NAME = "omniASR_tokenizer"

# Regions (from your data)
REGIONS = [
    "Rajshahi", "Kushtia", "Sylhet", "Mymensingh", "Comilla",
    "Lakshmipur", "Pabna", "Natore", "Dhaka", "Chittagong",
    "Khulna", "Jessore", "Bogura", "Brahmanbaria", "Noakhali",
    "Barisal", "Bhola", "Rangpur", "Feni", "Jhenaidah"
]

# Create output directories
os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(PARQUET_DIR, exist_ok=True)
os.makedirs(TRAINING_DIR, exist_ok=True)

print("="*80)
print("🎙️  OMNILINGUAL ASR FINE-TUNING FOR BENGALI DIALECTS")
print("="*80)
print(f"Base directory: {BASE_DIR}")
print(f"Training data: {TRAIN_DIR}")
print(f"Annotations: {ANNOTATION_DIR}")
print(f"Test data: {TEST_DIR}")
print(f"Output: {OUTPUT_BASE}")
print(f"Model: {MODEL_NAME}")
print(f"Language: {LANGUAGE_CODE}")
print(f"Regions: {len(REGIONS)}")
print("="*80)

# ============================================================
# SECTION 3: DATA CONVERSION TO PARQUET
# ============================================================

def convert_to_parquet():
    """Convert regional CSV + WAV data to parquet format"""
    print("\n📊 STEP 1: Converting data to parquet format...")
    print("="*80)
    
    # Check if already converted
    version_dir = os.path.join(PARQUET_DIR, "version=0")
    stats_file = os.path.join(PARQUET_DIR, "language_distribution_0.tsv")
    
    if os.path.exists(version_dir) and os.path.exists(stats_file):
        print("✓ Parquet data already exists, skipping conversion")
        return
    
    # Import conversion script
    sys.path.insert(0, "./workflows/dataprep")
    from custom_dataset_to_parquet import CustomDatasetConverter
    
    # Create converter
    converter = CustomDatasetConverter(
        base_dir=BASE_DIR,
        audio_dir="Train",
        annotation_dir="Train_annotation",
        output_dir=PARQUET_DIR,
        corpus_name=CORPUS_NAME,
        language_code=LANGUAGE_CODE,
        regions=REGIONS,
        split_ratio=0.95,
        audio_column="audio",
        text_column="text",
        sample_rate=16000,
        audio_format="ogg",
        version=0,
    )
    
    # Run conversion
    converter.convert()
    
    print("✓ Data conversion complete!")
    print(f"  Parquet data: {version_dir}")
    print(f"  Statistics: {stats_file}")

convert_to_parquet()

# ============================================================
# SECTION 4: CREATE DATASET ASSET CARD
# ============================================================

def create_asset_card():
    """Create dataset asset card for Omnilingual ASR"""
    print("\n📝 STEP 2: Creating dataset asset card...")
    print("="*80)
    
    # Check if asset card already exists
    if os.path.exists(ASSET_CARD_PATH):
        print(f"✓ Asset card already exists at {ASSET_CARD_PATH}")
        return
    
    # Create asset card content
    asset_card_content = f"""# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

name: {CORPUS_NAME}
dataset_family: mixture_parquet_asr_dataset
dataset_config:
  data: {os.path.abspath(os.path.join(PARQUET_DIR, "version=0"))}
tokenizer_ref: {TOKENIZER_NAME}
"""
    
    # Create directory if needed
    os.makedirs(os.path.dirname(ASSET_CARD_PATH), exist_ok=True)
    
    # Write asset card
    with open(ASSET_CARD_PATH, 'w') as f:
        f.write(asset_card_content)
    
    print(f"✓ Asset card created at {ASSET_CARD_PATH}")

create_asset_card()

# ============================================================
# SECTION 5: CREATE TRAINING CONFIGURATION
# ============================================================

def create_training_config():
    """Create training configuration YAML"""
    print("\n⚙️  STEP 3: Creating training configuration...")
    print("="*80)
    
    config_path = os.path.join(OUTPUT_BASE, "finetune_config.yaml")
    
    # Check if config already exists
    if os.path.exists(config_path):
        print(f"✓ Training config already exists at {config_path}")
        return config_path
    
    stats_path = os.path.join(PARQUET_DIR, "language_distribution_0.tsv")
    
    config_content = f"""# Fine-tuning configuration for Bengali Regional Dialects
model:
  name: "{MODEL_NAME}"

dataset:
  name: "{CORPUS_NAME}"
  train_split: "train"
  valid_split: "dev"
  storage_mode: "MIXTURE_PARQUET"
  task_mode: "ASR"
  mixture_parquet_storage_config:
    dataset_summary_path: "{os.path.abspath(stats_path)}"
    beta_corpus: 0.5
    beta_language: 0.5
    fragment_loading:
      cache: True
  asr_task_config:
     min_audio_len: 32_000        # ~2 seconds at 16kHz
     max_audio_len: 960_000       # ~60 seconds at 16kHz
     max_num_elements: 7_680_000  # Maximum of eight 60s samples
     batch_shuffle_window: 1
     normalize_audio: true
     example_shuffle_window: 1

tokenizer:
  name: "{TOKENIZER_NAME}"

optimizer:
  config:
    lr: 1e-05  # Lower learning rate for fine-tuning

trainer:
  freeze_encoder_for_n_steps: 0
  mixed_precision:
    dtype: "torch.bfloat16"
  grad_accumulation:
    num_batches: 4

regime:
  num_steps: 5_000              # Adjust based on dataset size
  validate_after_n_steps: 0
  validate_every_n_steps: 500
  checkpoint_every_n_steps: 500
  publish_metrics_every_n_steps: 100
"""
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✓ Training config created at {config_path}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Learning rate: 1e-05")
    print(f"  Training steps: 5,000")
    print(f"  Batch size: ~4 x 4 (with gradient accumulation)")
    
    return config_path

config_path = create_training_config()

# ============================================================
# SECTION 6: RUN TRAINING
# ============================================================

def run_training():
    """Run Omnilingual ASR training"""
    print("\n🚀 STEP 4: Starting training...")
    print("="*80)
    
    # Check if training has been completed
    final_checkpoint = os.path.join(TRAINING_DIR, "checkpoints", "step_5000")
    if os.path.exists(final_checkpoint):
        print(f"✓ Training already complete, checkpoint exists at {final_checkpoint}")
        return final_checkpoint
    
    print("⏱️  This will take approximately 15-30 minutes...")
    print(f"📊 Training logs will be saved to {TRAINING_DIR}")
    
    # Set environment variable for output directory
    os.environ["OUTPUT_DIR"] = TRAINING_DIR
    
    # Run training command
    cmd = [
        sys.executable, "-m", "workflows.recipes.wav2vec2.asr",
        TRAINING_DIR,
        "--config-file", config_path
    ]
    
    print(f"\n🔧 Running: {' '.join(cmd)}\n")
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Training completed successfully!")
        return final_checkpoint
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")
        print("\n💡 Troubleshooting:")
        print("  1. Check GPU memory (use 300M model for 8GB GPU)")
        print("  2. Reduce max_num_elements if OOM")
        print("  3. Increase grad_accumulation if OOM")
        print("  4. Check training logs for errors")
        return None

# Uncomment to run training
# checkpoint_path = run_training()

# For demonstration, we'll assume training is complete
# In practice, you would run the training and then proceed to inference
checkpoint_path = os.path.join(TRAINING_DIR, "checkpoints", "step_5000", "model.pt")

print("\n⚠️  Training section skipped for demonstration")
print("   Uncomment `run_training()` to actually train the model")
print(f"   Expected checkpoint: {checkpoint_path}")

# ============================================================
# SECTION 7: INFERENCE AND SUBMISSION GENERATION
# ============================================================

print("\n🎯 STEP 5: Running inference and generating submission...")
print("="*80)

def load_model_for_inference():
    """Load fine-tuned model for inference"""
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    
    print("📦 Loading fine-tuned model...")
    
    if os.path.exists(checkpoint_path):
        # Load fine-tuned model
        pipeline = ASRInferencePipeline(
            model_path=checkpoint_path,
            model_card=MODEL_NAME
        )
        print(f"✓ Loaded fine-tuned model from {checkpoint_path}")
    else:
        # Fall back to pre-trained model (for testing)
        print("⚠️  Fine-tuned model not found, using pre-trained model")
        pipeline = ASRInferencePipeline(model_card=MODEL_NAME)
    
    return pipeline

def generate_submission():
    """Generate submission file using fine-tuned model"""
    
    # Find test files
    if not os.path.exists(TEST_DIR):
        print(f"❌ Test directory not found: {TEST_DIR}")
        return None
    
    test_files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(".wav")])
    print(f"📂 Found {len(test_files)} test files in {TEST_DIR}")
    
    if len(test_files) == 0:
        print("❌ No test files found!")
        return None
    
    # Load model
    try:
        pipeline = load_model_for_inference()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # Prepare test file paths
    test_paths = [os.path.join(TEST_DIR, f) for f in test_files]
    
    # Run inference in batches
    print(f"\n🎙️  Transcribing {len(test_files)} files...")
    print("⏱️  This will take approximately 3-5 minutes...")
    
    batch_size = 8
    all_transcriptions = []
    
    from tqdm import tqdm
    
    for i in tqdm(range(0, len(test_paths), batch_size), desc="Transcribing"):
        batch_paths = test_paths[i:i+batch_size]
        
        try:
            # Run inference
            transcriptions = pipeline.transcribe(
                batch_paths,
                lang=[LANGUAGE_CODE] * len(batch_paths),
                batch_size=len(batch_paths)
            )
            all_transcriptions.extend(transcriptions)
        except Exception as e:
            print(f"\n⚠️  Error processing batch {i}: {e}")
            # Use fallback text for failed batch
            all_transcriptions.extend(["আমি ভাত খাই."] * len(batch_paths))
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        "audio": [f.lower() for f in test_files],
        "text": all_transcriptions
    })
    
    # Sort and save
    submission_df = submission_df.sort_values("audio").reset_index(drop=True)
    
    # Quality check
    print("\n" + "="*80)
    print("SUBMISSION QUALITY CHECK")
    print("="*80)
    print(f"Total rows: {len(submission_df)}")
    print(f"Expected files: {len(test_files)}")
    
    empty = (submission_df["text"].str.strip() == "").sum()
    print(f"Empty texts: {empty}")
    
    unique_texts = submission_df["text"].nunique()
    print(f"Unique transcriptions: {unique_texts}")
    
    if unique_texts == 1:
        print("⚠️  WARNING: All transcriptions are IDENTICAL!")
    elif unique_texts < 10:
        print(f"⚠️  WARNING: Very low diversity ({unique_texts} unique texts)")
    else:
        print("✅ Good transcription diversity!")
    
    # Show samples
    print("\n" + "="*80)
    print("FIRST 10 PREDICTIONS:")
    print("="*80)
    print(submission_df.head(10).to_string(index=False, max_colwidth=80))
    
    print("\n" + "="*80)
    print("RANDOM 10 PREDICTIONS:")
    print("="*80)
    print(submission_df.sample(min(10, len(submission_df))).to_string(index=False, max_colwidth=80))
    
    # Save submission
    submission_path = os.path.join(OUTPUT_BASE, "submission.csv")
    submission_df.to_csv(submission_path, index=False, encoding="utf-8")
    
    print(f"\n💾 Saved submission to: {submission_path}")
    print("="*80)
    
    return submission_path

# Generate submission (uncomment when model is trained)
# submission_path = generate_submission()

print("\n⚠️  Inference section skipped for demonstration")
print("   Uncomment `generate_submission()` to run inference after training")

# ============================================================
# SECTION 8: SUMMARY
# ============================================================

print("\n" + "="*80)
print("📋 SUMMARY")
print("="*80)
print(f"""
✓ Data converted to parquet format: {PARQUET_DIR}/version=0
✓ Dataset asset card created: {ASSET_CARD_PATH}
✓ Training config created: {config_path}
✓ Training output directory: {TRAINING_DIR}
✓ Expected checkpoint: {checkpoint_path}

🎯 NEXT STEPS:

1. Uncomment `run_training()` in SECTION 6 to start training
   - Training will take ~15-30 minutes on a single GPU
   - Model will be saved to {TRAINING_DIR}/checkpoints/

2. After training, uncomment `generate_submission()` in SECTION 7
   - Inference will take ~3-5 minutes for all test files
   - Submission will be saved to {OUTPUT_BASE}/submission.csv

3. Upload submission.csv to Kaggle competition

💡 TIPS:

- Start with omniASR_CTC_300M for fastest training (2GB VRAM)
- Use omniASR_CTC_1B for better quality (3GB VRAM)
- Use omniASR_LLM_1B for best quality with language conditioning (6GB VRAM)
- Adjust num_steps in config based on dataset size
- Monitor validation WER during training
- If OOM, reduce max_num_elements or increase grad_accumulation

📚 DOCUMENTATION:

- Quick Start: workflows/dataprep/QUICKSTART.md
- Full Guide: workflows/dataprep/CUSTOM_DATASET_GUIDE.md
- Training Recipes: workflows/recipes/wav2vec2/asr/README.md

🎉 HAPPY FINE-TUNING!
""")

print("="*80)
