# Fine-Tuning with Custom WAV Dataset

This guide shows you how to fine-tune one of the Omnilingual ASR models with your own custom WAV audio dataset. We'll walk through converting your data to the required parquet format and training a model.

## Overview

Fine-tuning involves three main steps:
1. **Data Preparation**: Convert your audio files and transcriptions to parquet format
2. **Dataset Configuration**: Create an asset card for your dataset
3. **Training**: Run the fine-tuning recipe with your dataset

## Prerequisites

Install the required dependencies:

```bash
pip install "omnilingual-asr[data]"
```

## Step 1: Understanding Your Data Format

This guide assumes you have:
- **Audio files**: WAV format (or FLAC/OGG) organized in a directory structure
- **Transcriptions**: CSV files or similar format with audio filenames and their text transcriptions
- **Language information**: Optional language codes for multilingual datasets

### Example Directory Structure

```
your_dataset/
├── Train/
│   ├── Region1/
│   │   ├── audio001.wav
│   │   ├── audio002.wav
│   │   └── ...
│   ├── Region2/
│   │   └── ...
│   └── ...
├── Train_annotation/
│   ├── Region1.csv
│   ├── Region2.csv
│   └── ...
└── ...
```

### Example CSV Format

Each CSV file should have at least two columns:
- `audio`: Filename of the audio file (relative to the audio directory)
- `text`: Transcription text

```csv
audio,text
audio001.wav,This is the transcription text
audio002.wav,Another example transcription
...
```

## Step 2: Prepare Your Data to Parquet Format

We provide a script [`custom_dataset_to_parquet.py`](./custom_dataset_to_parquet.py) that converts your custom dataset to the required parquet format.

### Basic Usage

```bash
python workflows/dataprep/custom_dataset_to_parquet.py \
    --base_dir="/path/to/your_dataset" \
    --audio_dir="Train" \
    --annotation_dir="Train_annotation" \
    --output_dir="/path/to/output/parquet" \
    --corpus_name="my_corpus" \
    --language_code="ben_Beng" \
    --regions="Region1,Region2,Region3"
```

### Parameters

- `--base_dir`: Root directory containing your dataset
- `--audio_dir`: Subdirectory containing audio files (relative to base_dir)
- `--annotation_dir`: Subdirectory containing CSV annotation files (relative to base_dir)
- `--output_dir`: Where to save the parquet files
- `--corpus_name`: Name for your corpus (e.g., "shobdotori", "my_dataset")
- `--language_code`: Language code in format `xxx_Xxxx` (e.g., "ben_Beng" for Bengali, "eng_Latn" for English)
- `--regions`: Comma-separated list of region names (should match CSV filenames without .csv extension)
- `--split_ratio`: Train/validation split ratio (default: 0.95, meaning 95% train, 5% validation)
- `--audio_column`: Column name in CSV for audio filenames (default: "audio")
- `--text_column`: Column name in CSV for transcriptions (default: "text")
- `--sample_rate`: Target sample rate for audio (default: 16000)
- `--audio_format`: Output audio format: "ogg" or "flac" (default: "ogg")

### Language Codes

Languages follow the format `{language_code}_{script}`. Common examples:
- Bengali (Bangla): `ben_Beng`
- English: `eng_Latn`
- Hindi: `hin_Deva`
- Urdu: `urd_Arab`
- Spanish: `spa_Latn`
- Mandarin Chinese (Simplified): `cmn_Hans`

See [supported languages](../../src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py) for the full list of 1600+ supported language codes.

### Example with Bangladeshi Regional Dataset

```bash
python workflows/dataprep/custom_dataset_to_parquet.py \
    --base_dir="/kaggle/input/shobdotori" \
    --audio_dir="Train" \
    --annotation_dir="Train_annotation" \
    --output_dir="/output/shobdotori_parquet" \
    --corpus_name="shobdotori" \
    --language_code="ben_Beng" \
    --regions="Rajshahi,Kushtia,Sylhet,Mymensingh,Comilla,Lakshmipur,Pabna,Natore,Dhaka,Chittagong,Khulna,Jessore,Bogura,Brahmanbaria,Noakhali,Barisal,Bhola,Rangpur,Feni,Jhenaidah"
```

### Output Structure

The script will create a parquet dataset with the following structure:

```
output_dir/
└── version=0/
    └── corpus=my_corpus/
        ├── split=train/
        │   └── language=ben_Beng/
        │       └── part-00000.parquet
        └── split=dev/
            └── language=ben_Beng/
                └── part-00000.parquet
```

### Generated Statistics

The script also creates a statistics file at `output_dir/language_distribution_0.tsv` with corpus-language audio duration statistics. This file is used for temperature sampling during training.

## Step 3: Create a Dataset Asset Card

Create a YAML file in `src/omnilingual_asr/cards/datasets/` for your dataset:

```bash
cat > src/omnilingual_asr/cards/datasets/my_custom_dataset.yaml << 'EOF'
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

name: my_custom_dataset
dataset_family: mixture_parquet_asr_dataset
dataset_config:
  data: /path/to/output/parquet/version=0
tokenizer_ref: omniASR_tokenizer
EOF
```

Replace:
- `name`: A unique identifier for your dataset
- `data`: The path to your parquet dataset (the `version=0` directory)
- `tokenizer_ref`: Use `omniASR_tokenizer` for most models, or `omniASR_tokenizer_v7` for `omniASR_LLM_7B`

## Step 4: Verify Your Dataset

Test your dataset before training:

```bash
python -m workflows.dataprep.dataloader_example \
    --dataset_path="/path/to/output/parquet/version=0" \
    --split="train" \
    --tokenizer_name="omniASR_tokenizer" \
    --num_iterations=5
```

This will:
1. Load your dataset
2. Create batches
3. Display sample texts and statistics
4. Verify the data pipeline works correctly

## Step 5: Create a Training Configuration

Create a training config YAML file (e.g., `my_finetune_config.yaml`):

```yaml
# Fine-tune the 300M CTC model on custom dataset
model:
  name: "omniASR_CTC_300M"

dataset:
  name: "my_custom_dataset"  # Must match the name in your dataset asset card
  train_split: "train"
  valid_split: "dev"
  storage_mode: "MIXTURE_PARQUET"
  task_mode: "ASR"
  mixture_parquet_storage_config:
    dataset_summary_path: "/path/to/output/parquet/language_distribution_0.tsv"
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
  name: "omniASR_tokenizer"

optimizer:
  config:
    lr: 1e-05  # Lower learning rate for fine-tuning

trainer:
  freeze_encoder_for_n_steps: 0
  mixed_precision:
    dtype: "torch.bfloat16"
  grad_accumulation:
    num_batches: 4  # Increase if running out of memory

regime:
  num_steps: 5_000              # Adjust based on dataset size
  validate_after_n_steps: 0
  validate_every_n_steps: 500
  checkpoint_every_n_steps: 500
  publish_metrics_every_n_steps: 100
```

### Key Configuration Parameters

**Model Selection:**
- `omniASR_CTC_300M`: Fastest, lowest memory (~2 GiB VRAM)
- `omniASR_CTC_1B`: Balanced performance (~3 GiB VRAM)
- `omniASR_CTC_3B`: High performance (~8 GiB VRAM)
- `omniASR_LLM_300M`: Best for multilingual with language conditioning (~5 GiB VRAM)
- `omniASR_LLM_1B`: Better multilingual performance (~6 GiB VRAM)

**Learning Rate:**
- Fine-tuning: `1e-05` to `5e-05`
- Training from scratch: `1e-04` to `5e-04`

**Batch Parameters:**
- `max_audio_len`: Maximum audio length in samples (960,000 = 60s at 16kHz)
- `max_num_elements`: Total audio samples across all sequences in a batch
- `grad_accumulation.num_batches`: Increase to simulate larger batches if OOM

**Training Duration:**
- Small dataset (<10 hours): 2,000-5,000 steps
- Medium dataset (10-100 hours): 5,000-20,000 steps
- Large dataset (>100 hours): 20,000-50,000 steps

## Step 6: Run Fine-Tuning

Set your output directory and start training:

```bash
export OUTPUT_DIR="/path/to/training/output"

python -m workflows.recipes.wav2vec2.asr \
    $OUTPUT_DIR \
    --config-file my_finetune_config.yaml
```

### Training Output

The training script will:
- Save checkpoints to `$OUTPUT_DIR/checkpoints/`
- Log metrics to `$OUTPUT_DIR/logs/`
- Display training progress including loss and validation metrics

### Monitoring Training

Monitor the training progress:
- Watch the loss decrease over time
- Check validation metrics every N steps
- Look for signs of overfitting (validation loss increasing while training loss decreases)

### Resume Training

If training is interrupted, resume from the last checkpoint:

```bash
python -m workflows.recipes.wav2vec2.asr \
    $OUTPUT_DIR \
    --config-file my_finetune_config.yaml \
    --resume-from-checkpoint
```

## Step 7: Evaluate Your Model

After training, evaluate on your validation set:

```bash
python -m workflows.recipes.wav2vec2.asr.eval \
    $OUTPUT_DIR \
    --config-file eval_config.yaml \
    --checkpoint-path "$OUTPUT_DIR/checkpoints/step_5000/model.pt"
```

## Advanced Options

### Training from a Pre-trained Encoder

To train a CTC head from a pre-trained W2V encoder:

```yaml
model:
  name: ""  # Leave empty for new model

pretrained_encoder:
  name: "omniASR_W2V_300M"  # Use pre-trained encoder

dataset:
  name: "my_custom_dataset"
  # ... rest of config
```

Use config template: [`workflows/recipes/wav2vec2/asr/configs/ctc-from-encoder-recommendation.yaml`](../recipes/wav2vec2/asr/configs/ctc-from-encoder-recommendation.yaml)

### Multilingual Fine-Tuning

For multiple languages, prepare data for each language with appropriate language codes:

```bash
python workflows/dataprep/custom_dataset_to_parquet.py \
    --base_dir="/path/to/dataset" \
    --language_code="ben_Beng" \
    --corpus_name="bengali_corpus" \
    --output_dir="/output/parquet" \
    # ... other params

python workflows/dataprep/custom_dataset_to_parquet.py \
    --base_dir="/path/to/dataset2" \
    --language_code="hin_Deva" \
    --corpus_name="hindi_corpus" \
    --output_dir="/output/parquet" \
    # ... other params
```

The parquet format supports multiple corpora and languages in the same dataset directory.

### Text Normalization

The conversion script automatically normalizes text:
- Converts to lowercase (configurable)
- Removes/normalizes punctuation
- Handles language-specific characters

To customize text normalization, edit the `text_normalize` function in [`text_tools.py`](./text_tools.py).

### Audio Preprocessing

Audio is automatically:
- Resampled to 16kHz
- Converted to mono channel
- Compressed to OGG or FLAC format for efficient storage

## Troubleshooting

### Out of Memory (OOM) Errors

If you encounter OOM errors during training:

1. **Reduce batch size** by adjusting `max_num_elements`:
   ```yaml
   asr_task_config:
     max_num_elements: 3_840_000  # Half the default
   ```

2. **Increase gradient accumulation**:
   ```yaml
   trainer:
     grad_accumulation:
       num_batches: 8  # Double the accumulation
   ```

3. **Use a smaller model**:
   - Switch from 1B → 300M
   - Or from CTC to SSL encoder only

4. **Reduce audio length**:
   ```yaml
   asr_task_config:
     max_audio_len: 480_000  # ~30 seconds instead of 60
   ```

### Poor Performance

If your model isn't learning well:

1. **Check your data quality**:
   - Verify audio files load correctly
   - Check transcription accuracy
   - Ensure language code matches your data

2. **Adjust learning rate**:
   - Try higher (5e-05) or lower (1e-06) learning rates
   - Monitor if loss is decreasing

3. **Train longer**:
   - Increase `num_steps` if loss is still decreasing

4. **Use a larger model**:
   - 1B or 3B models may perform better on complex datasets

### Audio Loading Errors

If audio files fail to load:

1. **Check file formats**: Ensure files are valid WAV/FLAC/OGG
2. **Verify paths**: Make sure audio paths in CSVs match actual files
3. **Check permissions**: Ensure read access to all audio files
4. **Test individual files**:
   ```python
   import librosa
   audio, sr = librosa.load("path/to/audio.wav", sr=16000)
   print(f"Loaded audio shape: {audio.shape}, sample rate: {sr}")
   ```

### Dataset Not Found

If the training script can't find your dataset:

1. **Verify asset card**:
   - Check the YAML file is in `src/omnilingual_asr/cards/datasets/`
   - Ensure `data` path points to the `version=0` directory
   - Verify the `name` field matches your training config

2. **Check parquet files exist**:
   ```bash
   ls -R /path/to/output/parquet/version=0/
   ```

3. **Test dataset loading**:
   ```python
   from omnilingual_asr.datasets.impl.mixture_parquet_asr_dataset import MixtureParquetAsrDataset
   dataset = MixtureParquetAsrDataset.from_path("/path/to/parquet/version=0")
   print(f"Dataset loaded successfully: {dataset}")
   ```

## Best Practices

1. **Start Small**: Test with a small subset of data first (a few hundred samples)
2. **Validate Early**: Check validation metrics frequently in early training
3. **Use Pre-trained Models**: Fine-tuning is faster and more effective than training from scratch
4. **Monitor Overfitting**: Stop training if validation metrics worsen
5. **Save Checkpoints**: Keep multiple checkpoints to roll back if needed
6. **Document Your Setup**: Keep track of hyperparameters that work well

## Example: Complete Workflow

Here's a complete example from data to trained model:

```bash
# 1. Prepare data
python workflows/dataprep/custom_dataset_to_parquet.py \
    --base_dir="/data/my_dataset" \
    --audio_dir="audio" \
    --annotation_dir="annotations" \
    --output_dir="/output/my_parquet" \
    --corpus_name="my_corpus" \
    --language_code="eng_Latn" \
    --regions="region1,region2"

# 2. Create dataset card
cat > src/omnilingual_asr/cards/datasets/my_dataset.yaml << EOF
name: my_dataset
dataset_family: mixture_parquet_asr_dataset
dataset_config:
  data: /output/my_parquet/version=0
tokenizer_ref: omniASR_tokenizer
EOF

# 3. Verify dataset
python -m workflows.dataprep.dataloader_example \
    --dataset_path="/output/my_parquet/version=0" \
    --split="train" \
    --num_iterations=5

# 4. Create training config (my_config.yaml)
cat > my_config.yaml << EOF
model:
  name: "omniASR_CTC_300M"
dataset:
  name: "my_dataset"
  train_split: "train"
  valid_split: "dev"
  storage_mode: "MIXTURE_PARQUET"
  task_mode: "ASR"
  mixture_parquet_storage_config:
    dataset_summary_path: "/output/my_parquet/language_distribution_0.tsv"
    beta_corpus: 0.5
    beta_language: 0.5
  asr_task_config:
    max_audio_len: 960_000
    max_num_elements: 7_680_000
tokenizer:
  name: "omniASR_tokenizer"
optimizer:
  config:
    lr: 1e-05
trainer:
  grad_accumulation:
    num_batches: 4
regime:
  num_steps: 5_000
  validate_every_n_steps: 500
  checkpoint_every_n_steps: 500
EOF

# 5. Train
export OUTPUT_DIR="/output/training"
python -m workflows.recipes.wav2vec2.asr \
    $OUTPUT_DIR \
    --config-file my_config.yaml

# 6. Use the trained model
python -c "
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
pipeline = ASRInferencePipeline(
    model_path='$OUTPUT_DIR/checkpoints/step_5000/model.pt',
    model_card='omniASR_CTC_300M'
)
result = pipeline.transcribe(['test_audio.wav'])
print(result)
"
```

## Additional Resources

- [Data Preparation README](./README.md) - Detailed guide on data preparation
- [Training Recipes README](../recipes/wav2vec2/asr/README.md) - Training configuration options
- [Model Architectures](../../src/omnilingual_asr/models/README.md) - Model family details
- [Main README](../../README.md) - Project overview and installation

## Citation

If you use this guide or fine-tune Omnilingual ASR models for your research, please cite:

```bibtex
@misc{omnilingualasr2025,
    title={{Omnilingual ASR}: Open-Source Multilingual Speech Recognition for 1600+ Languages},
    author={{Omnilingual ASR Team} and Keren, Gil and Kozhevnikov, Artyom and others},
    year={2025},
    url={https://ai.meta.com/research/publications/omnilingual-asr-open-source-multilingual-speech-recognition-for-1600-languages/},
}
```
