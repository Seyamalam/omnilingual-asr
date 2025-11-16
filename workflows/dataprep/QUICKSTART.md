# Quick Start: Fine-Tuning with Your Data

This is a quick reference for fine-tuning Omnilingual ASR models with custom WAV data. For the complete guide, see [CUSTOM_DATASET_GUIDE.md](./CUSTOM_DATASET_GUIDE.md).

## Your Data Format (from the problem statement)

You have:
- Audio files (`.wav`) organized by regions
- CSV files with audio filenames and transcriptions
- Structure like:
  ```
  /kaggle/input/shobdotori/
  ├── Train/
  │   ├── Rajshahi/
  │   │   ├── audio001.wav
  │   │   └── ...
  │   ├── Kushtia/
  │   └── ...
  └── Train_annotation/
      ├── Rajshahi.csv
      ├── Kushtia.csv
      └── ...
  ```

## Quick Workflow

### 1. Convert Your Data (5-10 minutes)

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

This will create:
- Parquet dataset at `/output/shobdotori_parquet/version=0/`
- Statistics file at `/output/shobdotori_parquet/language_distribution_0.tsv`

### 2. Create Dataset Card (1 minute)

```bash
cat > src/omnilingual_asr/cards/datasets/shobdotori.yaml << 'EOF'
name: shobdotori
dataset_family: mixture_parquet_asr_dataset
dataset_config:
  data: /output/shobdotori_parquet/version=0
tokenizer_ref: omniASR_tokenizer
EOF
```

### 3. Verify Dataset (2 minutes)

```bash
python -m workflows.dataprep.dataloader_example \
    --dataset_path="/output/shobdotori_parquet/version=0" \
    --split="train" \
    --num_iterations=5
```

### 4. Create Training Config (2 minutes)

```bash
cat > shobdotori_finetune.yaml << 'EOF'
model:
  name: "omniASR_CTC_300M"

dataset:
  name: "shobdotori"
  train_split: "train"
  valid_split: "dev"
  storage_mode: "MIXTURE_PARQUET"
  task_mode: "ASR"
  mixture_parquet_storage_config:
    dataset_summary_path: "/output/shobdotori_parquet/language_distribution_0.tsv"
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
```

### 5. Start Training

```bash
export OUTPUT_DIR="/output/training"

python -m workflows.recipes.wav2vec2.asr \
    $OUTPUT_DIR \
    --config-file shobdotori_finetune.yaml
```

## Model Selection

Based on your hardware:

| GPU Memory | Recommended Model | VRAM Usage | Training Speed |
|------------|------------------|------------|----------------|
| 8 GB       | `omniASR_CTC_300M` | ~2 GiB | Fast |
| 16 GB      | `omniASR_CTC_1B` | ~3 GiB | Medium |
| 24 GB+     | `omniASR_CTC_3B` | ~8 GiB | Slower |
| 40 GB+     | `omniASR_LLM_1B` | ~6 GiB | Slowest, Best Quality |

For multilingual with language conditioning, use `omniASR_LLM_*` models instead of `omniASR_CTC_*`.

## Common Issues

### Out of Memory
Reduce `max_num_elements` in config:
```yaml
asr_task_config:
  max_num_elements: 3_840_000  # Half the default
```

### Poor Performance
- Increase training steps: `num_steps: 10_000`
- Try higher learning rate: `lr: 5e-05`
- Use a larger model: `omniASR_CTC_1B` instead of `300M`

### Files Not Found
Check that:
- CSV files exist in `Train_annotation/`
- Audio files exist in `Train/[Region]/`
- Paths in CSV match actual audio filenames

## Understanding Your Code Snippet

Your data loading code showed:
```python
# You were using librosa and tokenizer directly
audio = load_audio_fast(audio_path)
features = tokenizer.feature_extractor(audio, sampling_rate=16000)
tokens = tokenizer.tokenizer(text)
```

The parquet converter does this automatically:
- Loads audio with proper resampling to 16kHz
- Normalizes text for the language
- Saves in compressed format (OGG/FLAC)
- Creates train/validation splits

The training recipe then:
- Loads parquet batches efficiently
- Extracts features using the model's feature extractor
- Tokenizes text using the specified tokenizer
- Handles batching and shuffling
- Trains with mixed precision (bfloat16)

## Next Steps

1. **Read the full guide**: [CUSTOM_DATASET_GUIDE.md](./CUSTOM_DATASET_GUIDE.md)
2. **Check training recipes**: [Training README](../recipes/wav2vec2/asr/README.md)
3. **Explore model architectures**: [Models README](../../src/omnilingual_asr/models/README.md)

## Support

If you encounter issues:
1. Check the troubleshooting section in [CUSTOM_DATASET_GUIDE.md](./CUSTOM_DATASET_GUIDE.md)
2. Verify all paths and file permissions
3. Start with a small subset of data for testing
4. Monitor GPU memory and adjust batch size accordingly

## Example Output

After training completes, use your model:

```python
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

pipeline = ASRInferencePipeline(
    model_path='/output/training/checkpoints/step_5000/model.pt',
    model_card='omniASR_CTC_300M'
)

# Transcribe your audio
transcription = pipeline.transcribe(['test_audio.wav'])
print(transcription)
```

Happy fine-tuning! 🎙️🚀
