# Omnilingual ASR Fine-Tuning with Submission Generation

This script (`omniasr_finetune_submission.py`) provides an end-to-end workflow for:
1. Converting your regional CSV + WAV data to Omnilingual ASR format
2. Fine-tuning an Omnilingual ASR model on your data
3. Running inference on test files
4. Generating a submission file for Kaggle competitions

## Key Differences from Whisper/Unsloth Approach

| Feature | Original (Whisper/Unsloth) | This Script (Omnilingual ASR) |
|---------|----------------------------|-------------------------------|
| Model | Whisper Large V3 | omniASR_CTC_300M/1B/3B or LLM variants |
| Training | LoRA fine-tuning with Unsloth | Full parameter or LoRA with fairseq2 |
| Language Support | Single language (Bengali) | 1600+ languages including Bengali dialects |
| Data Format | In-memory format (slower) | Efficient parquet format (faster, scalable) |
| Batch Processing | Custom collator | Built-in fairseq2 batching |
| Inference | Manual batching | ASRInferencePipeline with optimizations |

## Usage

### Quick Start

```bash
# Run the entire workflow
python workflows/dataprep/omniasr_finetune_submission.py
```

The script is organized into sections that can be run step-by-step:

### Section 1: Installation

Uncomment the installation section to install dependencies:
```python
# Uncomment to run installation
install_dependencies()
```

### Section 2: Configuration

Set your paths and parameters at the top of the script:
```python
BASE_DIR = "/kaggle/input/shobdotori"  # Your data directory
MODEL_NAME = "omniASR_CTC_300M"         # Change to 1B, 3B, or LLM variants
LANGUAGE_CODE = "ben_Beng"              # Bengali in Bengali script
```

### Section 3-5: Data Preparation

The script automatically:
- Converts your CSV + WAV data to parquet format
- Creates a dataset asset card
- Generates training configuration

### Section 6: Training

Uncomment to start training:
```python
# Uncomment to run training
checkpoint_path = run_training()
```

Training parameters:
- **Duration**: ~15-30 minutes on a single GPU
- **Output**: Checkpoints saved to `./omniasr_output/training/checkpoints/`
- **Steps**: 5,000 (adjust in config based on your dataset size)

### Section 7: Inference & Submission

Uncomment to generate submission:
```python
# Uncomment to run inference
submission_path = generate_submission()
```

This will:
- Load your fine-tuned model
- Transcribe all test files in batches
- Generate `submission.csv` ready for Kaggle

## Model Selection

Choose based on your GPU memory:

| Model | Parameters | VRAM | Speed | Use Case |
|-------|-----------|------|-------|----------|
| `omniASR_CTC_300M` | 325M | ~2 GiB | Fast | Quick experiments, T4 GPU |
| `omniASR_CTC_1B` | 975M | ~3 GiB | Medium | Balanced performance |
| `omniASR_CTC_3B` | 3.1B | ~8 GiB | Slower | Best CTC quality, A100 |
| `omniASR_LLM_1B` | 2.3B | ~6 GiB | Slower | Multilingual with conditioning |

Change the model by editing:
```python
MODEL_NAME = "omniASR_CTC_1B"  # For better quality
```

## Expected Output

### Training Output

```
⚡ TRAINING CONFIGURATION
  Batch size: 4
  Gradient accumulation: 4
  Effective batch size: 16
  Total training steps: 5,000
  Expected time: 15-30 minutes
```

### Submission Output

```
SUBMISSION QUALITY CHECK
Total rows: 450
Expected files: 450
Empty texts: 0
Unique transcriptions: 387
✅ Good transcription diversity!
```

## Troubleshooting

### Out of Memory

1. Use a smaller model: `omniASR_CTC_300M`
2. Reduce `max_num_elements` in the config:
   ```python
   max_num_elements: 3_840_000  # Half the default
   ```
3. Increase gradient accumulation:
   ```python
   num_batches: 8  # Double the accumulation
   ```

### Poor Transcription Quality

1. Train for more steps:
   ```python
   num_steps: 10_000  # Double the steps
   ```
2. Use a larger model: `omniASR_CTC_1B` or `omniASR_LLM_1B`
3. Check validation WER during training
4. Ensure your audio quality is good

### Data Loading Errors

1. Verify your data paths:
   ```bash
   ls /kaggle/input/shobdotori/Train/Rajshahi/
   ls /kaggle/input/shobdotori/Train_annotation/
   ```
2. Check CSV format (should have 'audio' and 'text' columns)
3. Ensure audio files are valid WAV format

## Comparison with Original Approach

### What's Better

✅ **Faster training**: fairseq2 optimizations + efficient data pipeline
✅ **Better scaling**: Parquet format handles large datasets efficiently
✅ **More languages**: Works with 1600+ languages out of the box
✅ **Production-ready**: Built on Meta's production ASR stack
✅ **Model flexibility**: Easy to switch between CTC and LLM variants

### What's Different

⚠️ **Setup**: Requires parquet conversion step (but handled automatically)
⚠️ **Training interface**: Uses fairseq2 recipes instead of Transformers Trainer
⚠️ **Memory**: May need slightly more VRAM for same model size

## Integration Notes

This script preserves the structure of your original approach:
- ✅ Same data loading from regional CSVs
- ✅ Same regions list
- ✅ Same train/val split (95/5)
- ✅ Same submission format
- ✅ Same quality checks

But uses Omnilingual ASR under the hood for better performance and language support.

## Next Steps

1. **Run data conversion**: Let the script convert your data
2. **Start training**: Uncomment `run_training()` and monitor progress
3. **Generate submission**: Uncomment `generate_submission()` after training
4. **Upload to Kaggle**: Submit the generated `submission.csv`

## Support

- **Quick Start**: `workflows/dataprep/QUICKSTART.md`
- **Full Guide**: `workflows/dataprep/CUSTOM_DATASET_GUIDE.md`
- **Training Recipes**: `workflows/recipes/wav2vec2/asr/README.md`

Happy fine-tuning! 🎙️🚀
