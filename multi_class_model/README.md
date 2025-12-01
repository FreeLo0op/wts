# Multi-Class Classification Model

This project implements a multi-class classification model based on the KimiAudio architecture.
It takes text tokens, audio tokens, and audio features (Whisper) as input, passes them through the KimiAudio Transformer backbone, and outputs a classification label.

## Structure

- `configuration_multiclass.py`: Configuration class for the model.
- `modeling_multiclass.py`: The model definition (`MultiClassModel`).
- `datasets.py`: Dataset class (`MultiClassDataset`) for loading and processing data.
- `train.py`: Training script using HuggingFace Trainer.

## Usage

### Training

To train the model, run the `train.py` script with appropriate arguments.

```bash
python -m multi_class_model.train \
    --model_path /path/to/pretrained/kimiaudio/model \
    --train_data_path /path/to/train.jsonl \
    --eval_data_path /path/to/eval.jsonl \
    --output_dir ./output \
    --num_labels 2 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-5 \
    --num_train_epochs 3
```

### Data Format

The training data should be in JSONL format, where each line is a JSON object containing:
- `conversation`: A list of messages (similar to the original KimiAudio format).
- `label`: An integer representing the class label.

Example:
```json
{
    "conversation": [
        {"role": "user", "message_type": "text", "content": "Hello"},
        {"role": "assistant", "message_type": "audio", "audio_tokens": [...], "content": "/path/to/audio.wav"}
    ],
    "label": 1
}
```
