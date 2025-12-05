import sys
from torch.utils.data import Dataset
from functools import lru_cache
import torch
from typing import Dict, List
import numpy as np
import librosa
from kimia_infer.utils.special_tokens import instantiate_extra_tokens
from kimia_infer.utils.data import KimiAContent

class MultiClassDataset(Dataset):
    """Dataset for multi-class classification."""

    def __init__(self, raw_data_list, whisper_model, text_tokenizer, max_len: int, kimia_token_offset: int):
        super(MultiClassDataset, self).__init__()
        self.whisper_model = whisper_model
        self.max_len = max_len
        
        print("There are {} samples in the dataset".format(len(raw_data_list)))
        self.text_tokenizer = text_tokenizer

        self.extra_tokens = instantiate_extra_tokens(self.text_tokenizer)

        self.pad_token = self.extra_tokens.pad
        self.kimia_token_offset = kimia_token_offset
        self.raw_data = raw_data_list

    def __len__(self):
        return len(self.raw_data)
    
    def extract_whisper_feat(self, wav: str):
        # Assuming wav is a path or bytes, load it
        # This part depends on how whisper_model expects input or if we pre-process
        # Reusing logic from original dataset
        wav, _ = librosa.load(wav, sr=16000)
        return wav

    def _tokenize_text(self, text):
        if text is None:
            return None
        token_ids = self.text_tokenizer.encode(text, bos=False, eos=False)
        return token_ids

    def process_sample(self, sample):
        # Assuming sample has 'audio_path', 'text', 'label'
        # Or 'conversation' format. Let's assume conversation format but we classify the whole conversation or the last response.
        # For simplicity, let's assume the input is a conversation and we want to classify it.
        
        conversation = sample.get("conversation", [])
        label = sample.get("label", 0) # Integer label

        # Reuse KimiAContent logic to tokenize conversation
        # We need to import tokenize_conversation logic or reimplement simplified version
        # I'll implement a simplified version that concatenates everything
        
        kimia_content = KimiAContent()
        
        for message in conversation:
            role = message["role"]
            if role == "user":
                kimia_content.audio_append(self.extra_tokens.kimia_user_msg_start)
                kimia_content.text_append(self.extra_tokens.kimia_text_blank)
            elif role == "assistant":
                kimia_content.audio_append(self.extra_tokens.kimia_assistant_msg_start)
                kimia_content.text_append(self.extra_tokens.kimia_text_blank)
            
            if message.get("message_type") == "text":
                text_tokens = self._tokenize_text(message["content"])
                kimia_content.text_extend(text_tokens)
                kimia_content.audio_extend([self.extra_tokens.kimia_text_blank] * len(text_tokens))
            elif message.get("message_type") == "audio":
                audio_tokens = message.get("audio_tokens", [])
                kimia_content.audio_append(self.extra_tokens.media_begin)
                kimia_content.audio_extend(audio_tokens, is_continuous=True)
                kimia_content.audio_append(self.extra_tokens.media_end)
                kimia_content.text_extend([self.extra_tokens.kimia_text_blank] * (len(audio_tokens) + 2))
                
                # Extract whisper feature
                if "content" in message and message["content"]:
                    wav_data = self.extract_whisper_feat(message["content"])
                    kimia_content.continuous_feature.append(wav_data)

        # Auto-append assistant start if not present
        if conversation and conversation[-1]["role"] != "assistant":
            kimia_content.audio_append(self.extra_tokens.kimia_assistant_msg_start)
            kimia_content.text_append(self.extra_tokens.kimia_text_blank)

        # Convert to tensor
        audio_input_ids, text_input_ids, is_continuous_mask, _, _ = kimia_content.to_tensor()
        audio_features = kimia_content.continuous_feature

        if audio_input_ids.shape[1] > self.max_len:
            print("Truncating input to max_len(512):", audio_input_ids.shape[1])
            audio_input_ids = audio_input_ids[:, :self.max_len]
            text_input_ids = text_input_ids[:, :self.max_len]
            is_continuous_mask = is_continuous_mask[:, :self.max_len]

        return {
            "input_ids": audio_input_ids,
            "text_input_ids": text_input_ids,
            "whisper_input_feature": audio_features,
            "is_continuous_mask": is_continuous_mask,
            "labels": torch.tensor(label, dtype=torch.long)
        }

    def __getitem__(self, i):
        return self.process_sample(self.raw_data[i])

    def collate_fn(self, batch):
        if not batch:
            return {}

        max_len = max(s['input_ids'].shape[1] for s in batch)
        
        input_ids_batch = []
        text_input_ids_batch = []
        is_continuous_mask_batch = []
        labels_batch = []
        whisper_features = []

        for sample in batch:
            pad_len = max_len - sample['input_ids'].shape[1]
            
            input_ids_batch.append(torch.nn.functional.pad(sample['input_ids'], (0, pad_len), value=self.pad_token).squeeze(0))
            text_input_ids_batch.append(torch.nn.functional.pad(sample['text_input_ids'], (0, pad_len), value=self.pad_token).squeeze(0))
            is_continuous_mask_batch.append(torch.nn.functional.pad(sample['is_continuous_mask'], (0, pad_len), value=False).squeeze(0))
            labels_batch.append(sample['labels'])
            
            if sample['whisper_input_feature']:
                # Assuming one feature per sample or we flatten them
                # Original code appends all features
                for f in sample['whisper_input_feature']:
                    whisper_features.append(f)

        # Pad whisper features
        if whisper_features:
            max_feat_len = max(f.shape[0] for f in whisper_features)
            padded_features = []
            for feat in whisper_features:
                pad_len = max_feat_len - feat.shape[0]
                padded_feat = np.pad(feat, (0, pad_len), 'constant', constant_values=0)
                padded_features.append(padded_feat)
            whisper_input_feature_tensor = torch.tensor(np.array(padded_features), dtype=torch.float32)
        else:
            whisper_input_feature_tensor = torch.empty(0)

        return dict(
            input_ids=torch.stack(input_ids_batch),
            text_input_ids=torch.stack(text_input_ids_batch),
            whisper_input_feature=whisper_input_feature_tensor,
            is_continuous_mask=torch.stack(is_continuous_mask_batch),
            labels=torch.stack(labels_batch)
        )
