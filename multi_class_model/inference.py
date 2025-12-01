import argparse
import torch
import os
import numpy as np
from multi_class_model.modeling_multiclass import MultiClassModel
from multi_class_model.configuration_multiclass import MultiClassConfig
from kimia_infer.api.prompt_manager import KimiAPromptManager
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/ts_model/output_multiclass_flu', help="Path to the trained model checkpoint")
    parser.add_argument("--base_model_path", type=str, default='/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/llm-base-models/Kimi-Audio-7B', help="Path to the base model (for tokenizer and whisper init)")
    # parser.add_argument("--audio_path", type=str, required=True, help="Path to the input audio file")
    parser.add_argument("--text_prompt", type=str, default="请对这段音频进行评分（0-10分）", help="Text prompt for the model")
    parser.add_argument("--num_labels", type=int, default=None, help="Number of classification labels")
    parser.add_argument("--audio_tokenizer_path", type=str, default="/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/llm-base-models/THUDM/glm-4-voice-tokenizer", help="Path to audio tokenizer")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load Config
    config = MultiClassConfig.from_pretrained(args.model_path)
    if args.num_labels is not None:
        config.num_labels = args.num_labels
    
    # Load Model
    print(f"Loading model from {args.model_path}...")
    model = MultiClassModel.from_pretrained(args.model_path, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.to(args.device)
    model.eval()

    # Initialize Prompt Manager for preprocessing
    # Use base_model_path for tokenizer and initial whisper loading
    prompt_manager = KimiAPromptManager(
        model_path=args.base_model_path,
        audio_tokenizer=args.audio_tokenizer_path,
        kimia_token_offset=config.kimia_token_offset,
        kimia_text_audiodelaytokens=config.kimia_mimo_audiodelaytokens,
        device=args.device
    )
    
    # Replace prompt_manager's whisper model with the one from the loaded model
    # This ensures we use the correct weights (if fine-tuned) and saves memory
    if hasattr(model, 'whisper_model'):
        prompt_manager.whisper_model = model.whisper_model
        prompt_manager.whisper_model.bfloat16()
    else:
        print("Warning: Model does not have whisper_model attribute. Using default loaded by PromptManager.")

    # Construct Chat
    # We simulate a user sending audio + text
    # The prompt manager handles audio tokenization if "audio_tokens" is missing but "content" is path.
    # file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/test/label_sent_score'
    # wavpath = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/wavpath'
    # prompt = '你是一个英文口语评测助手，根据音频和评测文本，评测句子整体发音准确性，评分标准为0-10分。[TASK:k12口语评测]'
    file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/next_fluency/test/label_sent_score'
    wavpath = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/next_fluency/test/wavpath'
    prompt = '你是一个英文口语评测助手，根据音频和评测文本，评测句子整体发音流利度，评分标准为0-3分。[TASK:k12口语评测]'

    # file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/speechocean762/test/label_sent_score'
    # wavpath = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/speechocean762/test/wavpath'
    # prompt = '你是一个英文口语评测助手，根据音频和评测文本，评测句子整体发音准确性，评分标准为0-10分。[TASK:成人口语评测]'
    # prompt = '你是一个英文口语评测助手，根据音频和评测文本，评测句子整体发音流利度，评分标准为0-10分。[TASK:成人口语评测]'
    wav_map = {}
    with open(wavpath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            key, path = line.strip().split('\t', maxsplit=1)
            wav_map[key] = path
    with open(file, 'r') as fin:
        lines = fin.readlines()
        for line in tqdm(lines):
            line = line.strip().split('\t')
            key, text = line[0], line[1]
            audio = wav_map.get(key, None)
            if not audio or not os.path.exists(audio):
                continue
            chats = [
                {
                    "role": "user",
                    "content": f"{prompt} ,参考文本:{text}",
                    # "content": f"你是一个英文口语评测助手，根据音频和评测文本，评测句子整体发音准确性，评分标准为0-10分。[TASK:k12口语评测] ,参考文本:{text}",
                    # "content": f"你是一个英文口语评测助手，根据音频和评测文本，评测句子整体发音准确性，评分标准为0-10分。[TASK:成人口语评测] ,参考文本:{text}",
                    "message_type": "text"
                },
                {
                    "role": "user",
                    "content": audio,
                    "message_type": "audio"
                },
            ]
            
            # Get input tensors
            # output_type="text" because we are doing understanding/classification, not generating audio
            history = prompt_manager.get_prompt(chats, output_type="text", add_assistant_start_msg=True)
            
            audio_input_ids, text_input_ids, is_continuous_mask, _, _ = history.to_tensor()
            audio_features = history.continuous_feature
            
            # Move to device and add batch dimension
            # audio_input_ids = audio_input_ids.unsqueeze(0).to(args.device)
            # text_input_ids = text_input_ids.unsqueeze(0).to(args.device)
            # is_continuous_mask = is_continuous_mask.unsqueeze(0).to(args.device)
            audio_input_ids = audio_input_ids.to(args.device)
            text_input_ids = text_input_ids.to(args.device)
            is_continuous_mask = is_continuous_mask.to(args.device)
            # Audio features: list of tensors -> tensor with batch dim
            # Assuming one audio feature per sample (from the audio message)
            # prompt_manager.extract_whisper_feat returns [1, T, D]
            # history.continuous_feature is a list of [1, T, D]
            if audio_features:
                # Stack or pad? Here we have only one sample.
                # If multiple features (e.g. multiple audio turns), we might need to concat or handle as list?
                # The model expects `whisper_input_feature` as tensor [B, T, D] or similar.
                # In datasets.py, we padded features.
                # Here we have 1 sample.
                whisper_input_feature = torch.cat(audio_features, dim=0).unsqueeze(0).to(args.device) 
                # Wait, extract_whisper_feat returns [1, T/4, D*4].
                # If we have multiple audios, we might have multiple features.
                # But usually for classification we classify one audio.
                # If we have multiple, we concat along time?
                # Let's assume one audio.
                whisper_input_feature = audio_features[0].unsqueeze(0).to(args.device)
            else:
                whisper_input_feature = None

            # Inference
            with torch.no_grad():
                outputs = model(
                    input_ids=audio_input_ids,
                    text_input_ids=text_input_ids,
                    whisper_input_feature=whisper_input_feature,
                    is_continuous_mask=is_continuous_mask
                )
                
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                score = probabilities[0].tolist()

            # print(f"Predicted Class: {predicted_class}")
            # print(f"Probabilities: {score}")
            print(f"{key}\t{score}\t{predicted_class}")

if __name__ == "__main__":
    main()
