import argparse
import torch
import os
import sys
import time
import librosa
from multi_class_model.modeling_multiclass import MultiClassModel
from multi_class_model.configuration_multiclass import MultiClassConfig
from kimia_infer.api.prompt_manager import KimiAPromptManager
from tqdm import tqdm

def infer(model, prompt_manager, audio_path, text, prompt):
    chats = [
        {
            "role": "user",
            "content": f"{prompt},参考文本:{text}",
            "message_type": "text"
        },
        {
            "role": "user",
            "content": audio_path,
            "message_type": "audio"
        },
    ]
    
    history = prompt_manager.get_prompt(chats, output_type="text", add_assistant_start_msg=True)
    audio_input_ids, text_input_ids, is_continuous_mask, _, _ = history.to_tensor()
    audio_features = history.continuous_feature
    audio_input_ids = audio_input_ids.to(model.device)
    text_input_ids = text_input_ids.to(model.device)
    is_continuous_mask = is_continuous_mask.to(model.device)
    if audio_features:
        whisper_input_feature = torch.cat(audio_features, dim=0).unsqueeze(0).to(model.device) 
        whisper_input_feature = audio_features[0].unsqueeze(0).to(model.device)
    else:
        whisper_input_feature = None
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
        probabilities = probabilities[0].tolist()
    return predicted_class, probabilities

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/ts_model/output_multiclass_acc', help="Path to the trained model checkpoint")
    parser.add_argument("--base_model_path", type=str, default='/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/llm-base-models/Kimi-Audio-7B', help="Path to the base model (for tokenizer and whisper init)")
    parser.add_argument("--audio_tokenizer_path", type=str, default="/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/llm-base-models/THUDM/glm-4-voice-tokenizer", help="Path to audio tokenizer")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load Config
    config = MultiClassConfig.from_pretrained(args.model_path)
    id2label = config.id2label
    config.num_labels = len(id2label)
    
    # Load Model
    model = MultiClassModel.from_pretrained(args.model_path, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.to(args.device)
    model.eval()

    prompt_manager = KimiAPromptManager(
        model_path=args.base_model_path,
        audio_tokenizer=args.audio_tokenizer_path,
        kimia_token_offset=config.kimia_token_offset,
        kimia_text_audiodelaytokens=config.kimia_mimo_audiodelaytokens,
        device=args.device
    )
    
    if hasattr(model, 'whisper_model'):
        prompt_manager.whisper_model = model.whisper_model
        prompt_manager.whisper_model.bfloat16()
    else:
        print("Warning: Model does not have whisper_model attribute. Using default loaded by PromptManager.")

    # file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/next_fluency/test/label_sent_score'
    # wavpath = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/next_fluency/test/wavpath'
    # prompt = '你是一个英文口语评测助手，根据音频和评测文本，评测句子整体发音流利度，评分标准为0-3分。[TASK:k12口语评测]'
    infer_data = [
        # ('tal-k12_acc', '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/test/label_sent_score', '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/wavpath', '评测句子发音准确性，评分标准为0-10分。[TASK:k12口语评测]'),
        # ('tal-k12_acc', '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/test/label_sent_score', '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/wavpath', '你是一个英文口语评测助手，根据音频和评测文本，评测句子整体发音准确性，评分标准为0-10分。[TASK:k12口语评测] '),
        ('all_10_score_test', '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/test/label_snt_all_10_score_test', '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/wavs/wavpath_all_10', '你是一个英文口语评测助手，根据音频和评测文本，评测句子整体发音准确性，评分标准为0-10分。[TASK:k12口语评测] '),
        # ('speechocean762_acc', '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/speechocean762/test/label_sent_score', '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/speechocean762/test/wavpath', '评测句子发音准确性，评分标准为0-10分。[TASK:成人口语评测]'),
        # ('speechocean762_fluency', '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/speechocean762/test/label_sent_score', '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/speechocean762/test/wavpath', '评测句子发音流利度，评分标准为0-10分。[TASK:成人口语评测]'),
    ]

    for dataset_name, file, wavpath, prompt in infer_data:
        ouput_file = os.path.join(args.model_path, 'infer_res', f"{dataset_name}.res2")
        if not os.path.exists(os.path.dirname(ouput_file)):
            os.makedirs(os.path.dirname(ouput_file))
        fo = open(ouput_file, 'w', encoding='utf-8')
        wav_map = {}
        with open(wavpath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                key, path = line.strip().split('\t', maxsplit=1)
                wav_map[key] = path
        total_time, total_dur = 0.0, 0.0
        with open(file, 'r') as fin:
            lines = fin.readlines()
            for line in tqdm(lines):
                try:
                    line = line.strip().split('\t')
                    key, text = line[0], line[1]
                    audio = wav_map.get(key, None)
                    if not audio or not os.path.exists(audio):
                        continue
                    audio_data, sr = librosa.load(audio, sr=16000)
                    duration = len(audio_data) / sr
                    total_dur += duration
                    time_start = time.time()
                    predicted_class, probabilities = infer(model, prompt_manager, audio, text, prompt)
                    end_time = time.time()
                    # print(f'start:{time_start}, end:{end_time}, used:{end_time - time_start}, dur:{duration}, rtf:{(end_time - time_start)/duration}')
                    used_time = end_time - time_start
                    total_time += used_time
                except Exception as e:
                    print(f"Error processing {key}: {e}")
                    predicted_class = -1
                    probabilities = []

                fo.write(f'{key}\t{probabilities}\t{predicted_class}\n')
        fo.flush()
        fo.close()
        print(f"Dataset: {dataset_name}, Total Inference Time: {total_time:.4f}s, Total Audio Duration: {total_dur:.4f}s, Real-time Factor: {total_time/total_dur:.4f}")

def ad_main():
    model_path = '/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/ad_model/ad_model_v2.0'
    base_model_path = '/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/llm-base-models/Kimi-Audio-7B'
    audio_tokenizer_path = '/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/llm-base-models/THUDM/glm-4-voice-tokenizer'
    device = 'cuda:0'

    CODE_MAP = {
        0: '正常',  
        1: '噪声',
        2: '不相关中文',
        3: '不相关英文',
        4: '无意义语音',
        5: '音量小',
        6: '开头发音不完整',
        7: '空音频'
    }

    config = MultiClassConfig.from_pretrained(model_path)
    model = MultiClassModel.from_pretrained(model_path, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()

    prompt_manager = KimiAPromptManager(
        model_path=base_model_path,
        audio_tokenizer=audio_tokenizer_path,
        kimia_token_offset=config.kimia_token_offset,
        kimia_text_audiodelaytokens=config.kimia_mimo_audiodelaytokens,
        device=device
    )
    if hasattr(model, 'whisper_model'):
        prompt_manager.whisper_model = model.whisper_model
        prompt_manager.whisper_model.bfloat16()
    else:
        print("Warning: Model does not have whisper_model attribute. Using default loaded by PromptManager.")

    test_data = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/audio_detect/test/label_batch2.csv'
    with open(test_data, 'r', encoding='utf-8') as fin:
        infer_data = fin.readlines()[1:]
    for data in tqdm(infer_data, total=len(infer_data)):
        key, text, label_true, audio = data.strip().split('\t', maxsplit=3)

        data = [
            {
                "role": "user",
                "message_type": "text",
                "content": text
            },
            {
                "role": "user",
                "message_type": "audio",
                "content": audio
            }
        ]
        
        history = prompt_manager.get_prompt(data, output_type="text", add_assistant_start_msg=True)
        audio_input_ids, text_input_ids, is_continuous_mask, _, _ = history.to_tensor()
        audio_features = history.continuous_feature
        audio_input_ids = audio_input_ids.to(device)
        text_input_ids = text_input_ids.to(device)
        is_continuous_mask = is_continuous_mask.to(device)
        if audio_features:
            whisper_input_feature = torch.cat(audio_features, dim=0).unsqueeze(0).to(device) 
            whisper_input_feature = audio_features[0].unsqueeze(0).to(device)
        else:
            whisper_input_feature = None
        
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
            probabilities = probabilities[0].tolist()
            predicted_label = CODE_MAP[int(predicted_class)]
            print(f"{key}\t{probabilities}\t{predicted_label}")

if __name__ == "__main__":
    # ad_main()
    main()