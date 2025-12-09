import os
import json
import random
import torchaudio
from collections import defaultdict
from tqdm import tqdm

def  save_prompt(json_data, output_file):
    with open(output_file, 'w', encoding='utf8') as fo:
        for item in json_data:
            fo.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"提示数据已保存到： {output_file} \n")

def train_eval_split(data, train_output_file:str, eval_output_file:str, train_ratio:float=0.8, eval_sample_size:int=None, train_sample_size:int=None):
    """
    将数据集分为训练集和验证集。
    """
    # 打乱数据顺序    
    random.shuffle(data)
    if eval_sample_size is not None:
        split_index = eval_sample_size
    else:
        split_index = int(len(data) * (1-train_ratio))
    
    if train_sample_size is not None:
        end_index = train_sample_size + split_index
    else:
        end_index = len(data)

    train_data = data[split_index:end_index]
    eval_data = data[:split_index]
    
    save_prompt(train_data, train_output_file)
    save_prompt(eval_data, eval_output_file)
    print(f"初始数据集划分: 训练集 {len(train_data)} 条，验证集 {len(eval_data)} 条")

def check_audio_valid(wavpath:str):
    return True
    try:
        waveform, sample_rate = torchaudio.load(wavpath)
        duration = waveform.shape[1] / sample_rate
        if 0 < duration and sample_rate == 16000:
            return True
        else:
            return False
    except Exception as e:
        return False
    
def load_wavpath(wavpth:str) -> dict:
    wav_dict = dict()
    with open(wavpth, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="加载音频路径"):
            audio_key, audio_path = line.strip().split('\t', maxsplit=1)
            if check_audio_valid(audio_path):
                wav_dict[audio_key] = audio_path
    return wav_dict

def phoneme_pa_prompt_creator(wavptah:str, label:str, output_file:str):
    wav_dict = load_wavpath(wavptah)
    label_dict = defaultdict(list)

    with open(label, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            key, words, ipas, scores = line.strip().split('\t', maxsplit=3)
            words = words.split(' ')
            ipas = ipas.split(',')
            scores = scores.split(',')

            # ref_text = ' '.join(words)
            ref_text = []
            for w, p in zip(words, ipas):
                ref_text.append(f"{w}[{p}]")
            ref_text = ' '.join(ref_text)

            pred_content = []
            for w, p, s in zip(words, ipas, scores):
                pred_content.append(f"{w} [{p}]({s})")
            pred_text = '|'.join(pred_content)
            label_dict[key] = [ref_text, pred_text]
    
    prompts = []
    for key in label_dict:
        audio = wav_dict.get(key, None)
        if audio is None:
            continue
        ref_text, pred_text = label_dict[key]
        data = {
            "task_type": "understanding",
            "conversation":[
                {
                    "role": "user",
                    "message_type": "text",
                    # "content": f"评测文本:{ref_text}，请根据评测文本对下面的音频进行发音评分，并输出每个词的音素及其对应的发音得分。"
                    "content": f"评测文本:{ref_text}，请根据评测文本对下面的音频进行发音评分，并输出每个词的音素及其对应的发音得分。"
                },
                {
                    "role": "user",
                    "message_type": "audio",
                    "content": audio
                },
                {
                    "role": "assistant",
                    "message_type": "text",
                    "content": pred_text
                }
            ]
        }
        prompts.append(data)
    with open(output_file, 'w', encoding='utf-8') as fo:
        for item in prompts:
            fo.write(json.dumps(item, ensure_ascii=False) + '\n')


def wts_prompt_creator(snt_label, word_label, wavpath):
    wav_dict = load_wavpath(wavpath)
    label, prompts = defaultdict(list), list()
    with open(snt_label, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            key, words, snt_score = line.strip().split('\t', maxsplit=2)
            label[key].append(words)
            label[key].append(int(float(snt_score)))
    with open(word_label, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            key, words, score_list = line.strip().split('\t', maxsplit=2)
            score_list = [int(float(s)) for s in score_list.split(' ')]
            label[key].append(score_list)
    for key in label:
        audio = wav_dict.get(key, None)
        if audio is None:
            continue
        try:
            words, snt_score, word_scores = label[key]
            if len(words.split(' ')) != len(word_scores):
                print(f"标签数据异常，跳过样本 {key}，单词数量与评分数量不匹配")
                continue
        except Exception as e:
            # print(f"标签数据异常，跳过样本 {key}，错误信息：{e}")
            continue
        data = {
            "conversation":[
                {
                    "role": "user",
                    "message_type": "text",
                    "content": f"你是一个英文口语评测助手，根据音频和评测文本，评测句子整体发音准确性和每个单词的发音准确性[TASK:k12口语评测]，评测文本：{words}"
                },
                {
                    "role": "user",
                    "message_type": "audio",
                    "content": audio
                },
            ],
            "label": int(float(snt_score)),
            "label_words": word_scores
        }
        prompts.append(data)
    return prompts


if __name__ == "__main__":
    # # 训练集
    # wavpath = "/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/wavpath"
    # label = "/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/train/label_phones_ipa"
    # output_file = "/mnt/pfs_l2/jieti_team/SFT/hupeng/llm_data/kimi_style/sft/train/phoneme_pa_ipa_v2_train.jsonl"

    # # 测试集
    # wavpath = "/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/wavpath"
    # label = "/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/test/label_phones_ipa"
    # output_file = "/mnt/pfs_l2/jieti_team/SFT/hupeng/llm_data/kimi_style/sft/test/phoneme_pa_ipa_v2_test.jsonl"
    # phoneme_pa_prompt_creator(wavpath, label, output_file)


    snt_label = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/train/label_sent_score'
    word_label = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/train/label_word_accuracy_modify'
    wavpath = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/wavpath'
    prompts = wts_prompt_creator(snt_label, word_label, wavpath)

    train_output_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/llm_data/kimi_style/sft/train/wts_train.jsonl'
    eval_output_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/llm_data/kimi_style/sft/dev/wts_eval.jsonl'
    train_eval_split(prompts, train_output_file, eval_output_file, train_ratio=0.9)



