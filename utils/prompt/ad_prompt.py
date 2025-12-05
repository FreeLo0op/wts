import os
import sys
import json
import pandas as pd
import logging
from logging import getLogger

pd.set_option('display.max_columns', None)

logger = getLogger(__name__)
logger.setLevel("INFO")
logger.addHandler(logging.StreamHandler(sys.stdout))

CODE_MAP = {
    '正常': 0,
    '噪声': 1,
    '不相关中文': 2,
    '不相关英文': 3,
    '无意义语音': 4,
    '音量小': 5,
    '开头发音不完整': 6,
    '空音频': 7
}

def load_label(label_file:str):
    df = pd.read_csv(label_file, sep="\t")
    logger.info(f"Loaded {len(df)} rows from {label_file}")
    df['type'] = df['type'].replace('噪声,音量小', '噪声')
    df['type'] = df['type'].replace('噪声,开头发音不完整', '开头发音不完整')
    df['type'] = df['type'].replace('噪声,多说话人', '噪声')
    df['type'] = df['type'].replace('无意义语音,音量小', '无意义语音')
    df['type'] = df['type'].replace('不相关中文,不相关英文', '不相关中文')
    df['type'] = df['type'].replace('多说话人,开头发音不完整', '开头发音不完整')
    df['type'] = df['type'].replace('不相关中文,音量小', '不相关中文')
    df['type'] = df['type'].replace('不相关中文,噪声', '不相关中文')
    df['type'] = df['type'].replace('不相关英文,噪声', '不相关英文')
    df['type'] = df['type'].replace('噪声,空音频', '空音频')
    df['type'] = df['type'].replace('不相关英文,多说话人', '不相关英文')
    df['type'] = df['type'].replace('无意义语音,噪声', '无意义语音')
    df['type'] = df['type'].replace('多说话人,音量小', '音量小')
    
    df['type'] = df['type'].replace('音量小,开头发音不完整', '开头发音不完整')
    df['type'] = df['type'].replace('噪声,多说话人,开头发音不完整', '开头发音不完整')
    df['type'] = df['type'].replace('无意义语音,不相关中文', '无意义语音')
    df['type'] = df['type'].replace('噪声,音量小,开头发音不完整', '开头发音不完整')
    df['type'] = df['type'].replace('无意义语音,噪声,音量小', '无意义语音')
    df = df[df['type'].isin(CODE_MAP.keys())]
    logger.info(f"Filtered to {len(df)} valid rows")
    logger.info(f"Label distribution:\n{df['type'].value_counts()}")
    df['code'] = df['type'].map(CODE_MAP)
    logger.info(df.head(n=5))
    return df

def data_convertor(df:pd.DataFrame):
    all_prompt = []
    for _, row in df.iterrows():
        audio = row['wavpath']
        label = row['code']
        text = row['text']

        data = {
            "conversation":[
                {
                    "role": "user",
                    "message_type": "text",
                    "content": f"{text}"
                },
                {
                    "role": "user",
                    "message_type": "audio",
                    "content": audio
                },
            ],
            "label": int(float(label))
        }
        all_prompt.append(data)
    return all_prompt

def save(data:list, output_path:str):
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"Saved {len(data)} records to {output_path}")

def main(label_file:str, output_path:str):
    df = load_label(label_file)
    data = data_convertor(df)
    save(data, output_path)

if __name__ == "__main__":
    label_file = sys.argv[1]
    output_path = sys.argv[2]
    main(label_file, output_path)
    
