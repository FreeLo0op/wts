import re
import sys
import ast
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def read_label_file(label_file:str, if_word:bool=False):
    label_dict = {}
    with open(label_file, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
        if re.findall('[0-9]', lines[0]) == []:
            lines = lines[1:]
        for line in lines:
            line = line.strip().split('\t')
            try:
                key, score = line[0], float(line[1])
            except:
                key, score = line[0], float(line[2])
            if if_word:
                key, score = line[0], line[2]
                score = score.split(' ')
                score = [float(s) for s in score]
            label_dict[key] = score
    return label_dict

def read_pred_file(pred_file:str):
    pred_dict = {}
    with open(pred_file, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.strip().split('\t')
            try:
                key, score = line[0], float(line[2])
            except Exception as e:
                print(f"Error processing line: {line}, error: {e}")
                sys.exit(1)
            # word_score = ast.literal_eval(word_score)
            # if '{' in score:
            #     continue
            pred_dict[key] = score
    return pred_dict

def pcc_compute(true_list:list, pred_list:list, dataset_name:str=None):
    """
    Compute the Pearson correlation coefficient (PCC) between two lists.
    """
    assert len(true_list) == len(pred_list), f"Length mismatch: {len(true_list)} != {len(pred_list)}"
    print(f"真实标签长度: {len(true_list)}, 预测标签长度: {len(pred_list)}")
    
    # Convert elements to float
    true_list = [float(x) for x in true_list]
    pred_list = [float(x) for x in pred_list]
    
    true_list = np.array(true_list)
    pred_list = np.array(pred_list)

    if np.all(true_list == true_list[0]) and np.all(pred_list == pred_list[0]):
        pcc = 1.0 if np.all(true_list == pred_list) else 0.0
    
    pcc = np.corrcoef(true_list, pred_list)[0, 1]

    if dataset_name:
        print(f"[{dataset_name}] PCC: {pcc:.4f}")
    else:
        print(f"PCC: {pcc:.4f}")


    return pcc

def snt_main(
    label_file: str,
    pred_file: str
    ):
    label_dict = read_label_file(label_file)
    pred_dict = read_pred_file(pred_file)
    true_list, pred_list = [], []
    for key in label_dict:
        if key in pred_dict:
            pred_snt_score = pred_dict[key]
            true_snt_score = label_dict[key]
            true_list.append(true_snt_score)
            pred_list.append(pred_snt_score)
    dataset_neme = label_file.split('/')[-1]
    pcc = pcc_compute(true_list, pred_list, dataset_name=dataset_neme)

def word_main(
    label_file: str,
    pred_file: str
    ):
    label_dict = read_label_file(label_file, if_word=True)
    pred_dict = read_pred_file(pred_file)
    true_word_list, pred_word_list = [], []
    for key in label_dict:
        if key in pred_dict:
            pred_word_score = pred_dict[key][1]
            true_word_score = label_dict[key]
            if len(pred_word_score) != len(true_word_score):
                print(f"Warning: Key {key} has length mismatch between true and predicted word scores. Skipping.")
                continue
            true_word_list.extend(true_word_score)
            pred_word_list.extend(pred_word_score)
    pcc_word = pcc_compute(true_word_list, pred_word_list, dataset_name="word")

if __name__ == "__main__":
    # 业务测试集
    label_files = [
        '/mnt/cfs/workspace/speech/luohaixia/evl_data/en/test_data/10_score/ledu_word/vegas_label_1216/ledu_word_final_score',
        '/mnt/cfs/workspace/speech/luohaixia/evl_data/en/test_data/10_score/ledu_snt/vegas_label_1216/ledu_snt_final_score',
        '/mnt/cfs/workspace/speech/luohaixia/evl_data/en/test_data/10_score/tal_xpad/vegas_label_1216/tal_xpad_final_score',
        '/mnt/cfs/workspace/speech/luohaixia/evl_data/en/test_data/10_score/abc-reading/vegas_label_1216/abc-reading_final_score',
        '/mnt/cfs/workspace/speech/luohaixia/evl_data/en/test_data/10_score/xes_tushu/vegas_label_1216/xes_tushu_final_score',
        '/mnt/cfs/workspace/speech/luohaixia/evl_data/en/test_data/10_score/xiaohou/vegas_label_1216/xiaohou_final_score',
        '/mnt/cfs/workspace/speech/luohaixia/evl_data/en/test_data/10_score/xpad_libu/vegas_label_1216/xpad_libu_final_score',
        '/mnt/cfs/workspace/speech/luohaixia/evl_data/en/test_data/10_score/dabanyun/vegas_label_1216/dabanyun_final_score',
        '/mnt/cfs/workspace/speech/luohaixia/evl_data/en/test_data/10_score/xiao_e_tong/vegas_label_1216/xiao_e_tong_final_score'
    ]

    pred_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/ts_model/wt6s_snt_word_acc_model_v3.1/infer_res/all_10_score_test.res'
    for label_file in label_files:
        snt_main(
            label_file=label_file,
            pred_file=pred_file
        )
    
    # 异常音频测试集batch1 & batch2
    label_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/test/label_snt_score_batch1'
    pred_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/ts_model/wt6s_snt_word_acc_model_v3.1/infer_res/next-250919.res'
    snt_main(
        label_file=label_file,
        pred_file=pred_file
    )

    label_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/test/label_snt_score_batch2'
    pred_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/ts_model/wt6s_snt_word_acc_model_v3.1/infer_res/next-251013.res'
    snt_main(
        label_file=label_file,
        pred_file=pred_file
    )

    # tal-k12 测试集
    label_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/test/label_sent_score'
    pred_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/ts_model/wt6s_snt_word_acc_model_v3.1/infer_res/tal-k12_acc.res'
    snt_main(
        label_file=label_file,
        pred_file=pred_file
    )

    # word_main(
    #     label_file=word_label_file,
    #     pred_file=pred_file
    # )

    