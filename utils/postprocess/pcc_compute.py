
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def read_label_file(label_file:str):
    label_dict = {}
    with open(label_file, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.strip().split('\t')
            key, score = line[0], line[2]
            label_dict[key] = score
    return label_dict

def read_pred_file(pred_file:str):
    pred_dict = {}
    with open(pred_file, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.strip().split('\t')
            key, score = line[0], line[1]
            if '{' in score:
                continue
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


if __name__ == "__main__":
    # label_file = '/mnt/cfs/workspace/speech/luohaixia/evl_data/en/test_data/10_score/xpad_libu/vegas_label_1216/xpad_libu_final_score'
    # label_files = [
    #     '/mnt/cfs/workspace/speech/luohaixia/evl_data/en/test_data/10_score/ledu_word/vegas_label_1216/ledu_word_final_score',
    #     '/mnt/cfs/workspace/speech/luohaixia/evl_data/en/test_data/10_score/ledu_snt/vegas_label_1216/ledu_snt_final_score',
    #     '/mnt/cfs/workspace/speech/luohaixia/evl_data/en/test_data/10_score/tal_xpad/vegas_label_1216/tal_xpad_final_score',
    #     '/mnt/cfs/workspace/speech/luohaixia/evl_data/en/test_data/10_score/abc-reading/vegas_label_1216/abc-reading_final_score',
    #     '/mnt/cfs/workspace/speech/luohaixia/evl_data/en/test_data/10_score/xes_tushu/vegas_label_1216/xes_tushu_final_score',
    #     '/mnt/cfs/workspace/speech/luohaixia/evl_data/en/test_data/10_score/xiaohou/vegas_label_1216/xiaohou_final_score',
    #     '/mnt/cfs/workspace/speech/luohaixia/evl_data/en/test_data/10_score/xpad_libu/vegas_label_1216/xpad_libu_final_score',
    #     '/mnt/cfs/workspace/speech/luohaixia/evl_data/en/test_data/10_score/dabanyun/vegas_label_1216/dabanyun_final_score',
    #     '/mnt/cfs/workspace/speech/luohaixia/evl_data/en/test_data/10_score/xiao_e_tong/vegas_label_1216/xiao_e_tong_final_score'
    # ]

    # pred_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/PaMLLM_kimi_v2.3/model_infer_2/infer_res/infer_snt_all_10_score_test_modify'
    # pred_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/mdd_lm/saves/post_training/base_model_v5/multi_pa_v15_1/infers/api_res/snt_all_10_score_test.txt'

    label_files = ['/mnt/pfs_l2/jieti_team/SFT/hupeng/data/tal-k12/test/label_snt_score_batch1']
    pred_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/PaMLLM_kimi_v2.3/model_infer_2/infer_res/infer_batch1'

    pred_dict = read_pred_file(pred_file)

    for file in label_files:
        label_dict = read_label_file(file)
        dataset_name = file.split('/')[-1]
        true_list = []
        pred_list = []

        for key in label_dict:
            if key in pred_dict:
                true_list.append(label_dict[key])
                pred_list.append(pred_dict[key])

        pcc = pcc_compute(true_list, pred_list, dataset_name=dataset_name)