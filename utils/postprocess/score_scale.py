import ast



def fluency_scale(score:float, probability:float) -> float:
    score = (1 + score) * 25 * probability
    return round(score, 2)

def score_scale(score:float, probability:float) -> float:
    if score == 0:
        score = 0.0 + (1 - probability) * 5
    elif score == 10:
        score =  95 + probability * 5
    else:
        score = (score * 10 - 5) + probability * 10

    return round(score, 2)


if __name__ == "__main__":
    input_file = '/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/ts_model/output_multiclass_acc/infer_res/batch1.res2'
    output_file = f'{input_file}.scaled'


    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        lines = fin.readlines()
        for line in lines:
            line = line.strip().split('\t')
            key, prob_str, pred_class = line[0], line[1], line[2]
            probabilities = ast.literal_eval(prob_str)
            pred_index = int(pred_class)
            pred_prob = probabilities[pred_index]
            scaled_score = score_scale(float(pred_class), pred_prob)
            # scaled_score = fluency_scale(float(pred_class), pred_prob)
            fout.write(f'{key}\t{scaled_score}\n')