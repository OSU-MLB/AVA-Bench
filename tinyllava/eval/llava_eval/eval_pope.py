import os
import json
import argparse

def eval_pope(answers, label_file, output_f):
    label_list = [json.loads(q)['label'] for q in open(label_file, 'r') if q.strip()]

    for answer in answers:
        text = answer['text']

        # Only keep the first sentence
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            answer['text'] = 'no'
        else:
            answer['text'] = 'yes'

    for i in range(len(label_list)):
        if label_list[i] == 'no':
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer['text'] == 'no':
            pred_list.append(0)
        else:
            pred_list.append(1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    output_f.write('TP\tFP\tTN\tFN\t\n')
    output_f.write('{}\t{}\t{}\t{}\n'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP) if (TP + FP) > 0 else 0.0
    recall = float(TP) / float(TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2*precision*recall / (precision + recall) if (precision + recall) > 0 else 0.0
    acc = (TP + TN) / (TP + TN + FP + FN)

    output_f.write('Accuracy: {}\n'.format(acc))
    output_f.write('Precision: {}\n'.format(precision))
    output_f.write('Recall: {}\n'.format(recall))
    output_f.write('F1 score: {}\n'.format(f1))
    output_f.write('Yes ratio: {}\n'.format(yes_ratio))
    output_f.write('%.3f, %.3f, %.3f, %.3f, %.3f\n' % (f1, acc, precision, recall, yes_ratio))
    output_f.write('\n')
    
    # Also print to console (optional)
    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))
    print('%.3f, %.3f, %.3f, %.3f, %.3f' % (f1, acc, precision, recall, yes_ratio))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--output-file", type=str, default="pope_eval_results.txt", help="Where to save evaluation results")
    args = parser.parse_args()

    questions = [json.loads(line) for line in open(args.question_file)]
    questions = {question['question_id']: question for question in questions}
    answers = [json.loads(q) for q in open(args.result_file)]

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True) if os.path.dirname(args.output_file) else None

    with open(args.output_file, "w") as output_f:
        for file in os.listdir(args.annotation_dir):
            if not (file.startswith('coco_pope_') and file.endswith('.json')):
                continue
            category = file[10:-5]
            cur_answers = [x for x in answers if questions[x['question_id']]['category'] == category]
            output_f.write('Category: {}, # samples: {}\n'.format(category, len(cur_answers)))
            print('Category: {}, # samples: {}'.format(category, len(cur_answers)))
            eval_pope(cur_answers, os.path.join(args.annotation_dir, file), output_f)
            output_f.write("====================================\n")
            print("====================================")
