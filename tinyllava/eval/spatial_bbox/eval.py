#!/usr/bin/env python
import re
import sys
import json
import argparse
import math
import os
import csv


class VQAEval:
    def __init__(self, vqa, vqaRes, n=2):
        self.n              = n
        self.accuracy       = {}
        self.evalQA         = {}
        self.evalQuesType   = {}
        self.evalAnsType    = {}
        self.vqa            = vqa
        self.vqaRes         = vqaRes
        self.params         = {'question_id': vqa.getQuesIds()}
        self.articles     = ['a', 'an', 'the']
        self.periodStrip  = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip   = re.compile("(\d)(\,)(\d)")
        self.punct        = [';', r"/", '[', ']', '"', '{', '}',
                             '(', ')', '=', '+', '\\', '_', '-',
                             '>', '<', '@', '`', ',', '?', '!']
        
        self.class_wise_acc = {
            'Animal': 0,
            'bird': 0,
            'fungi': 0,
            'aircraft': 0,
            'plant': 0
        }
        
    def evaluate(self, quesIds=None):

        class_wise_eval = {
            'Animal': [],
            'bird': [],
            'fungi': [],
            'aircraft': [],
            'plant': []
        }

        if quesIds is None:
            quesIds = [quesId for quesId in self.params['question_id']]
        gts = {}
        res = {}
        for quesId in quesIds:
            gts[quesId] = self.vqa.qa[quesId]
            res[quesId] = self.vqaRes.qa[quesId]

        # CSV setup
        error_log_path = os.path.join(args.output_result, "mismatches.csv")
        os.makedirs(args.output_result, exist_ok=True)
        csv_file = open(error_log_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Question ID", "Ground Truth Answer", "Predicted Answer", "Dataset Type"])


        # =================================================
        # Compute accuracy
        # =================================================
        import pdb;pdb.set_trace()
        accQA       = []
        print("computing accuracy")
        for quesId in quesIds:
            gts[quesId]['answer'] = gts[quesId]['text'].replace('\n', ' ')
            gts[quesId]['answer'] = gts[quesId]['text'].replace('\t', ' ')
            gts[quesId]['answer'] = gts[quesId]['text'].strip()

            resAns = res[quesId]['text']
            resAns = resAns.replace('\n', ' ')
            resAns = resAns.replace('\t', ' ')
            resAns = resAns.strip()
            gtAcc = 0

            gt_tokens = re.split(r'[. ]+', gts[quesId]['answer'])
            res_tokens = re.split(r'[. ]+', resAns)

            
            correct=False
            gt_lower = [t.lower() for t in gt_tokens]
            res_lower = [t.lower() for t in res_tokens]

            if len(gt_lower) > 0 and len(res_lower) > 0 and gt_lower[0] == res_lower[0]:
                gtAcc+=1
                correct=True
            else:
                gt_rest = gt_lower[1:]
                n = len(gt_rest)
                if n > 0:
                    correct=True
                    for i in range(len(res_lower) - n + 1):
                        if res_lower[i:i+n] != gt_rest:
                            correct=False
                            break
                    if correct==True:
                        gtAcc+=1
            if not correct:
                print(f"prediction: {resAns} and gt {gts[quesId]['answer']} is != True")
                csv_writer.writerow([quesId, gts[quesId]['answer'], resAns, 'N/A'])
                   

            accQA.append(gtAcc)
            self.setAccuracy(accQA)

            
        csv_file.close()

    def setAccuracy(self, accQA):
        self.accuracy['overall']  = round(100 * float(sum(accQA)) / len(accQA), self.n)




# Helper class to wrap the JSON data into an object that VQAEval expects.
class VQA:
    def __init__(self, data):
        # Data may be provided either as a list of annotations or as a dict (e.g., with an "annotations" field)
        if isinstance(data, list):
            self.qa = {item['question_id']: self.standardize(item) for item in data}
        elif isinstance(data, dict):
            if 'annotations' in data:
                self.qa = {item['question_id']: self.standardize(item) for item in data['annotations']}
            else:
                # If the data is a dict with question IDs as keys
                self.qa = {k: self.standardize(v) for k, v in data.items()}
        else:
            raise ValueError("Unexpected data format for VQA")
        self.params = {}
        self.params['question_id'] = self.getQuesIds()
    
    def getQuesIds(self):
        return list(self.qa.keys())

    def standardize(self, item):

        
        # For predicted answers, if 'answer' is missing, create it from 'text'
        if 'answer' not in item:
            item['answer'] = item.get('text', "")
        
        # Provide default question and answer types if not present.
        if 'question_type' not in item:
            item['question_type'] = "default"
        if 'answer_type' not in item:
            item['answer_type'] = "default"
        
        return item
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VQA results using ground truth and answers JSON files")
    parser.add_argument("--ground_truth", type=str, required=True, help="Path to the ground truth JSON file")
    parser.add_argument("--answers", type=str, required=True, help="Path to the answers JSON file")
    parser.add_argument("--model_name", type=str, required=True, help="model_name")
    parser.add_argument("--output_result", type=str, required=True, help="output_result_file_name")
    args = parser.parse_args()

    # Load the JSON files.
    with open(args.ground_truth, "r") as f:
        gt_data = json.load(f)
    
    with open(args.answers, "r") as f:
        ans_data = json.load(f)
        print(f"loaded_ {args.answers}")

    # Wrap the loaded JSON in helper objects.
    gt_vqa = VQA(gt_data)
    ans_vqa = VQA(ans_data)

    # Evaluate.
    evaluator = VQAEval(gt_vqa, ans_vqa, n=2)
    evaluator.evaluate()

    # Print overall, per-question-type, and per-answer-type accuracies.
    print("\nOverall Accuracy: {}".format(evaluator.accuracy['overall']))

    # Print overall, per-question-type, and per-answer-type accuracies.
    print("\nOverall Accuracy: {}".format(evaluator.accuracy['overall']))

    output_file = args.output_result+"/results.txt"
    os.makedirs(args.output_result,exist_ok=True)

    with open(output_file, "w") as f:
        f.write(f"Model: {args.model_name}\n")
        
        # Write overall accuracies
        f.write("Overall Accuracy Metrics:\n")

        sentence= f"acc: {evaluator.accuracy['overall']}."
        f.write(sentence)

        f.write("\nACC per Dataset:\n")
        for key, value in evaluator.class_wise_acc.items():
            f.write(f"  {key}: {value*100:.2f}\n")