#!/usr/bin/env python
import re
import json
import argparse
import os
import csv


class VQAEval:
    def __init__(self, vqa, vqaRes, n=2):
        self.n = n
        self.accuracy = {}
        self.evalQA = {}
        self.evalQuesType = {}
        self.evalAnsType = {}
        self.vqa = vqa
        self.vqaRes = vqaRes
        self.params = {'question_id': vqa.getQuesIds()}

        self.articles = ['a', 'an', 'the']
        self.periodStrip = re.compile(r"(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile(r"(\d)(\,)(\d)")
        self.punct = [
            ';', r"/", '[', ']', '"', '{', '}',
            '(', ')', '=', '+', '\\', '_', '-',
            '>', '<', '@', '`', ',', '?', '!'
        ]

        self.class_wise_acc = {
            'Animal': 0,
            'bird': 0,
            'fungi': 0,
            'aircraft': 0,
            'plant': 0
        }

    def normalize_answer(self, s):
        if s is None:
            return ""
        s = str(s).replace('\n', ' ').replace('\t', ' ').strip().lower()
        s = self.commaStrip.sub(r"\1\3", s)
        s = self.periodStrip.sub("", s)

        for p in self.punct:
            s = s.replace(p, " ")

        s = " ".join(w for w in s.split() if w not in self.articles)
        s = " ".join(s.split())
        return s

    def extract_choice_id(self, s):
        """
        Extracts the leading option number from strings like:
        '90. bulldozing' -> '90'
        '6. crawling'    -> '6'
        Returns None if not found.
        """
        s = self.normalize_answer(s)
        m = re.match(r"^(\d+)", s)
        return m.group(1) if m else None

    def evaluate(self, output_result, quesIds=None):
        if quesIds is None:
            quesIds = [quesId for quesId in self.params['question_id']]

        gts = {}
        res = {}
        for quesId in quesIds:
            if quesId in self.vqa.qa and quesId in self.vqaRes.qa:
                gts[quesId] = self.vqa.qa[quesId]
                res[quesId] = self.vqaRes.qa[quesId]

        os.makedirs(output_result, exist_ok=True)
        error_log_path = os.path.join(output_result, "mismatches.csv")

        with open(error_log_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Question ID", "Ground Truth Answer", "Predicted Answer", "Dataset Type"])

            accQA = []
            print("computing accuracy")

            for quesId in gts:
                gt_item = gts[quesId]
                pred_item = res[quesId]

                gt_answer = self.normalize_answer(gt_item["answer"])
                pred_answer = self.normalize_answer(pred_item["answer"])

                gt_choice = self.extract_choice_id(gt_answer)
                pred_choice = self.extract_choice_id(pred_answer)

                # Prefer exact choice-number match when present.
                if gt_choice is not None and pred_choice is not None:
                    correct = (gt_choice == pred_choice)
                else:
                    # Fall back to normalized full-string match.
                    correct = (gt_answer == pred_answer)

                gtAcc = 1 if correct else 0

                if not correct:
                    print(f"prediction: {pred_answer} and gt {gt_answer} is != True")
                    csv_writer.writerow([
                        quesId,
                        gt_item["answer"],
                        pred_item["answer"],
                        gt_item.get("dataset_type", "default")
                    ])

                accQA.append(gtAcc)
                self.setAccuracy(accQA)

    def setAccuracy(self, accQA):
        if len(accQA) == 0:
            self.accuracy['overall'] = 0.0
        else:
            self.accuracy['overall'] = round(100 * float(sum(accQA)) / len(accQA), self.n)


class VQA:
    def __init__(self, data):
        """
        Supports:
        1. A list of dicts:
           [
             {
               "question_id": "...",
               "prompt": "...",
               "text": "90. bulldozing"
             }
           ]

        2. A dict with "annotations":
           {"annotations": [...]}

        3. A dict keyed by question_id
        """
        if isinstance(data, list):
            self.qa = {str(item['question_id']): self.standardize(item) for item in data}
        elif isinstance(data, dict):
            if 'annotations' in data:
                self.qa = {str(item['question_id']): self.standardize(item) for item in data['annotations']}
            else:
                self.qa = {str(k): self.standardize(v) for k, v in data.items()}
        else:
            raise ValueError("Unexpected data format for VQA")

        self.params = {'question_id': self.getQuesIds()}

    def getQuesIds(self):
        return list(self.qa.keys())

    def standardize(self, item):
        item = dict(item)

        # The JSON you showed stores answer in "text".
        if 'answer' not in item:
            item['answer'] = item.get('text', "")

        # Keep compatibility with older evaluation code.
        if 'answers' not in item:
            item['answers'] = [{'answer': item['answer']}]

        if 'question_type' not in item:
            item['question_type'] = "default"

        if 'answer_type' not in item:
            item['answer_type'] = "default"

        if 'dataset_type' not in item:
            item['dataset_type'] = "default"

        if 'prompt' not in item:
            item['prompt'] = ""

        return item


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VQA results using ground truth and answers JSON files")
    parser.add_argument("--ground_truth", type=str, required=True, help="Path to the ground truth JSON file")
    parser.add_argument("--answers", type=str, required=True, help="Path to the answers JSON file")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--output_result", type=str, required=True, help="Output result directory")
    args = parser.parse_args()

    with open(args.ground_truth, "r") as f:
        gt_data = json.load(f)

    with open(args.answers, "r") as f:
        ans_data = json.load(f)
        print(f"loaded {args.answers}")

    gt_vqa = VQA(gt_data)
    ans_vqa = VQA(ans_data)

    evaluator = VQAEval(gt_vqa, ans_vqa, n=2)
    evaluator.evaluate(output_result=args.output_result)

    print("\nOverall Accuracy: {}".format(evaluator.accuracy['overall']))

    os.makedirs(args.output_result, exist_ok=True)
    output_file = os.path.join(args.output_result, "results.txt")

    with open(output_file, "w") as f:
        f.write(f"Model: {args.model_name}\n")
        f.write("Overall Accuracy Metrics:\n")
        f.write(f"acc: {evaluator.accuracy['overall']}.\n")
        f.write("\nACC per Dataset:\n")
        for key, value in evaluator.class_wise_acc.items():
            f.write(f"  {key}: {value*100:.2f}\n")