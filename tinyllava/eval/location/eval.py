from pathlib import Path
import re
import json
import argparse
import os
import csv


class VQAEval:
    def __init__(self, vqa, vqaRes, n=4):
        self.n = n
        self.metrics = {}
        self.vqa = vqa
        self.vqaRes = vqaRes
        self.params = {'question_id': vqa.getQuesIds()}
        self.dataset_scores = {}

    def parse_bbox(self, s):
        """
        Parse bbox strings like:
        "[0.56, 0.42, 0.67, 0.58]"
        Returns (x1, y1, x2, y2)
        """
        if s is None:
            raise ValueError("Bounding box string is None")

        if isinstance(s, (list, tuple)) and len(s) >= 4:
            vals = [float(x) for x in s[:4]]
        else:
            nums = re.findall(r"-?\d+(?:\.\d+)?", str(s))
            if len(nums) < 4:
                raise ValueError(f"Could not parse bbox from: {s}")
            vals = [float(nums[i]) for i in range(4)]

        x1, y1, x2, y2 = vals
        return x1, y1, x2, y2

    def giou(self, box1, box2):
        """
        Compute Generalized IoU for boxes in [x1, y1, x2, y2] format.
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Ensure valid box ordering
        if x1_1 > x2_1:
            x1_1, x2_1 = x2_1, x1_1
        if y1_1 > y2_1:
            y1_1, y2_1 = y2_1, y1_1
        if x1_2 > x2_2:
            x1_2, x2_2 = x2_2, x1_2
        if y1_2 > y2_2:
            y1_2, y2_2 = y2_2, y1_2

        # Intersection
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        # Areas
        area1 = max(0.0, x2_1 - x1_1) * max(0.0, y2_1 - y1_1)
        area2 = max(0.0, x2_2 - x1_2) * max(0.0, y2_2 - y1_2)

        union = area1 + area2 - inter_area
        iou = inter_area / union if union > 0 else 0.0

        # Smallest enclosing box
        c_x1 = min(x1_1, x1_2)
        c_y1 = min(y1_1, y1_2)
        c_x2 = max(x2_1, x2_2)
        c_y2 = max(y2_1, y2_2)

        c_area = max(0.0, c_x2 - c_x1) * max(0.0, c_y2 - c_y1)

        if c_area <= 0:
            return iou

        giou = iou - ((c_area - union) / c_area)
        return giou

    def evaluate(self, output_result, quesIds=None):
        if quesIds is None:
            quesIds = [qid for qid in self.params['question_id']]

        gts = {}
        res = {}
        for qid in quesIds:
            if qid in self.vqa.qa and qid in self.vqaRes.qa:
                gts[qid] = self.vqa.qa[qid]
                res[qid] = self.vqaRes.qa[qid]

        os.makedirs(output_result, exist_ok=True)
        csv_path = os.path.join(output_result, "giou_scores.csv")

        all_giou = []

        with open(csv_path, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                "Question ID",
                "Ground Truth Answer",
                "Predicted Answer",
                "Ground Truth BBox",
                "Predicted BBox",
                "GIoU",
                "Dataset Type"
            ])

            print("computing GIoU")

            for qid in gts:
                gt_item = gts[qid]
                pred_item = res[qid]

                gt_raw = gt_item["answer"]
                pred_raw = pred_item["answer"]
                dataset_type = gt_item.get("dataset_type", "default")

                try:
                    gt_box = self.parse_bbox(gt_raw)
                    pred_box = self.parse_bbox(pred_raw)
                    giou_score = self.giou(gt_box, pred_box)
                except Exception as e:
                    print(f"Skipping {qid}: {e}")
                    gt_box = None
                    pred_box = None
                    giou_score = None

                if giou_score is not None:
                    all_giou.append(giou_score)
                    self.dataset_scores.setdefault(dataset_type, []).append(giou_score)

                writer.writerow([
                    qid,
                    gt_raw,
                    pred_raw,
                    gt_box,
                    pred_box,
                    "" if giou_score is None else round(giou_score, self.n),
                    dataset_type
                ])

        self.setMetrics(all_giou)

    def setMetrics(self, all_giou):
        if len(all_giou) == 0:
            self.metrics["mean_giou"] = None
            self.metrics["final_result"] = None
            self.metrics["num_samples"] = 0
            return

        mean_giou = sum(all_giou) / len(all_giou)
        self.metrics["mean_giou"] = round(mean_giou, self.n)
        self.metrics["final_result"] = round(mean_giou, self.n)
        self.metrics["num_samples"] = len(all_giou)


class VQA:
    def __init__(self, data):
        """
        Supports:
        1. list of dicts
        2. dict with "annotations"
        3. dict keyed by question_id
        """
        if isinstance(data, list):
            self.qa = {str(item["question_id"]): self.standardize(item) for item in data}
        elif isinstance(data, dict):
            if "annotations" in data:
                self.qa = {str(item["question_id"]): self.standardize(item) for item in data["annotations"]}
            else:
                self.qa = {str(k): self.standardize(v) for k, v in data.items()}
        else:
            raise ValueError("Unexpected data format for VQA")

        self.params = {'question_id': self.getQuesIds()}

    def getQuesIds(self):
        return list(self.qa.keys())

    def standardize(self, item):
        item = dict(item)

        if "answer" not in item:
            item["answer"] = item.get("text", "")

        if "answers" not in item:
            item["answers"] = [{"answer": item["answer"]}]

        if "question_type" not in item:
            item["question_type"] = "default"

        if "answer_type" not in item:
            item["answer_type"] = "default"

        if "dataset_type" not in item:
            item["dataset_type"] = "default"

        if "prompt" not in item:
            item["prompt"] = ""

        return item


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predicted bounding boxes using GIoU")
    parser.add_argument("--ground_truth", type=str, required=True, help="Path to the ground truth JSON file")
    parser.add_argument("--answers", type=str, required=True, help="Path to the predicted answers JSON file")
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

    evaluator = VQAEval(gt_vqa, ans_vqa, n=4)
    evaluator.evaluate(output_result=args.output_result)

    print(f"\nMean GIoU: {evaluator.metrics['mean_giou']}")
    print(f"Final Result: {evaluator.metrics['final_result']}")
    print(f"Num Samples: {evaluator.metrics['num_samples']}")

    os.makedirs(args.output_result, exist_ok=True)
    output_file = os.path.join(args.output_result, "results.txt")

    with open(output_file, "w") as f:
        f.write(f"Model: {args.model_name}\n")
        f.write("Bounding Box Metrics:\n")
        f.write(f"mean_giou: {evaluator.metrics['mean_giou']}\n")
        f.write(f"final_result: {evaluator.metrics['final_result']}\n")
        f.write(f"num_samples: {evaluator.metrics['num_samples']}\n")
        f.write("\nMean GIoU per Dataset:\n")
        for key, values in evaluator.dataset_scores.items():
            if len(values) > 0:
                mean_val = sum(values) / len(values)
                f.write(f"  {key}: {mean_val:.4f}\n")
            else:
                f.write(f"  {key}: N/A\n")