#!/usr/bin/env python
import re
import json
import argparse
import os
import csv
import math


class VQAEval:
    def __init__(self, vqa, vqaRes, n=4):
        self.n = n
        self.metrics = {}
        self.evalQA = {}
        self.vqa = vqa
        self.vqaRes = vqaRes
        self.params = {'question_id': vqa.getQuesIds()}

        # Optional per-dataset tracking
        self.dataset_scores = {}

    def parse_rgb(self, s):
        """
        Parses strings like:
        "[232, 9, 10]"
        "232, 9, 10"
        "(232, 9, 10)"
        "232 9 10"
        Returns (r, g, b) as floats in [0,255].
        """
        if s is None:
            raise ValueError("RGB string is None")

        if isinstance(s, (list, tuple)) and len(s) == 3:
            vals = [float(x) for x in s]
        else:
            s = str(s).strip()
            nums = re.findall(r"-?\d+(?:\.\d+)?", s)
            if len(nums) < 3:
                raise ValueError(f"Could not parse RGB from: {s}")
            vals = [float(nums[0]), float(nums[1]), float(nums[2])]

        vals = [max(0.0, min(255.0, x)) for x in vals]
        return tuple(vals)

    def srgb_to_linear(self, c):
        c = c / 255.0
        if c <= 0.04045:
            return c / 12.92
        return ((c + 0.055) / 1.055) ** 2.4

    def rgb_to_xyz(self, rgb):
        r, g, b = rgb
        r = self.srgb_to_linear(r)
        g = self.srgb_to_linear(g)
        b = self.srgb_to_linear(b)

        # sRGB D65
        x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
        y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
        z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

        return (x, y, z)

    def xyz_to_lab(self, xyz):
        x, y, z = xyz

        # D65 reference white
        Xn = 0.95047
        Yn = 1.00000
        Zn = 1.08883

        def f(t):
            delta = 6 / 29
            if t > delta ** 3:
                return t ** (1 / 3)
            return t / (3 * delta ** 2) + 4 / 29

        fx = f(x / Xn)
        fy = f(y / Yn)
        fz = f(z / Zn)

        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)

        return (L, a, b)

    def rgb_to_lab(self, rgb):
        return self.xyz_to_lab(self.rgb_to_xyz(rgb))

    def ciede2000(self, lab1, lab2):
        """
        CIEDE2000 color-difference formula.
        Returns Delta E 2000.
        """
        L1, a1, b1 = lab1
        L2, a2, b2 = lab2

        avg_L = (L1 + L2) / 2.0
        C1 = math.sqrt(a1 ** 2 + b1 ** 2)
        C2 = math.sqrt(a2 ** 2 + b2 ** 2)
        avg_C = (C1 + C2) / 2.0

        G = 0.5 * (1 - math.sqrt((avg_C ** 7) / (avg_C ** 7 + 25 ** 7))) if avg_C != 0 else 0

        a1_prime = (1 + G) * a1
        a2_prime = (1 + G) * a2

        C1_prime = math.sqrt(a1_prime ** 2 + b1 ** 2)
        C2_prime = math.sqrt(a2_prime ** 2 + b2 ** 2)
        avg_C_prime = (C1_prime + C2_prime) / 2.0

        def hp_func(x, y):
            if x == 0 and y == 0:
                return 0.0
            h = math.degrees(math.atan2(y, x))
            return h + 360 if h < 0 else h

        h1_prime = hp_func(a1_prime, b1)
        h2_prime = hp_func(a2_prime, b2)

        delta_L_prime = L2 - L1
        delta_C_prime = C2_prime - C1_prime

        if C1_prime * C2_prime == 0:
            delta_h_prime = 0.0
        else:
            dh = h2_prime - h1_prime
            if dh > 180:
                dh -= 360
            elif dh < -180:
                dh += 360
            delta_h_prime = dh

        delta_H_prime = 2 * math.sqrt(C1_prime * C2_prime) * math.sin(math.radians(delta_h_prime / 2.0))

        avg_L_prime = (L1 + L2) / 2.0

        if C1_prime * C2_prime == 0:
            avg_h_prime = h1_prime + h2_prime
        else:
            h_sum = h1_prime + h2_prime
            h_diff = abs(h1_prime - h2_prime)

            if h_diff > 180:
                if h_sum < 360:
                    avg_h_prime = (h_sum + 360) / 2.0
                else:
                    avg_h_prime = (h_sum - 360) / 2.0
            else:
                avg_h_prime = h_sum / 2.0

        T = (
            1
            - 0.17 * math.cos(math.radians(avg_h_prime - 30))
            + 0.24 * math.cos(math.radians(2 * avg_h_prime))
            + 0.32 * math.cos(math.radians(3 * avg_h_prime + 6))
            - 0.20 * math.cos(math.radians(4 * avg_h_prime - 63))
        )

        delta_theta = 30 * math.exp(-(((avg_h_prime - 275) / 25) ** 2))
        R_C = 2 * math.sqrt((avg_C_prime ** 7) / (avg_C_prime ** 7 + 25 ** 7)) if avg_C_prime != 0 else 0
        S_L = 1 + (0.015 * ((avg_L_prime - 50) ** 2)) / math.sqrt(20 + ((avg_L_prime - 50) ** 2))
        S_C = 1 + 0.045 * avg_C_prime
        S_H = 1 + 0.015 * avg_C_prime * T
        R_T = -math.sin(math.radians(2 * delta_theta)) * R_C

        k_L = 1
        k_C = 1
        k_H = 1

        delta_E = math.sqrt(
            (delta_L_prime / (k_L * S_L)) ** 2
            + (delta_C_prime / (k_C * S_C)) ** 2
            + (delta_H_prime / (k_H * S_H)) ** 2
            + R_T * (delta_C_prime / (k_C * S_C)) * (delta_H_prime / (k_H * S_H))
        )

        return delta_E

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
        error_log_path = os.path.join(output_result, "ciede2000_scores.csv")

        all_delta_e = []

        with open(error_log_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "Question ID",
                "Ground Truth Answer",
                "Predicted Answer",
                "Ground Truth RGB",
                "Predicted RGB",
                "CIEDE2000",
                "Dataset Type"
            ])

            print("computing CIEDE2000")

            for quesId in gts:
                gt_item = gts[quesId]
                pred_item = res[quesId]

                gt_raw = gt_item["answer"]
                pred_raw = pred_item["answer"]

                dataset_type = gt_item.get("dataset_type", "default")

                try:
                    gt_rgb = self.parse_rgb(gt_raw)
                    pred_rgb = self.parse_rgb(pred_raw)

                    gt_lab = self.rgb_to_lab(gt_rgb)
                    pred_lab = self.rgb_to_lab(pred_rgb)

                    delta_e = self.ciede2000(gt_lab, pred_lab)
                except Exception as e:
                    print(f"Skipping {quesId}: {e}")
                    delta_e = None
                    gt_rgb = None
                    pred_rgb = None

                if delta_e is not None:
                    all_delta_e.append(delta_e)
                    self.dataset_scores.setdefault(dataset_type, []).append(delta_e)

                csv_writer.writerow([
                    quesId,
                    gt_raw,
                    pred_raw,
                    gt_rgb,
                    pred_rgb,
                    "" if delta_e is None else round(delta_e, self.n),
                    dataset_type
                ])

        self.setMetrics(all_delta_e)

    def setMetrics(self, all_delta_e):
        if len(all_delta_e) == 0:
            self.metrics["mean_ciede2000"] = None
            self.metrics["median_ciede2000"] = None
            self.metrics["num_samples"] = 0
            return

        sorted_vals = sorted(all_delta_e)
        n = len(sorted_vals)

        if n % 2 == 1:
            median_val = sorted_vals[n // 2]
        else:
            median_val = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0

        self.metrics["mean_ciede2000"] = round(sum(all_delta_e) / len(all_delta_e), self.n)
        self.metrics["median_ciede2000"] = round(median_val, self.n)
        self.metrics["num_samples"] = len(all_delta_e)


class VQA:
    def __init__(self, data):
        """
        Supports:
        1. A list of dicts
        2. A dict with "annotations"
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

        # Your JSON stores the answer in "text"
        if 'answer' not in item:
            item['answer'] = item.get('text', "")

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
    parser = argparse.ArgumentParser(description="Evaluate VQA results using CIEDE2000 on RGB answers")
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

    evaluator = VQAEval(gt_vqa, ans_vqa, n=4)
    evaluator.evaluate(output_result=args.output_result)

    print("\nMean CIEDE2000: {}".format(evaluator.metrics["mean_ciede2000"]))
    print("Median CIEDE2000: {}".format(evaluator.metrics["median_ciede2000"]))
    print("Num Samples: {}".format(evaluator.metrics["num_samples"]))

    os.makedirs(args.output_result, exist_ok=True)
    output_file = os.path.join(args.output_result, "results.txt")

    with open(output_file, "w") as f:
        f.write(f"Model: {args.model_name}\n")
        f.write("Color Difference Metrics:\n")
        f.write(f"mean_ciede2000: {evaluator.metrics['mean_ciede2000']}\n")
        f.write(f"median_ciede2000: {evaluator.metrics['median_ciede2000']}\n")
        f.write(f"num_samples: {evaluator.metrics['num_samples']}\n")
        f.write("\nMean CIEDE2000 per Dataset:\n")

        for key, values in evaluator.dataset_scores.items():
            if len(values) > 0:
                mean_val = sum(values) / len(values)
                f.write(f"  {key}: {mean_val:.4f}\n")
            else:
                f.write(f"  {key}: N/A\n")