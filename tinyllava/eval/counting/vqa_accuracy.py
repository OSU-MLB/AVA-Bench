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
        self.evalQA         = {}
        self.evalQuesType   = {}
        self.evalAnsType    = {}
        self.vqa            = vqa
        self.vqaRes         = vqaRes
        self.params         = {'question_id': vqa.getQuesIds()}
        self.contractions   = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", 
                               "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", 
                               "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", 
                               "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", 
                               "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", 
                               "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", 
                               "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", 
                               "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow'sat", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", 
                               "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", 
                               "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", 
                               "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", 
                               "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", 
                               "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", 
                               "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", 
                               "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", 
                               "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", 
                               "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", 
                               "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", 
                               "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", 
                               "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", 
                               "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", 
                               "youll": "you'll", "youre": "you're", "youve": "you've"}
        self.manualMap    = { 'none': '0',
                              'zero': '0',
                              'one': '1',
                              'two': '2',
                              'three': '3',
                              'four': '4',
                              'five': '5',
                              'six': '6',
                              'seven': '7',
                              'eight': '8',
                              'nine': '9',
                              'ten': '10'
                            }
        self.articles     = ['a', 'an', 'the']
        self.periodStrip  = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip   = re.compile("(\d)(\,)(\d)")
        self.punct        = [';', r"/", '[', ']', '"', '{', '}',
                             '(', ')', '=', '+', '\\', '_', '-',
                             '>', '<', '@', '`', ',', '?', '!']
        
        self.count_keys={
            '0_5':[],
            '5_10':[],
            '10_20':[],
            '20_':[],
        }

        self.count_keys_mae={
            '0_5':[],
            '5_10':[],
            '10_20':[],
            '20_':[],
        }

        self.count_keys_mae_gt={
            '0_5':[],
            '5_10':[],
            '10_20':[],
            '20_':[],
        }

        self.count_keys_rmse={
            '0_5':[],
            '5_10':[],
            '10_20':[],
            '20_':[],
        }

        self.count_keys_rmse_gt={
            '0_5':[],
            '5_10':[],
            '10_20':[],
            '20_':[],
        }

        #Final Results
        self.accuracy        = {}
        self.accuracy_mae    = {}
        self.accuracy_mae_gt = {}
        self.accuracy_rmse   = {}
        self.accuracy_rmse_gt= {}

    def evaluate(self, quesIds=None):
        if quesIds is None:
            quesIds = [quesId for quesId in self.params['question_id']]
        gts = {}
        res = {}
        for quesId in quesIds:
            gts[quesId] = self.vqa.qa[quesId]
            res[quesId] = self.vqaRes.qa[quesId]

        error_log_path = os.path.join(args.output_result, "mismatches.csv")
        os.makedirs(args.output_result, exist_ok=True)
        csv_file = open(error_log_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Question ID", "Ground Truth", "Prediction"])

        # =================================================
        # Compute accuracy
        # =================================================
        accQA       = []
        abs_errs = []
        sq_errs  = []
        abs_errs_gt = []
        sq_errs_gt = []
        print("computing accuracy")
        step = 0
        for quesId in quesIds:
            for ansDic in gts[quesId]['answers']:
                ansDic['answer'] = ansDic['answer'].replace('\n', ' ')
                ansDic['answer'] = ansDic['answer'].replace('\t', ' ')
                ansDic['answer'] = ansDic['answer'].strip()
            resAns = res[quesId]['answer']
            resAns = resAns.replace('\n', ' ')
            resAns = resAns.replace('\t', ' ')
            resAns = resAns.strip()
            gtAcc = 0
            gtAnswers = [ans['answer'] for ans in gts[quesId]['answers']]
            if len(set(gtAnswers)) == 1:
                gtAnswers[0] = self.processPunctuation(gtAnswers[0])
                gtAnswers[0]  = self.processDigitArticle(gtAnswers[0])
                resAns = self.processPunctuation(resAns)
                resAns = self.processDigitArticle(resAns)

                if int(gtAnswers[0]) <=5:
                    cur_key='0_5'
                elif int(gtAnswers[0]) <=10:
                    cur_key='5_10'
                elif int(gtAnswers[0]) <=20:
                    cur_key='10_20'
                else:
                    cur_key='20_'

                if resAns==gtAnswers[0]:
                    gtAcc=1
                else:
                    csv_writer.writerow([quesId, gtAnswers[0], resAns])
                    
            accQA.append(gtAcc)
            self.count_keys[cur_key].append(gtAcc)
            try:
                pred = int(resAns)
                gt = int(gtAnswers[0])
                abs_errs.append(abs(pred - gt))
                sq_errs.append((pred - gt) ** 2)

                abs_errs_gt.append(abs(pred - gt)/gt)
                sq_errs_gt.append(((pred - gt) ** 2)/gt)

                self.count_keys_mae[cur_key].append(abs(pred - gt))
                self.count_keys_rmse[cur_key].append((pred - gt) ** 2)

                self.count_keys_mae_gt[cur_key].append(abs(pred- gt)/gt)
                self.count_keys_rmse_gt[cur_key].append(((pred - gt) ** 2)/gt)

            except ValueError as e:
                print(f"Invalid {gtAnswers}----{res[quesId]}")
                continue

        #import pdb;pdb.set_trace()
        self.setAccuracy(accQA)
        self.set_perCount_acc(self.count_keys)
        self.set_perCount_mae(self.count_keys_mae)
        self.set_perCount_rmse(self.count_keys_rmse)

        self.set_perCount_mae_gt(self.count_keys_mae_gt)
        self.set_perCount_rmse_gt(self.count_keys_rmse_gt)
        
        self.setMAE(abs_errs,abs_errs_gt)
        self.setRMSE(sq_errs,sq_errs_gt)
        print("\nDone computing accuracy")
        csv_file.close()
    
    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + ' ' in inText or ' ' + p in inText) or (re.search(self.commaStrip, inText) is not None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = ' '.join(outText)
        return outText

    def setAccuracy(self, accQA):
        self.accuracy['overall']  = round(100 * float(sum(accQA)) / len(accQA), self.n)

    def set_perCount_acc(self, count_keys):
        for keys in count_keys.keys():
            self.accuracy[f'{keys}']  = round(100 * float(sum(count_keys[keys])) / len(count_keys[keys]), self.n) if len(count_keys[keys]) > 0 else 0.0

    def set_perCount_mae(self, count_keys_mae):
        for keys in count_keys_mae.keys():
            self.accuracy_mae[f'{keys}']  = sum(count_keys_mae[keys]) / len(count_keys_mae[keys]) if len(count_keys_mae[keys]) > 0 else 0.0

    def set_perCount_mae_gt(self, count_keys_mae_gt):
        for keys in count_keys_mae_gt.keys():
            self.accuracy_mae_gt[f'{keys}']  = sum(count_keys_mae_gt[keys]) / len(count_keys_mae_gt[keys]) if len(count_keys_mae_gt[keys]) > 0 else 0.0   

    def set_perCount_rmse(self,count_keys_rmse):
        for keys in count_keys_rmse.keys():
            length=len(count_keys_rmse[keys])
            self.accuracy_rmse[f'{keys}']  = math.sqrt(sum(count_keys_rmse[keys]) / length) if length > 0 else 0.0

    def set_perCount_rmse_gt(self,count_keys_rmse_gt):
        for keys in count_keys_rmse_gt.keys():
            length=len(count_keys_rmse_gt[keys])
            self.accuracy_rmse_gt[f'{keys}']  = math.sqrt(sum(count_keys_rmse_gt[keys]) / length) if length > 0 else 0.0

    def setMAE(self, abs_errs,abs_errs_gt):
        self.accuracy['mae']  = sum(abs_errs) / len(abs_errs) if len(abs_errs) > 0 else 0.0
        self.accuracy['overall_mae_gt']  = sum(abs_errs_gt) / len(abs_errs_gt) if len(abs_errs_gt) > 0 else 0.0

    def setRMSE(self, sq_errs,sq_errs_gt):
        self.accuracy['rmse']  = math.sqrt(sum(sq_errs) / len(sq_errs)) if len(sq_errs) > 0 else 0.0
        self.accuracy['overall_rmse_gt']  = math.sqrt(sum(sq_errs_gt) / len(sq_errs_gt)) if len(sq_errs_gt) > 0 else 0.0

    def setEvalQA(self, quesId, acc):
        self.evalQA[quesId] = round(100 * acc, self.n)

    def setEvalQuesType(self, quesId, quesType, acc):
        if quesType not in self.evalQuesType:
            self.evalQuesType[quesType] = {}
        self.evalQuesType[quesType][quesId] = round(100 * acc, self.n)

    def setEvalAnsType(self, quesId, ansType, acc):
        if ansType not in self.evalAnsType:
            self.evalAnsType[ansType] = {}
        self.evalAnsType[ansType][quesId] = round(100 * acc, self.n)

    def updateProgress(self, progress):
        barLength = 20
        status = ""
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
            status = "error: progress var must be float\r\n"
        if progress < 0:
            progress = 0
            status = "Halt...\r\n"
        if progress >= 1:
            progress = 1
            status = "Done...\r\n"
        block = int(round(barLength * progress))
        text = "\rFinished Percent: [{}] {}% {}".format("#" * block + "-" * (barLength - block), int(progress * 100), status)
        sys.stdout.write(text)
        sys.stdout.flush()

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
        # If the item does not have an 'answers' field, create one from 'text'
        if 'answers' not in item:
            answer_text = item.get('text', "")
            item['answers'] = [{"answer": answer_text}]
        
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

    output_file = args.output_result+"/results.txt"
    os.makedirs(args.output_result,exist_ok=True)

    with open(output_file, "w") as f:
        f.write(f"Model: {args.model_name}\n")
        
        # Write overall accuracies
        f.write("Overall Accuracy Metrics:\n")
        for key, value in evaluator.accuracy.items():
            f.write(f"  {key}: {value}\n")

        f.write("\nMAE per Count Range:\n")
        for key, value in evaluator.accuracy_mae.items():
            f.write(f"  {key}: {value}\n")

        f.write("\nMAE_GT per Count Range:\n")
        for key, value in evaluator.accuracy_mae_gt.items():
            f.write(f"  {key}: {value}\n")

        f.write("\nRMSE per Count Range:\n")
        for key, value in evaluator.accuracy_rmse.items():
            f.write(f"  {key}: {value}\n")

        f.write("\nRMSE_GT per Count Range:\n")
        for key, value in evaluator.accuracy_rmse_gt.items():
            f.write(f"  {key}: {value}\n")

        f.write("\n-------------------------------------------\n\n")
