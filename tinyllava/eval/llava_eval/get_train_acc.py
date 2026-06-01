import argparse
import time

import torch
import os
import json
from tqdm import tqdm
import shortuuid

from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *

from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import re

to_add_str= "\nAnswer the question using a single word or phrase."

contractions   = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", 
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
manualMap    = { 'none': '0',
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
articles     = ['a', 'an', 'the']
periodStrip  = re.compile("(?!<=\d)(\.)(?!\d)")
commaStrip   = re.compile("(\d)(\,)(\d)")
punct        = [';', r"/", '[', ']', '"', '{', '}',
                '(', ')', '=', '+', '\\', '_', '-',
                '>', '<', '@', '`', ',', '?', '!']

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, text_processor, image_processor):
        self.questions = questions
        self.image_folder = image_folder
        self.text_processor = text_processor
        self.image_processor = image_processor

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["conversations"][0]['value']#+to_add_str

        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = self.image_processor(image)
        
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        msg = Message()
        msg.add_message(qs)
        #print(prompt)
        result = self.text_processor(msg.messages, mode='eval')
        input_ids = result['input_ids']
        answer = line["conversations"][1]['value']

        return input_ids, image_tensor, image.size,answer

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes, answers = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes,answers

# DataLoader
def create_data_loader(questions, image_folder, text_processor, image_processor, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, text_processor, image_processor)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader

def eval_model(args):
    #import pdb;pdb.set_trace()
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)
    
    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    data_args = model.config
    image_processor = ImagePreprocess(image_processor, data_args)

    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = json.load(f)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    data_loader = create_data_loader(questions, args.image_folder, text_processor, image_processor)
    # print("Tokenizer's eos token: ", tokenizer.eos_token)
    model.to(device='cuda')
    answers = []
    acc=0
    total_sample=0

    for (input_ids, image_tensor, image_sizes,groundTruth), line in tqdm(zip(data_loader, questions), total=len(questions)):
        total_sample+=1

        idx = line["id"]
        cur_prompt = line["conversations"][0]['value'] #+ to_add_str

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                # stopping_criteria=[stopping_criteria],
                image_sizes=image_sizes,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        answers.append({"question_id": idx,
                        "question": cur_prompt,
                        "prediction": outputs,
                        "answer": groundTruth[0]})

        cur_ac, pred=get_acc(outputs,groundTruth[0])

        acc+=cur_ac
        
        if re.search(r"[A-Za-z]", pred):
            answer_record["contains_letter"] = "as well"

    answers.append({"Accuracy":acc/total_sample})
    with open(answers_file, "w") as f:
        json.dump(answers, f, indent=4)
    print(f"Train Acc: {acc/total_sample}")

def get_acc(pred,gt):
    pred=pred.replace('\n', ' ')
    pred=pred.replace('\t', ' ')
    pred=pred.strip()
    pred=processPunctuation(pred)
    pred=processDigitArticle(pred)


    gt=gt.replace('\n', ' ')
    gt=gt.replace('\t', ' ')
    gt=gt.strip()
    gt=processPunctuation(gt)
    gt=processDigitArticle(gt)

    if gt == pred:
        return 1,pred
    else:
        return 0,pred


def processPunctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(commaStrip, inText) is not None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub("", outText, re.UNICODE)
    return outText

def processDigitArticle(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llama")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")
    args = parser.parse_args()

    eval_model(args)