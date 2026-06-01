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
from pathlib import Path

#to_add_str= "\nAnswer the question using only the correct option."
to_add_str=""

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
        qs = line["text"]+to_add_str

        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = self.image_processor(image)
        
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        msg = Message()
        msg.add_message(qs)
        #print(prompt)
        result = self.text_processor(msg.messages, mode='eval')
        input_ids = result['input_ids']

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


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
    print(f"Loading model from {model_path}")
    model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)
    #import pdb;pdb.set_trace()
    
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

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"] + to_add_str

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
        ans_id = shortuuid.uuid()
        answers.append({"question_id": idx,
                        "prompt": cur_prompt,
                        "text": outputs,
                        "answer_id": ans_id,
                        "model_id": args.model_base,
                        "metadata": {}})
    with open(answers_file, "w") as f:
        json.dump(answers, f, indent=4)



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