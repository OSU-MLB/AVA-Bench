import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid


from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *

import pandas as pd
from PIL import Image
import math

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    
    model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)
    model.to(device='cuda')    
    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    data_args = model.config
    image_processor = ImagePreprocess(image_processor, data_args)


    # Load and read the CSV
    benchmark_dir = os.path.join(args.directory, 'Questions.csv')
    df = pd.read_csv(benchmark_dir)  # Assuming the fields are separated by tabs
    answers_file = os.path.expanduser(args.answers_file)
    # Check if the directory is specified in the path
    if os.path.dirname(answers_file):
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    # Now open the file
    ans_file = open(answers_file, "w")

    # Loop through each row in the DataFrame
    for index, row in tqdm(df.iterrows()):
        cur_prompt = row['Question'] + " " + row['Options']
        qs = cur_prompt

        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs 
        qs+=args.question_extension

        msg = Message()
        msg.add_message(qs)

        result = text_processor(msg.messages, mode='eval')
        input_ids = result['input_ids']
        prompt = result['prompt']
        input_ids = input_ids.unsqueeze(0).cuda()

        # Load the corresponding image
        photo_id = index+1
        image_path = os.path.join(args.directory, 'MMVP Images', f"{photo_id}.jpg")
        image = Image.open(image_path).convert('RGB')
        image_tensor = image_processor(image)
        image_tensors = image_tensor.unsqueeze(0).half().cuda()
        image_sizes = [image.size]
        print(image_sizes)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": photo_id,
                                   "prompt": cur_prompt,
                                   "answer": row["Correct Answer"], 
                                   "response": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_path,
                                   }) + "\n")
        ans_file.flush()
    ans_file.close()









if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--directory", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--question_extension", type=str, default="Answer with the option's letter from the given choices directly.")
    args = parser.parse_args()

    eval_model(args)