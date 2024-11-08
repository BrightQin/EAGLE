import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from eagle.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from eagle.conversation import conv_templates, SeparatorStyle
from eagle.model.builder import load_pretrained_model, load_video_pretrain_model
from eagle.utils import disable_torch_init
from eagle.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math

import sys

TRACE_PREX = ''
CURRENT_FILE = ''

def trace_calls(frame, event, arg):

    global TRACE_PREX
    global CURRENT_FILE
    if CURRENT_FILE is None:
        CURRENT_FILE = ''
    if TRACE_PREX is None:
        TRACE_PREX = ''
    if 'anaconda' in frame.f_code.co_filename or \
            '<' in frame.f_code.co_name or \
            '<' in frame.f_code.co_filename:
        return
    if event == 'call':
        TRACE_PREX += '\t'
        current_file = frame.f_code.co_filename.replace('/home4/hxl/MLLM/EAGLE/', '')
        if current_file != CURRENT_FILE:
            CURRENT_FILE = current_file
            print(TRACE_PREX + CURRENT_FILE)
        print(f'{TRACE_PREX + frame.f_code.co_name} at line: {frame.f_lineno}')
    elif event == 'return':
        TRACE_PREX = TRACE_PREX[1:]
    return trace_calls

sys.settrace(trace_calls)

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
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_video_pretrain_model(
        model_path, 
        args.model_base, 
        model_name,
        device_map='cuda'
    )
    model.set_modal('video')

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for i, line in enumerate(tqdm(questions)):
        idx = line["question_id"]
        question = line['question']
        cur_prompt = question.replace('<image>', '').strip()
        qs = cur_prompt

        if 'video_name' in line:
            video_name = 'v_' + line["video_name"]
            if os.path.exists(
                os.path.join(
                    args.image_folder,
                    video_name + '.mp4'
                )
            ):
                video_path = os.path.join(
                    args.image_folder,
                    video_name + '.mp4'
                )
            elif os.path.exists(
                os.path.join(
                    args.image_folder,
                    video_name + '.mkv'
                )
            ):
                video_path = os.path.join(
                    args.image_folder,
                    video_name + '.mkv'
                )
            else:
                print(f'Unfound video {video_name}')
                continue

            video = image_processor(video_path, return_tensors='pt')['pixel_values']
            video = video.half().cuda()
            print(video.shape)
            image_sizes = [video.size]

            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            video = None
            

        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = model.forward(
                input_ids,
                images=video,
                modality='video',
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=1024,
                use_cache=True,
            )
        print(isinstance(output_ids[0], list))

        outputs = [list(past_key_value) for past_key_value in output_ids.past_key_values]

        outputs = tokenizer.batch_decode(
            # output_ids, 
            outputs,
            skip_special_tokens=True
        )[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home4/hxl/MLLM/EAGLE/checkpoints/finetune-eagle-x1-llama3.2-1b/checkpoint-2")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/home4/hxl/MLLM/EAGLE/dataset/Image_Eval/scienceqa/images")
    parser.add_argument("--question-file", type=str, default="/home4/hxl/MLLM/EAGLE/dataset/Image_Eval/scienceqa/llava_test_CQM-A.json")
    parser.add_argument("--answers-file", type=str, default="/home4/hxl/MLLM/EAGLE/dataset/Image_Eval/scienceqa/answers/EAGLE_SQA.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llama3")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    args = parser.parse_args()

    eval_model(args)
