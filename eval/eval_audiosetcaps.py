import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

# for debug
import sys
sys.path.append(os.getcwd())

from eagle.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from eagle.conversation import conv_templates, SeparatorStyle
from eagle.model.builder import load_pretrained_model, load_audio_pretrained_model
from eagle.utils import disable_torch_init
from eagle.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from eagle.datasets.old.audio_dataset import AudioEvalDataset, audio_eval_collate
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

# DataLoader
def create_data_loader(audio_text_path, audio_path, context_length, audio_processor, conv_mode='plain', batch_size=1, num_workers=4):
    # assert batch_size == 1, "batch_size must be 1"
    dataset = AudioEvalDataset(
        conv_mode=conv_mode,
        audio_text_path=audio_text_path,
        audio_path=audio_path,
        audio_processor=audio_processor,
        context_length=context_length
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=audio_eval_collate)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, audio_processor, context_len = load_audio_pretrained_model(
        model_path, 
        # args.model_base, 
        # model_name
    )

    output_file = os.path.expanduser(args.output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    out_file = open(output_file, "w")

    data_loader = create_data_loader(
        audio_text_path=args.audio_text_path,
        audio_path=args.audio_path,
        context_length=context_len,
        audio_processor=audio_processor,
        conv_mode=args.conv_mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    correct = 0
    strict = 0
    length = len(data_loader)

    for input_ids, audio, question, answer in tqdm(data_loader, total=length):

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            # TODO
            output_ids = model.generate(
                input_ids,
                images=audio.to(dtype=torch.float16, device='cuda', non_blocking=True),
                # image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True
            )

        ouptus_all = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        print(ouptus_all)
        outputs = ouptus_all[0].strip() # For batch size == 0 temporally

        ans_id = shortuuid.uuid()
        out_file.write(
            json.dumps(
                {   
                    "question": question,
                    "predict": outputs,
                    "answer_gt": answer,
                    "answer_id": ans_id,
                    "model_id": model_name,
                    "metadata": {}
                }
            ) + "\n")
        if (outputs.lower() in answer.lower()) or (answer.lower() in outputs.lower()):
            correct += 1
            if outputs.lower() == answer.lower():
                strict += 1
        # out_file.flush()

    print(f"correct: {correct / length}")
    print(f"strict: {strict / length}")
    out_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--audio-path", type=str, default="")
    parser.add_argument("--audio-text-path", type=str, default="tables/question.jsonl")
    parser.add_argument("--output-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
