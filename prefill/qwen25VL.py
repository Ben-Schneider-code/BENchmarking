"""
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00,  7.11it/s]
qwen-vl-utils using decord to read video.
Total time elapsed for video decompression: 18.429545640945435
Processor overhead: 2.294198513031006
Video porcessing overhead: 16.13534712791443
Video tensor shape: torch.Size([1, 69138])
Time elapsed for prefill with FA2: 6.546677589416504
This 61 minute video used for this test is:
https://www.youtube.com/watch?v=AO2S2QZjykE
peak memory used is ~60 Gb on A100
"""

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import time

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)
video_path = "/scratch/b3schnei/data/video/60min.mp4"
model.cuda()

def load_video():
    start = time.time()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{video_path}",
                    "max_pixels": 360 * 420,
                    "fps": 1.0,
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    image_inputs, video_inputs = process_vision_info(messages)
    proc_start = time.time()
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    end = time.time()
    elapsed = end - start
    proc_elapsed = end-proc_start
    print(f"Total time elapsed for video decompression: {elapsed}")
    print(f"Processor overhead: {proc_elapsed}")
    print(f"Video porcessing overhead: {elapsed-proc_elapsed}")
    return inputs

# Can run multiple time to get average time.
# This timing is more inconsistent
for i in range(1):
    inputs = load_video()

inputs = inputs.to("cuda")
print(f"Video tensor shape: {inputs.data["input_ids"].shape}")

start = time.time()
with torch.no_grad():
    outputs = model.forward(**inputs)
end = time.time()
elapsed = end - start
print(f"Time elapsed for prefill with FA2: {elapsed}")

