"""
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00,  8.45it/s]
qwen-vl-utils using decord to read video.
Time elapsed for video decompression: 19.405189514160156 [NOTE: This is usually much longer 30-40 seconds, if the cideo is out of disk cache]
Time elapsed for prefill with FA2: 6.688928127288818
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
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
end = time.time()
elapsed = end - start
print(f"Time elapsed for video decompression: {elapsed}")

inputs = inputs.to("cuda")

start = time.time()
with torch.no_grad():
    outputs = model.forward(**inputs)
end = time.time()
elapsed = end - start
print(f"Time elapsed for prefill with FA2: {elapsed}")

