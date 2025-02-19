import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from util.plot import plot_heatmap, freq_plot
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from util.short_story import *
from util.math_function import *

model_path = "Qwen/Qwen2-VL-7B-Instruct"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

# Video 
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "video",
#                 "video": "file:///scratch/b3schnei/data/video/ny.mp4",
#                 "max_pixels": 224*224,
#                 "fps": 1.0,
#             },
#             {"type": "text", "text": f"What famous location is this?"},
#         ],
#     }
# ]

messages = [
    {
        "role": "user",
        "content": [
            # {
            #     "type": "video",
            #     "video": "file:///scratch/b3schnei/data/video/hamburger.mp4",
            #     "max_pixels": 224*360,
            #     "fps": 1.0,
            # },
            {"type": "text", "text": f"{short_story}\nGive me a two sentence summary of this story."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

with torch.no_grad():
    outputs = model.forward(
        **inputs,
        output_attentions=True
    )

attn = outputs.attentions

first_attn = attn[0].float().cpu()
mid_attn = attn[9].float().cpu()
last_attn = attn[-1].float().cpu()
del attn

def color_frames(attns):
    offset = 15
    num_frames = 22
    color = 1
    dynamic_offset = 60
    for j in range(num_frames):
        for i in range(len(attns)):
            attns[i][-60:-1,(j*60)+offset:(j*60)+60+offset] = color%2
        color += 1
    return attns

def get_attn_matrix(t: torch.Tensor):
    t = t.squeeze(dim=0)
    mean_pooled_attn = t.mean(dim=0, keepdim=False)
    arb_chosen_attn = t[0]
    attns = mean_pooled_attn,  arb_chosen_attn
    #attns = color_frames(attns)
    return attns 

# seq_len x seq_len fp32 tensor on cpu
fm, fa = get_attn_matrix(first_attn)
mm, ma = get_attn_matrix(mid_attn)
lm, la = get_attn_matrix(last_attn)

fm = fm[:,15:-11]
fa = fa[:,15:-11]
mm = mm[:,15:-11]
ma = ma[:,15:-11]
lm = lm[:,15:-11]
la = la[:,15:-11]

l = la.mean(axis=1)
f = fa.mean(axis=1)

freq_plot(l, "./plot/last_freq.png")
freq_plot(f, "./plot/first_freq.png")