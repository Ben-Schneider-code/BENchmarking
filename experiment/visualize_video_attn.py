import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from util.plot import plot_heatmap
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from util.short_story import *

model_path = "Qwen/Qwen2-VL-7B-Instruct"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

# Messages containing a video and a text query
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
            {"type": "text", "text": f"{short_story}\nGive me a two sentence summary of the short story you just read."},
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


# Generate 100 tokens to sanity check
# with torch.no_grad():
#     generated_ids = model.generate(
#         **inputs,
#         max_new_tokens=100,  # Generate up to 100 new tokens
#         do_sample=True,      # Enable sampling for more diverse outputs
#         top_k=50,            # Limit sampling to the top-k tokens
#         top_p=0.95,          # Nucleus sampling (top-p sampling)
#     )

# # Decode the generated tokens to text
# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(generated_text)
# exit()

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

def get_attn_matrix(t: torch.Tensor):
    t = t.squeeze(dim=0)
    mean_pooled_attn = t.mean(dim=0, keepdim=False)
    arb_chosen_attn = t[0]
    return mean_pooled_attn, arb_chosen_attn

# seq_len x seq_len fp32 tensor on cpu
fm, fa = get_attn_matrix(first_attn)
mm, ma = get_attn_matrix(mid_attn)
lm, la = get_attn_matrix(last_attn)

plot_heatmap(fm, "./plot/first_mean_attn.png")
plot_heatmap(fa, "./plot/first_arb_attn.png")
plot_heatmap(mm, "./plot/mid_mean_attn.png")
plot_heatmap(ma, "./plot/mid_arb_attn.png")
plot_heatmap(lm, "./plot/last_mean_attn.png")
plot_heatmap(la, "./plot/last_arb_attn.png")