import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import util.hf_patches
from util.inference_util import prefill
from util.qwen_util import load_video 
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import time
def main():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)
    video_path = "/scratch/b3schnei/data/video/60min.mp4"
    model.cuda() 
    # Can run multiple time to get average time.
    # This timing is more inconsistent
    inputs = load_video(video_path, 360*420, processor, timer=True, template=True)
    inputs = inputs.to("cuda")

    input_shape = inputs.data["input_ids"].shape[1]
    print(f"Video tensor shape: {input_shape}")
    keys = prefill(model=model, inputs=inputs, timer=True)

    # add prefill keys
    inputs["past_key_values"] = keys

    def generate_with_cache(inputs, timer=False):
        start = time.time()
        generated_tokens = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,  # Disable sampling for greedy decoding
            temperature=1.0,  # Neutral temperature (ignored in greedy decoding)
        )
        end = time.time()
        elapsed =end-start
        if timer: print(f"Generation with KV cache took: {elapsed}")
        return generated_tokens

    generated_tokens = generate_with_cache(inputs=inputs, timer=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    new_tokens = generated_tokens.shape[1] - inputs.data["input_ids"].shape[1]
    print(f"Number of new tokens generated was: {new_tokens}")
    gen = generated_tokens[:, -new_tokens:]
    generated_text = tokenizer.batch_decode(gen, skip_special_tokens=True)
    
    print("Generated Output:")
    print(generated_text)


if __name__ == "__main__":
    with torch.no_grad():
        main()