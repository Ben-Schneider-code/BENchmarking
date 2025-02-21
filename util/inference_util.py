import time 
import torch

def prefill(model, inputs, timer=False):
    start = time.time()
    with torch.no_grad():
        outputs = model.forward(**inputs, use_cache=True)
    end = time.time()
    elapsed = end - start
    if timer: print(f"Time elapsed for prefill with FA2: {elapsed}")
    return outputs.past_key_values