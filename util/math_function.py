import torch

def invert_softmax(attention_tensor, eps=1e-8):
    # Add epsilon to avoid log(0)
    safe_attention = attention_tensor + eps
    # Element-wise logarithm to invert softmax
    logits = torch.log(safe_attention)
    return logits