import time
from qwen_vl_utils import process_vision_info

def load_video(video_path,resolution, processor, timer=False, template=False):
    start = time.time()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{video_path}",
                    #"max_pixels": 360 * 420, # default res
                    "max_pixels": resolution, # the video's native resolution
                    "fps": 1.0,
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=template
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
    if timer: print(f"Total time elapsed for video decompression: {elapsed}")
    return inputs