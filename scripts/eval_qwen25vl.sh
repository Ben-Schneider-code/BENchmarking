export CUDA_VISIBLE_DEVICES="4,5,6,7"
python3 -m accelerate.commands.launch \
    --num_processes=4 \
    -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained="Qwen/Qwen2.5-VL-7B-Instruct,use_flash_attention_2=True" \
    --tasks longvideobench_val_v,mme,mmbench_en\
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix qwen_out \
    --output_path ./logs/