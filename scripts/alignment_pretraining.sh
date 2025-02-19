accelerate launch --multi_gpu --num_processes 32 \
  --num_machines ${WORLD_SIZE} \
  --machine_rank ${RANK} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --mixed_precision "bf16" \
  finetune_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --image_encoder_path="Qwen/Qwen2-VL-2B-Instruct" \
  --data_json_file="{data.json}" \
  --data_root_path="{image_path}" \
  --mixed_precision="bf16" \
  --resolution=1024 \
  --train_batch_size=8 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-04 \
  --weight_decay=0.0 \
  --lr_warmup_steps=0 \
  --t_drop_rate=0.05 \
  --num_train_epochs=100 \
  --gradient_accumulation_steps=1 \
  --num_reference_tokens=64 \
  --random_crop \
  --output_dir="{outputs/alignment_pretraining}" \
  --save_steps=20000