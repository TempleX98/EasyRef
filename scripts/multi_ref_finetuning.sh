accelerate launch --use_deepspeed --num_processes 32 \
  --num_machines ${WORLD_SIZE} \
  --machine_rank ${RANK} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --mixed_precision "bf16" \
  --deepspeed_multinode_launcher standard \
  --zero_stage 2 --offload_param_device none --offload_optimizer_device none --gradient_accumulation_steps 4 --zero3_init_flag false \
  finetune_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --pretrained_easy_ref_path="{outputs/single_ref_finetuning/checkpoint-250000/pytorch_model/mp_rank_00_model_states.pt}" \
  --image_encoder_path="Qwen/Qwen2-VL-2B-Instruct" \
  --data_json_file="{data.json}" \
  --data_root_path="{image_path}" \
  --mixed_precision="bf16" \
  --resolution=1024 \
  --train_batch_size=1 \
  --dataloader_num_workers=2 \
  --learning_rate=2e-05 \
  --weight_decay=0.0 \
  --lr_warmup_steps=0 \
  --t_drop_rate=0.05 \
  --truncate_rate=0.4 \
  --num_train_epochs=100 \
  --gradient_accumulation_steps=4 \
  --num_reference_tokens=64 \
  --multi_ref_finetuning \
  --unfreeze_mllm \
  --use_lora \
  --lora_rank=128 \
  --output_dir="{outputs/multi_ref_finetuning}" \
  --save_steps=20000

#/mnt/afs/zongzhuofan/project/IP-Adapter/checkpoints/pretrain/qwen2_vl_2b_1resampler_64query_tdrop0.05_wd0_black_res1024_10mdata/checkpoint-300000/model.safetensors