import os
import io
import random
import argparse
from pathlib import Path
import json
import itertools
import time
import math
from safetensors import safe_open
import hashlib 

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from ip_adapter.ip_adapter import MLPProjModel
from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
    from ip_adapter.attention_processor_faceid import LoRAAttnProcessor2_0 as LoRAAttnProcessor, LoRAIPAttnProcessor2_0 as LoRAIPAttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor
    from ip_adapter.attention_processor_faceid import LoRAAttnProcessor, LoRAIPAttnProcessor


def adaptive_resize(image, size=1024, stride=64):
    w, h = image.size
    times = math.sqrt(h * w / (size**2))
    if w==h:
        w, h = size, size
    elif times > 1.1:
        w, h = math.ceil(w / times), math.ceil(h / times)
    elif times < 0.8:
        w, h = math.ceil(w / times), math.ceil(h / times)
    new_w, new_h = stride * (math.ceil(w / stride)), stride * (math.ceil(h / stride))
    image = image.resize([new_w, new_h])
    return image 

def make_square_region(x1, y1, x2, y2):
    center = [int((x1 + x2)/2), int((y1 + y2)/2)]
    w, h = int(x2 - x1 + 1) // 2 * 2, int(y2-y1+1) // 2 * 2
    distance = max(w, h)
    x1, y1, x2, y2 = center[0] - distance / 2, center[1] - distance / 2, center[0] + distance / 2, center[1] + distance / 2
    return x1, y1, x2, y2

def random_expand(x1, y1, x2, y2, width, height, max_expanding_percent=50):
    center_point = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
    w, h = x2 - x1, y2 - y1
    # random expanding ratio
    ratio = np.random.randint(low=0, high=max_expanding_percent) / 100
    ratio = min(ratio, (width / w - 1) / 2)
    ratio = min(ratio, (height / h - 1) / 2)
    x1 = x1 - int(ratio * w)
    y1 = y1 - int(ratio * h)
    x2 = x2 + int(ratio * w)
    y2 = y2 + int(ratio * h)
    # if the box is out of image, we need to shift it
    shift_x, shift_y = 0, 0
    if x1 < 0:
        shift_x = (-x1)
    elif x2 > width:
        shift_x = width - x2 - 1
    if y1 < 0:
        shift_y = (-y1)
    elif y2 > height:
        shift_y = height - y2 - 1
    x1, y1, x2, y2 = x1 + shift_x, y1 + shift_y, x2 + shift_x, y2 + shift_y
    return x1, y1, x2, y2

def read_image(image_file, image_root_path):
    raw_image = Image.open(os.path.join(image_root_path, image_file)).convert("RGB")      
    return raw_image


# Dataset
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 json_file, 
                 tokenizer, 
                 tokenizer_2, 
                 size=1024, 
                 center_crop=True, 
                 t_drop_rate=0.05, 
                 i_drop_rate=0.05, 
                 ti_drop_rate=0.05, 
                 truncate_rate=0.0, 
                 max_num_images=24, 
                 cond_image_size=336,
                 image_root_path="", 
                 random_crop=True, 
                 multi_ref_finetuning=False,
                 multimodal_llm_path="Qwen/Qwen2-VL-Instruct-2B"):
        super().__init__()

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.center_crop = center_crop
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.truncate_rate = truncate_rate
        self.image_root_path = image_root_path
        self.max_num_images = max_num_images
        self.random_crop = random_crop
        self.multi_ref_finetuning = multi_ref_finetuning
    
        self.data = json.load(open(json_file, 'r')) # list of dict: [{"image_file": "1.png", "text": "A dog"}]

        if self.random_crop:
            self.transform = transforms.Compose([
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])            
        self.resize_transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
        ])
        self.flip_transform = transforms.RandomHorizontalFlip()
        
        # min_pixels = 121 * 28 * 28
        # max_pixels = 145 * 28 * 28
        min_pixels = ((cond_image_size // 28 - 1)**2) * 28 * 28
        max_pixels = ((cond_image_size // 28)**2 + 1) * 28 * 28
        self.image_processor = AutoProcessor.from_pretrained(
            multimodal_llm_path, min_pixels=min_pixels, max_pixels=max_pixels)
    
    def __getitem__(self, idx):
        return self.read_data(idx)

    def read_data(self, idx):
        item = self.data[idx]
        if "face" in item:
            text = item["text"]
            image_file = item["target"]
            face = item["face"][0]
        elif "target" in item:
            text = item["text"]
            image_file = item["target"]
            cond_image_files = item["image_file"]

        if isinstance(text, list):
            if len(text)>0:
                text = text[np.random.randint(0, len(text))]
            else:
                text = ""        

        # read image
        Image.open(os.path.join(self.image_root_path, image_file)).convert("RGB")
        raw_image = read_image(image_file, self.image_root_path)

        # prepare face region
        if "face" in item:
            width, height = raw_image.size
            x1, y1, x2, y2 = face[0], face[1], face[2], face[3]
            x1, y1, x2, y2 = make_square_region(x1, y1, x2, y2)
            x1, y1, x2, y2 = random_expand(x1, y1, x2, y2, width, height)
            cond_image_files = [self.flip_transform(raw_image.crop((x1, y1, x2, y2)))]

        if not self.random_crop:
            # keep the original aspect ratio
            raw_image = adaptive_resize(raw_image, self.size)
         
        # prepare conditioning images
        if isinstance(cond_image_files, str):
            cond_image_files = [cond_image_files]
        # random shuffle augmentation
        random.shuffle(cond_image_files)
        cond_image_files = cond_image_files[ : self.max_num_images]
        # random truncation augmentation
        rand_num = random.random()
        if rand_num < self.truncate_rate and len(cond_image_files) > 2:
            num_conds = np.random.randint(2, len(cond_image_files))
            cond_image_files = cond_image_files[:num_conds]

        cond_raw_images = []
        for cond_image_file in cond_image_files:
            if isinstance(cond_image_file, str):
                cond_raw_image = read_image(cond_image_file, self.image_root_path)
                cond_raw_images.append(cond_raw_image)
            else:
                cond_raw_images.append(cond_image_file.convert("RGB"))

        # original size
        original_width, original_height = raw_image.size
        original_size = torch.tensor([original_height, original_width])
        
        image_tensor = self.transform(raw_image.convert("RGB"))
        # random crop
        if self.random_crop:
            delta_h = image_tensor.shape[1] - self.size
            delta_w = image_tensor.shape[2] - self.size
            assert not all([delta_h, delta_w])
            
            if self.center_crop:
                top = delta_h // 2
                left = delta_w // 2
            else:
                top = np.random.randint(0, delta_h + 1)
                left = np.random.randint(0, delta_w + 1)
            image = transforms.functional.crop(
                image_tensor, top=top, left=left, height=self.size, width=self.size
            )
            
            if len(cond_raw_images) == 1:
                # We center crop the single reference image to ensure consistency between reference image and target image
                cond_raw_images[0] = transforms.functional.crop(
                    self.resize_transform(cond_raw_images[0].convert("RGB")), top=top, left=left, height=self.size, width=self.size
                )
                
            target_size = torch.tensor([self.size, self.size])
        else:
            top = left = 0
            image = transforms.functional.crop(
                image_tensor, top=top, left=left, height=original_height, width=original_width
            )            
            target_size = torch.tensor([original_height, original_width])
        crop_coords_top_left = torch.tensor([top, left]) 

        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1

        system_prompt = "Visualize a scene that closely resembles the provided images, capturing the essence and details described in this prompt:\n"
        system_prompt = system_prompt + text

        if drop_image_embed:
            raw_image = Image.new(
                mode="RGB", size=(int(512), int(512))
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": raw_image},
                        {"type": "text", "text": system_prompt},
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [],
                }
            ]
            for cond in cond_raw_images:
                messages[0]["content"].append({"type": "image", "image": cond})
            messages[0]["content"].append({"type": "text", "text": system_prompt})

        prompt = self.image_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        if self.multi_ref_finetuning:
            inputs = self.image_processor(
                text=[prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )            
        else:
            inputs = self.image_processor(
                text=[prompt],
                images=image_inputs,
                videos=video_inputs,
                max_length=256,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        text_input_ids_2 = self.tokenizer_2(
            text,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "text_input_ids_2": text_input_ids_2,
            "drop_image_embed": drop_image_embed,
            "original_size": original_size,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size": target_size,
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
        }
        
    
    def __len__(self):
        return len(self.data)
    

def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    text_input_ids_2 = torch.cat([example["text_input_ids_2"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]
    original_size = torch.stack([example["original_size"] for example in data])
    crop_coords_top_left = torch.stack([example["crop_coords_top_left"] for example in data])
    target_size = torch.stack([example["target_size"] for example in data])
    input_ids = torch.stack([example["input_ids"] for example in data], dim=0)
    attention_mask = torch.cat([example["attention_mask"] for example in data], dim=0)
    pixel_values = [example["pixel_values"] for example in data]
    image_grid_thw = torch.stack([example["image_grid_thw"] for example in data], dim=0)

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "text_input_ids_2": text_input_ids_2,
        "drop_image_embeds": drop_image_embeds,
        "original_size": original_size,
        "crop_coords_top_left": crop_coords_top_left,
        "target_size": target_size,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }
    

class EasyRef(torch.nn.Module):
    """EasyRef"""
    def __init__(self, unet, multimodal_llm, mllm_final_layer, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.multimodal_llm = multimodal_llm
        self.mllm_final_layer = mllm_final_layer
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        for i in range(len(self.mllm_final_layer.layers)):
            self.mllm_final_layer.layers[i].self_attn.is_causal = False

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def multimodal_llm_forward(self, image_cond_kwargs):
        input_ids = image_cond_kwargs["input_ids"]
        attention_mask = image_cond_kwargs["attention_mask"]
        pixel_values = image_cond_kwargs["pixel_values"]
        image_grid_thw = image_cond_kwargs["image_grid_thw"]
        inputs_embeds = self.multimodal_llm.model.embed_tokens(input_ids)
        new_inputs_embeds = []
        for i in range(len(pixel_values)):
            pixel_value = pixel_values[i].type(self.multimodal_llm.visual.get_dtype()).to(inputs_embeds.device)
            grid_thw = image_grid_thw[i]
            cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
                dim=0, dtype=torch.int32
            )
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)  
            image_embeds = []          
            for j in range(1, len(cu_seqlens)):
                image_embed = self.multimodal_llm.visual(pixel_value[cu_seqlens[j - 1] : cu_seqlens[j]], grid_thw=grid_thw[(j - 1) : j]).to(inputs_embeds.device)
                image_embeds.append(image_embed)
            image_embeds = torch.cat(image_embeds, dim=0)
            image_mask = input_ids[i] == self.multimodal_llm.config.image_token_id
            inputs_embed = inputs_embeds[i].clone()
            inputs_embed[image_mask] = image_embeds
            new_inputs_embeds.append(inputs_embed)
        inputs_embeds = torch.cat(new_inputs_embeds, dim=0)

        image_embeds = self.multimodal_llm(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True
        ).hidden_states[-1] 
        return image_embeds

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, unet_added_cond_kwargs, image_embeds, attention_mask, drop_image_embeds, image_cond_kwargs=None, unfreeze_mllm=False):
        if image_cond_kwargs is not None and image_embeds is None:
            if unfreeze_mllm:
                image_embeds = self.multimodal_llm_forward(image_cond_kwargs)
            else:
                with torch.no_grad():
                    image_embeds = self.multimodal_llm_forward(image_cond_kwargs)
        reference_tokens = self.mllm_final_layer.reference_tokens
        image_embeds = torch.cat([image_embeds, reference_tokens.unsqueeze(0).repeat(image_embeds.shape[0], 1, 1)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, :reference_tokens.shape[0]])], dim=1)
        image_embeds = self.mllm_final_layer(
            attention_mask=attention_mask,
            inputs_embeds=image_embeds,
            output_hidden_states=True             
        ).hidden_states[-1]

        image_embeds_ = []
        for image_embed, drop_image_embed in zip(image_embeds, drop_image_embeds):
            new_image_embed = image_embed[-reference_tokens.shape[0]:]
            image_embeds_.append(new_image_embed)
        image_embeds = torch.stack(image_embeds_)

        ip_tokens = self.image_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=unet_added_cond_kwargs).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):

        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p).to("cuda") for p in self.image_proj_model.parameters()]))
        orig_final_layer_sum = torch.sum(torch.stack([torch.sum(p).to("cuda") for p in self.mllm_final_layer.parameters()]))
        orig_mllm_sum = torch.sum(torch.stack([torch.sum(p).to("cuda") for p in self.multimodal_llm.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p).to("cuda") for p in self.adapter_modules.parameters()]))

        if os.path.splitext(ckpt_path)[-1] == ".safetensors":
            state_dict = {"image_proj_model": {}, "mllm_final_layer": {}, "multimodal_llm": {}, "unet": {}}
            with safe_open(ckpt_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj_model."):
                        state_dict["image_proj_model"][key.replace("image_proj_model.", "")] = f.get_tensor(key)
                    elif key.startswith("mllm_final_layer."):
                        state_dict["mllm_final_layer"][key.replace("mllm_final_layer.", "")] = f.get_tensor(key)
                    elif key.startswith("multimodal_llm."):
                        state_dict["multimodal_llm"][key.replace("multimodal_llm.", "")] = f.get_tensor(key)   
                    elif key.startswith("unet."):
                        state_dict["unet"][key.replace("unet.", "")] = f.get_tensor(key)
        else:
            state_dict = {"image_proj_model": {}, "mllm_final_layer": {}, "multimodal_llm": {}, "unet": {}}
            f = torch.load(ckpt_path, map_location="cpu")["module"]
            for key in f.keys():
                if key.startswith("image_proj_model."):
                    state_dict["image_proj_model"][key.replace("image_proj_model.", "")] = f[key]
                elif key.startswith("mllm_final_layer."):
                    state_dict["mllm_final_layer"][key.replace("mllm_final_layer.", "")] = f[key]
                elif key.startswith("multimodal_llm."):
                    state_dict["multimodal_llm"][key.replace("multimodal_llm.", "")] = f[key]
                elif key.startswith("unet."):
                    state_dict["unet"][key.replace("unet.", "")] = f[key]

        if len(list(state_dict["multimodal_llm"].keys())) > 0:               
            self.multimodal_llm.load_state_dict(state_dict["multimodal_llm"])            
        self.image_proj_model.load_state_dict(state_dict["image_proj_model"], strict=True)
        self.mllm_final_layer.load_state_dict(state_dict["mllm_final_layer"], strict=True)
        unet_state_dict = self.unet.state_dict()
        unet_state_dict.update(state_dict["unet"])
        self.unet.load_state_dict(unet_state_dict, strict=False)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p).to("cuda") for p in self.image_proj_model.parameters()]))
        new_final_layer_sum = torch.sum(torch.stack([torch.sum(p).to("cuda") for p in self.mllm_final_layer.parameters()]))
        new_mllm_sum = torch.sum(torch.stack([torch.sum(p).to("cuda") for p in self.multimodal_llm.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p).to("cuda") for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_final_layer_sum != new_final_layer_sum, "Weights of mllm_final_layer did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"
        if orig_mllm_sum == new_mllm_sum:
            print("Weights of multimodal_llm did not change!")

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")
    

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_easy_ref_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default=None,
        required=True,
        help="Training data",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="",
        required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--multimodal_llm_path",
        type=str,
        default=None,
        required=True,
        help="Path to Multimodal LLM",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-easy_ref",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )    
    parser.add_argument("--t_drop_rate", type=float, default=0.05, help="Prob to drop text condition.")
    parser.add_argument("--truncate_rate", type=float, default=0, help="Prob to truncate condition images.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument('--random_crop', action='store_true')
    parser.add_argument('--multi_ref_finetuning', action='store_true')
    parser.add_argument('--unfreeze_mllm', action='store_true')
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--resume_from_pretrained', action='store_true')
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--noise_offset", type=float, default=None, help="noise offset")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--num_reference_tokens",
        type=int,
        default=64,
        help=(
            "Number of reference tokens used in the LLM final layer"
        ),
    )
    parser.add_argument(
        "--max_num_images",
        type=int,
        default=28,
        help=(
            "Max number of reference images"
        ),
    )
    parser.add_argument(
        "--cond_image_size",
        type=int,
        default=336,
        help=(
            "Resolution of reference image"
        ),
    )         
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    args = parser.parse_args()

    print(args)
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
    

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # default: Load the model on the available device(s)
    mllm_final_layer = Qwen2VLForConditionalGeneration.from_pretrained(
        args.multimodal_llm_path, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="sdpa",
        device_map="cuda"
    )
    mllm_final_layer = mllm_final_layer.model
    mllm_final_layer.layers = mllm_final_layer.layers[-1:]
    mllm_final_layer.embed_tokens = torch.nn.Identity()
    mllm_final_layer.visual = torch.nn.Identity()
    mllm_final_layer.lm_head = torch.nn.Identity()
    mllm_final_layer.reference_tokens = torch.nn.Parameter(0.1 * torch.randn(args.num_reference_tokens, mllm_final_layer.config.hidden_size))

    multimodal_llm = Qwen2VLForConditionalGeneration.from_pretrained(
        args.multimodal_llm_path, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="sdpa",
        device_map="cuda"
    )
    multimodal_llm.model.layers = multimodal_llm.model.layers[:-1]
    multimodal_llm.norm = torch.nn.Identity()
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    if not args.unfreeze_mllm:
        multimodal_llm.requires_grad_(False)
    
    image_proj_model = MLPProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=multimodal_llm.config.hidden_size,
    )
    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            if args.use_lora:
                attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=args.lora_rank)
            else:
                attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            if args.use_lora:
                attn_procs[name] = LoRAIPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=args.num_reference_tokens, rank=args.lora_rank)
                attn_procs[name].load_state_dict(weights, strict=False)                
            else:
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=args.num_reference_tokens)
                attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    
    easy_ref = EasyRef(unet, multimodal_llm, mllm_final_layer, image_proj_model, adapter_modules, args.pretrained_easy_ref_path)
    # easy_ref.multimodal_llm = multimodal_llm

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    #unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device) # use fp32
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    #multimodal_llm.to(accelerator.device, dtype=weight_dtype)
    
    # optimizer
    if args.unfreeze_mllm:
        params_to_opt = itertools.chain(easy_ref.mllm_final_layer.parameters(), easy_ref.image_proj_model.parameters(),  easy_ref.adapter_modules.parameters(), easy_ref.multimodal_llm.parameters())
    else:
        params_to_opt = itertools.chain(easy_ref.mllm_final_layer.parameters(), easy_ref.image_proj_model.parameters(),  easy_ref.adapter_modules.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)

    # dataloader
    train_dataset = MyDataset(
        args.data_json_file, 
        tokenizer=tokenizer, 
        tokenizer_2=tokenizer_2, 
        size=args.resolution, 
        t_drop_rate=args.t_drop_rate, 
        truncate_rate=args.truncate_rate, 
        max_num_images=args.max_num_images, 
        cond_image_size=args.cond_image_size,        
        image_root_path=args.data_root_path, 
        random_crop=args.random_crop, 
        multi_ref_finetuning=args.multi_ref_finetuning,
        multimodal_llm_path=args.multimodal_llm_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes / args.gradient_accumulation_steps
    len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
    num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
    num_training_steps_for_scheduler = (
        args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    easy_ref, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(easy_ref, optimizer, train_dataloader, lr_scheduler)
    
    if args.resume_from_pretrained:
        if args.pretrained_easy_ref_path.endswith(".pt"):
            resume_path = "/".join(args.pretrained_easy_ref_path.split("/")[:-2])
        elif args.pretrained_easy_ref_path.endswith("safetensors"):
            resume_path = "/".join(args.pretrained_easy_ref_path.split("/")[:-1])
        else:
            resume_path = args.pretrained_easy_ref_path
        accelerator.load_state(resume_path)

    global_step = 0
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(easy_ref):
                # Convert images to latent space
                with torch.no_grad():
                    # vae of sdxl should use fp32
                    latents = vae.encode(batch["images"].to(accelerator.device, dtype=torch.float32)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents = latents.to(accelerator.device, dtype=weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1)).to(accelerator.device, dtype=weight_dtype)

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
                input_ids = batch["input_ids"].to(accelerator.device)
                attention_mask = batch["attention_mask"].to(accelerator.device)
                # pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                pixel_values = batch["pixel_values"]
                image_grid_thw = batch["image_grid_thw"].to(accelerator.device)

                image_embeds = None
                image_cond_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "pixel_values": pixel_values, "image_grid_thw": image_grid_thw}                 
            
                with torch.no_grad():
                    encoder_output = text_encoder(batch['text_input_ids'].to(accelerator.device), output_hidden_states=True)
                    text_embeds = encoder_output.hidden_states[-2]
                    encoder_output_2 = text_encoder_2(batch['text_input_ids_2'].to(accelerator.device), output_hidden_states=True)
                    pooled_text_embeds = encoder_output_2[0]
                    text_embeds_2 = encoder_output_2.hidden_states[-2]
                    text_embeds = torch.concat([text_embeds, text_embeds_2], dim=-1) # concat
                        
                # add cond
                add_time_ids = [
                    batch["original_size"].to(accelerator.device),
                    batch["crop_coords_top_left"].to(accelerator.device),
                    batch["target_size"].to(accelerator.device),
                ]
                add_time_ids = torch.cat(add_time_ids, dim=1).to(accelerator.device, dtype=weight_dtype)
                unet_added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids}
                
                noise_pred = easy_ref(noisy_latents, timesteps, text_embeds, unet_added_cond_kwargs, image_embeds, attention_mask, batch["drop_image_embeds"], image_cond_kwargs, args.unfreeze_mllm)
                
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    print("Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}, lr: {}".format(
                        epoch, step, load_data_time, time.perf_counter() - begin, avg_loss, lr_scheduler.get_last_lr()[0]))
            
            global_step += 1
            
            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
            
            begin = time.perf_counter()
                
if __name__ == "__main__":
    main()    
