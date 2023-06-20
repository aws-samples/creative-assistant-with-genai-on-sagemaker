import os
import sys
import numpy as np
import torch
import yaml
import glob
import argparse
from PIL import Image
from omegaconf import OmegaConf
from pathlib import Path
import json
import base64
from io import BytesIO

import triton_python_backend_utils as pb_utils

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

sys.path.insert(0, str(Path(__file__).resolve().parent / "lama"))
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.data import pad_tensor_to_modulo


class TritonPythonModel:
    
    def initialize(self, args):
        self.output_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(json.loads(args["model_config"]),
                                               "generated_image")["data_type"])
        
        self.model_dir = args['model_repository']
    
        device='cuda'
        
        print(os.listdir(self.model_dir))
        
        self.predict_config = OmegaConf.load(f'{self.model_dir}/1/lama/configs/prediction/default.yaml')
        self.predict_config.model.path = f'/home/models/big-lama'

        with open(f'{self.predict_config.model.path}/config.yaml', 'r') as f:
            self.train_config = OmegaConf.create(yaml.safe_load(f))

        self.train_config.training_model.predict_only = True
        
        self.train_config.visualizer.kind = 'noop'
        
        checkpoint_path = os.path.join(
            self.predict_config.model.path, 'models',
            self.predict_config.model.checkpoint
        )
        self.model = load_checkpoint(
            self.train_config, 
            checkpoint_path, 
            strict=False, 
            map_location='cpu')
        
        self.model.freeze()
        self.model.to(device)


    def encode_image(self, img): 
        # Convert the image to bytes
        with BytesIO() as output:
            img.save(output, format="JPEG")
            img_bytes = output.getvalue()

        return base64.b64encode(img_bytes).decode()
    
    def dilate_mask(self, mask, dilate_factor=15):
        mask = mask.astype(np.uint8)
        mask = cv2.dilate(
            mask,
            np.ones((dilate_factor, dilate_factor), np.uint8),
            iterations=1
        )
        return mask
    
    def execute(self, requests):
        responses = []
        for request in requests:
            
            ## image
            ## mask
            mod = 8
            device='cuda'

            img = pb_utils.get_input_tensor_by_name(request, "image")
            img_decoded = base64.b64decode(img.as_numpy()[0][0].decode())
            
            with Image.open(BytesIO(img_decoded)) as f:
                img_rgb = f.convert("RGB")
                img_np_array = np.array(img_rgb)
                
            del img
            
            mask = pb_utils.get_input_tensor_by_name(request, "mask_image")
            mask_decoded = base64.b64decode(mask.as_numpy()[0][0].decode())
            
            with Image.open(BytesIO(mask_decoded)) as f:
                mask_l = f.convert("L")
                mask_np_array = np.array(mask_l)
                
            del mask
            
            img = torch.from_numpy(img_np_array).float().div(255.)
            mask = torch.from_numpy(mask_np_array).float()

            batch = {}
            batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
            batch['mask'] = mask[None, None]
            unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
            batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
            batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
            batch = move_to_device(batch, device)
            batch['mask'] = (batch['mask'] > 0) * 1
            
            batch = self.model(batch)
            cur_res = batch['inpainted'][0].permute(1, 2, 0)
            cur_res = cur_res.detach().cpu().numpy()
            
            if unpad_to_size is not None:
                orig_height, orig_width = unpad_to_size
                cur_res = cur_res[:orig_height, :orig_width]
                
            cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')

            output_img = Image.fromarray(cur_res)
            output_img_bytes = self.encode_image(output_img)

            output_image_obj = np.array([output_img_bytes], dtype="object").reshape((-1, 1))
            inference_response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor(
                    "generated_image",
                    output_image_obj
                )
            ])
            
            responses.append(inference_response)
        
        return responses


