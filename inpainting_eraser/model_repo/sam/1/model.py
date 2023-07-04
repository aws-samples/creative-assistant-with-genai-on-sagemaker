import json
import base64

from PIL import Image
from six import BytesIO

import numpy as np
import cv2
import os

import triton_python_backend_utils as pb_utils

from segment_anything import SamPredictor, sam_model_registry

class TritonPythonModel:
    
    def initialize(self, args):
        self.output_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(json.loads(args["model_config"]),
                                               "generated_image")["data_type"])
        
        self.model_dir = args['model_repository']
    
        device='cuda'
        
        sam = sam_model_registry["vit_h"](checkpoint=f"/home/models/sam_vit_h_4b8939.pth").to(device)
        
        self.predictor = SamPredictor(sam)

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
            ## point_coords
            ## point_labels
            ## dilate_kernel_size
            img = pb_utils.get_input_tensor_by_name(request, "image")
            img_decoded = base64.b64decode(img.as_numpy()[0][0].decode())
            
            with Image.open(BytesIO(img_decoded)) as f:
                img_rgb = f.convert("RGB")
                img_np_array = np.array(img_rgb)
                
            del img
            
            gen_args = pb_utils.get_input_tensor_by_name(request, "gen_args")
            
            gen_args_decoded = json.loads(gen_args.as_numpy()[0][0].decode())

            del gen_args
            
            # predictor ============
            
            self.predictor.set_image(img_np_array)
            
            point_coords = np.array([gen_args_decoded["point_coords"]])
            point_labels = np.array([gen_args_decoded["point_labels"]])

            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
            
            masks = masks.astype(np.uint8) * 255
            
            masks = [self.dilate_mask(mask, gen_args_decoded["dilate_kernel_size"]) for mask in masks]

            mask_img = Image.fromarray(masks[1].astype(np.uint8))
            output_img_bytes = self.encode_image(mask_img)

            output_image_obj = np.array([output_img_bytes], dtype="object").reshape((-1, 1))
            
#             mask_imgs = []
#             for mask in masks:
#                 mask_img = Image.fromarray(mask.astype(np.uint8))
#                 mask_imgs.append(encode_image_jpeg(mask_img))

            
            inference_response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor(
                    "generated_image",
                    output_image_obj
                )
            ])
            
            responses.append(inference_response)
        
        return responses


