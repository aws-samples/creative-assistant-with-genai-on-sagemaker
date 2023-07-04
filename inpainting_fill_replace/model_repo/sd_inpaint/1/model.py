import json
import base64
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from PIL import Image
from io import BytesIO

from torch import autocast
from torch.utils.dlpack import to_dlpack, from_dlpack
from diffusers import StableDiffusionInpaintPipeline

from mask_processing import crop_for_filling_pre, crop_for_filling_post

class TritonPythonModel:

    def initialize(self, args):
        self.output_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(json.loads(args["model_config"]),
                                               "generated_image")["data_type"])
        
        self.model_dir = args['model_repository']
    
        device='cuda'
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained("/home/models/stable_diff_inpaint",
                                                             torch_dtype=torch.float16).to(device)
        
        self.pipe.enable_xformers_memory_efficient_attention()

        
        # This line of code does offload of model parameters to the CPU and only pulls them into the GPU as they are needed
        # Not tested with MME, since it will likely provoke CUDA OOM errors.
        #self.pipe.enable_sequential_cpu_offload()
        
    def encode_image(self, img): 
        # Convert the image to bytes
        with BytesIO() as output:
            img.save(output, format="JPEG")
            img_bytes = output.getvalue()

        return base64.b64encode(img_bytes).decode()
    
    def execute(self, requests):
        responses = []
        for request in requests:
            
            # Load inputs =============
            prompt_object = pb_utils.get_input_tensor_by_name(request, "prompt")
            prompt_text = prompt_object.as_numpy()[0][0].decode()
            
            nprompt_object = pb_utils.get_input_tensor_by_name(request, "negative_prompt")
            nprompt_text = nprompt_object.as_numpy()[0][0].decode()
            
            img = pb_utils.get_input_tensor_by_name(request, "image")
            img_decoded = base64.b64decode(img.as_numpy()[0][0].decode())
            
            with Image.open(BytesIO(img_decoded)) as f:
                img_rgb = f.convert("RGB")
                img_np_array = np.array(img_rgb)
                
            del img
            
            mask = pb_utils.get_input_tensor_by_name(request, "mask_image")
            mask_decoded = base64.b64decode(mask.as_numpy()[0][0].decode())
            
            with Image.open(BytesIO(mask_decoded)) as f:
                mask_rgb = f.convert("L")
                mask_np_array = np.array(mask_rgb)
                
            del mask
            
            gen_args = pb_utils.get_input_tensor_by_name(request, "gen_args")
            
            gen_args_decoded = json.loads(gen_args.as_numpy()[0][0].decode())

            del gen_args
            
            generator = [torch.Generator(device="cuda").manual_seed(gen_args_decoded['seed'])]
            
            img_crop, mask_crop = crop_for_filling_pre(img_np_array, mask_np_array)
            
            with torch.no_grad():
                img_crop_filled = self.pipe(
                    prompt = prompt_text,
                    negative_prompt = nprompt_text,
                    image = Image.fromarray(img_crop),
                    mask_image=Image.fromarray(mask_crop),
                    num_inference_steps=gen_args_decoded['num_inference_steps'],
                    guidance_scale=gen_args_decoded['guidance_scale'],
                    generator=generator,
                ).images[0]
            
            image_array = crop_for_filling_post(img_np_array, mask_np_array, 
                                               np.array(img_crop_filled))

            generated_image = Image.fromarray(np.squeeze(image_array))
            
            output_img_bytes = self.encode_image(generated_image)
            
            output_image_obj = np.array([output_img_bytes], dtype="object").reshape((-1, 1))

            
            inference_response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor(
                    "generated_image",
                    output_image_obj
                )
            ])
            
            responses.append(inference_response)
        
        return responses
