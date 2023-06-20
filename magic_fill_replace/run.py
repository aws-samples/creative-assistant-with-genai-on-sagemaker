import gradio as gr
import boto3
import base64
import io
import json
from PIL import Image
import PIL

runtime_sm_client = boto3.client("sagemaker-runtime")

fill_samples = [
    ["statics/dog.jpg", "Replace Background", "dog at a space museum"],
    ["statics/sample1.png", "Fill Object", "a teddy bear on a bench"],
    ["statics/sample1.png", "Fill Object", "a hamster on a bench"],
    ["statics/sample1.png", "Fill Object", ""],
    ["statics/sample1.png", "Replace Background", "dog sit on the swing"],
    ["statics/bus.jpeg", "Fill Object", "a sports car on a road"],
    ["statics/bus.jpeg", "Replace Background", "a bus, on the center of a country road, summer"]
]

# write a function that loads from a config file into dictionary
with open("config.json", "r") as f:
    config = json.load(f)

endpoint_name = config["endpoint_name"]
sam_model = config["models"]["sam"]
sd_inpaint_model = config["models"]["sd_inpaint"]

def encode_image(img):
    # Convert the image to bytes
    with io.BytesIO() as output:
        img.save(output, format="JPEG")
        img_bytes = output.getvalue()

    return base64.b64encode(img_bytes).decode('utf8')


with gr.Blocks(theme=gr.themes.Soft(primary_hue="emerald")) as demo:
    with gr.Row():
        gr.Markdown(
            """
            # Creative Assistant Powered By Generative AI
            
            Experience a new world of visual editing powered by generative AI. **Click and select any object on the input image to erase or fill.**
            """
        )

    with gr.Row():
        with gr.Column(scale=1):
            fill_output_img = gr.Image(label="Output Image", type="pil").style(height=400)
            fill_options = gr.Radio(choices=["Fill Object", "Replace Background"],
                                    value="Fill Object", type="index", show_label=False)
            fill_button = gr.Button(value="Generate")
            fill_prompt = gr.Textbox(label="Prompt", info="Supply text on things you want to generate")
            fill_nprompt = gr.Textbox(label="Negative prompt",
                                 info="Supply a text on things you don't want to generate",
                                 value="ugly, tiling, poorly drawn hands, poorly drawn feet, out of frame," +
                                       "extra limbs, disfigured, deformed, body out of frame, blurry," +
                                       "blurred, watermark, grainy, signature, cut off, multiple, gross," +
                                       "weird, uneven, text, poor, low, basic, worst unprofessional")
            fill_inference_step = gr.Slider(1, 100, value=50, label="Number of inference steps")
            fill_guidance_scale = gr.Slider(0, 20, value=7.5, label="Guidance scale",
                                            info="Choose between 0 and 20")
            fill_seed = gr.Slider(1, 10000000, value=10, label="Random Seed")
        with gr.Column(scale=1):
            fill_input_img = gr.Image(label="Input", type="pil").style(height=400)
            fill_original_img = gr.Image(type="pil", visible=False)
            fill_mask = gr.Image(label="mask", visible=False, type="pil")
            sample_img = gr.Examples(fill_samples, [fill_input_img, fill_options, fill_prompt])

    def get_select_coords(img, evt: gr.SelectData):
        original = img.copy()
        original_image_bytes = encode_image(img)
        gen_args = json.dumps(dict(point_coords=[evt.index[0], evt.index[1]], point_labels=1, dilate_kernel_size=15))

        inputs = dict(image=original_image_bytes,
                      gen_args=gen_args)

        payload = {
            "inputs":
                [{"name": name, "shape": [1, 1], "datatype": "BYTES", "data": [data]} for name, data in inputs.items()]
        }

        response = runtime_sm_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/octet-stream",
            Body=json.dumps(payload),
            TargetModel=sam_model,
        )

        output = json.loads(response["Body"].read().decode("utf8"))["outputs"]

        mask_decoded = io.BytesIO(base64.b64decode(output[0]["data"][0]))
        mask_rgb = Image.open(mask_decoded).convert("RGB")

        mask_rgb.putalpha(128)

        img.paste(mask_rgb, (0,0), mask_rgb)

        return img, original, mask_rgb

    def fill_object(img, mask_rgb, fill_option, p, np, inf_steps=50, scale=10, s=1):

        original_image_bytes = encode_image(img)

        if fill_option == 0:
            mask_image = encode_image(mask_rgb)
        else:
            inv_mask_rgb = PIL.ImageOps.invert(mask_rgb)
            mask_image = encode_image(inv_mask_rgb)

        gen_args = json.dumps(dict(num_inference_steps=inf_steps, guidance_scale=scale, seed=s))

        inputs = dict(image=original_image_bytes,
                      mask_image=mask_image,
                      prompt=p,
                      negative_prompt=np,
                      gen_args=gen_args)

        payload = {
            "inputs":
                [{"name": name, "shape": [1, 1], "datatype": "BYTES", "data": [data]} for name, data in inputs.items()]
        }

        response = runtime_sm_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/octet-stream",
            Body=json.dumps(payload),
            TargetModel=sd_inpaint_model,
        )

        output = json.loads(response["Body"].read().decode("utf8"))["outputs"]

        mask_decoded = io.BytesIO(base64.b64decode(output[0]["data"][0]))
        mask_rgb = Image.open(mask_decoded).convert("RGB")

        return mask_rgb

    def reset_images():
        # reset Eraser components
        return None, None, None

    # Events ===============================
    # [FILL TAB Actions]
    fill_input_img.select(get_select_coords, fill_input_img, [fill_input_img, fill_original_img, fill_mask])
    fill_button.click(fill_object, [fill_original_img, fill_mask, fill_options, fill_prompt,
                             fill_nprompt, fill_inference_step, fill_guidance_scale, fill_seed], fill_output_img)
    fill_input_img.clear(reset_images, None, [fill_original_img, fill_mask, fill_output_img])
    
if __name__ == "__main__":
    demo.launch(share=True)