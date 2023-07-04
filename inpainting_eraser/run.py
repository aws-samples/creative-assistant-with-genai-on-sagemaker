import gradio as gr
import boto3
import base64
import io
import json
from PIL import Image
import PIL

runtime_sm_client = boto3.client("sagemaker-runtime")

eraser_samples = ["statics/dog.jpg", "statics/sample1.png"]

# write a function that loads from a config file into dictionary
with open("config.json", "r") as f:
    config = json.load(f)

endpoint_name = config["endpoint_name"]
sam_model = config["models"]["sam"]
lama_model = config["models"]["lama"]

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
            eraser_input_img = gr.Image(label="Input", type="pil").style(height=400)
            eraser_original_img = gr.Image(type="pil", visible=False)
            eraser_mask = gr.Image(label="mask", visible=False, type="pil")
            eraser_sample_img = gr.Examples(eraser_samples, eraser_input_img)
        with gr.Column(scale=1):
            eraser_output_img = gr.Image(label="Output Image", type="pil").style(height=400)
            eraser_button = gr.Button(value="Erase")

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

    def rm_object(img, mask_rgb):
        original_image_bytes = encode_image(img)
        mask_image = encode_image(mask_rgb)

        inputs = dict(image=original_image_bytes,
                      mask_image=mask_image)

        payload = {
            "inputs":
                [{"name": name, "shape": [1, 1], "datatype": "BYTES", "data": [data]} for name, data in inputs.items()]
        }

        response = runtime_sm_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/octet-stream",
            Body=json.dumps(payload),
            TargetModel=lama_model,
        )

        output = json.loads(response["Body"].read().decode("utf8"))["outputs"]

        mask_decoded = io.BytesIO(base64.b64decode(output[0]["data"][0]))
        mask_rgb = Image.open(mask_decoded).convert("RGB")

        return mask_rgb

    def reset_images():
        # reset Eraser components
        return None, None, None

    # Events ===============================
    # [Eraser TAB Actions]
    eraser_input_img.select(get_select_coords, eraser_input_img, [eraser_input_img, eraser_original_img, eraser_mask])
    eraser_button.click(rm_object, [eraser_original_img, eraser_mask], eraser_output_img)
    eraser_input_img.clear(reset_images, None, [eraser_original_img, eraser_mask, eraser_output_img])

if __name__ == "__main__":
    demo.launch(share=True)