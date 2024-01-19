## Creative Content Assisted by Generative AI using Amazon SageMaker: Inpainting Fill and Outpainting (TBD)

---

In this notebook, we demonstrate another approach to modify images called inpainting fill. Inpainting fill is the process of replacing a portion of an image with synthesized content based on a textual prompt. We will accomplish this using generative AI models from Amazon Bedrock.


![magic_fill_replace](https://raw.github.com/geekyutao/Inpaint-Anything/main/example/MainFramework.png)

The workflow is to provide the model with three inputs:

A mask image that outlines the portion to be replaced
A textual prompt describing the desired contents
The original image
Bedrock can then produce a new image that replaces the masked area with the object, subject, or environment described in the prompt.

You can use the mask image provided in the data/mask.png file.

Note: The mask image must have the same resolution and aspect ratio as the image being inpainted upon.