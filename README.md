# Creative Content Assisted by Generative AI using Amazon SageMaker

This GitHub repository contains various language assisted editing techniques by Generative AI using Amazon SageMaker.  These tools can improve efficiency for graphic designers and creative artists by automating time-consuming and repetitive tasks such as remove unwanted objects from image, replace objects or backgrounds, reshape and resize objects. With help from generative AI, these complex edits can be done quickly without spending hours on manual labor. This can free up their time and allow them to focus on more creative and strategic aspects of their work, ultimately leading to higher productivity and better results.

## Pre-requisites:

- An AWS account with access to Amazon SageMaker and S3.
- Basic knowledge of Python and Jupyter Notebook.
- Good understanding of SageMaker Multi-Model Endpoint w/ GPU
- Some understanding of Triton server for deep learning models
- A GPU instance to achieve better performance. `ml.g5.2xlarge` is the minimum size we recommend to host these models

Also, keep in mind that running the models may incur additional AWS charges, so make sure to check the pricing before running any code. Please remove the endpoint after you are done.

## [Inpainting Eraser](inpainting_eraser)

we will demonstrate how to create a inpainting eraser that can remove any object from an image using Amazon SageMaker. This tool can come in handy when you need to remove photobombs, get rid of unwanted objects, or even clean up backgrounds. 

## [Inpainting Fill and Outpanting](inpainting_fill_outpainting)

we will demonstrate how to create a inpainting fill tool that can erase, fill, and replace any object from an image as well as extending images around the edges using Amazon Bedrock. 


## License

The code in this repository is licensed under the MIT-0 License. Please refer to the LICENSE file for more information. 

## Credits

We thank Meta Research and Roman Suvorov for developing the foundation models used in this project. We also thank Amazon Web Services for providing the SageMaker platform that makes this work possible.