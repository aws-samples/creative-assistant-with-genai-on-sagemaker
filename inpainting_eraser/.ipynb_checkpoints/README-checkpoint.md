## Creative Content Assisted by Generative AI using Amazon SageMaker: Magic Eraser

---

## Introduction

we will demonstrate how to create a magic eraser that can remove any object from an image using Amazon SageMaker. This tool can come in handy when you need to remove photobombs, get rid of unwanted objects, or even clean up backgrounds. Especially for graphic designers and creative artists who are repeating these task on the daily bases. This can be a great efficiency booster to enhance their creative workflow.

Our solution for magic eraser involves two main steps. The first step is to create a segmentation mask of the object to be removed based on a pixel coordinate input. Then second step is to “erase”, which fills the area using the contexts from the rest of the image. 

To generate segmentation, we used a foundation model developed by Meta Research called [Segment Anything Model (SAM)](https://segment-anything.com/). This model is trained on a massive dataset called SA-1B with over 11 million images and 1.1 billion segmentation masks.  This massive scale gave Sam model unprecedented ability to identify and isolate objects from an image out of the box without training.

To erase the object, we used a second model called [Resolution-robust Large Mask Inpainting with Fourier Convolutions (LaMa)](https://advimman.github.io/lama-project/) developed by Roman Suvorov. This model can fill in missing parts of images caused by irregular masks.