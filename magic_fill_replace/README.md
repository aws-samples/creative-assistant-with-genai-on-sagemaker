## Creative Content Assisted by Generative AI using Amazon SageMaker: Magic Fill/Replace
---

## Introduction

we will demonstrate how to create a magic fill/replace tool that can erase, fill, and replace background of any object from an image using Amazon SageMaker. The capability of this tool is illustrated in the digram below:

![magic_fill_replace](https://raw.github.com/geekyutao/Inpaint-Anything/main/example/MainFramework.png)

We will be using two models to build this solution. The first model is Segment Anything Model (SAM) - Apache-2.0 license from Meta Research. We will use this model to generate a segementation mask of the object. The second model is Stable Diffusion (SD) Inpaint - CreativeML Open RAIL++-M License from Stabilityai. This model allows us to generate new content in the masked area with a set of text prompts.