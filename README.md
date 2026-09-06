## Prototype Development for Image Generation Using the Stable Diffusion Model and Gradio Framework

### AIM:
To design and deploy a prototype application for image generation utilizing the Stable Diffusion model, integrated with the Gradio UI framework for interactive user engagement and evaluation.

### PROBLEM STATEMENT
To develop an interactive image-generation application that accepts a text prompt from the user and generates a corresponding image using the Stable Diffusion model. The application should provide a simple graphical interface using Gradio for easy interaction and experimentation.

### DESIGN STEPS

#### STEP 1: Install and Import Required Libraries
Install the required Python libraries such as Gradio, Diffusers, Transformers, PyTorch, and Pillow. Import the required modules and check whether CUDA/GPU is available.

#### STEP 2: Load the Stable Diffusion Model
Load the runwayml/stable-diffusion-v1-5 model using the Diffusers library. Configure the model to run on the available device, such as GPU or CPU. Define a function that accepts a text prompt and generates an image.

#### STEP 3: Develop the Gradio Interface
Create a Gradio interface containing a text box for entering the image-generation prompt and an image output component for displaying the generated image. Add example prompts and launch the application for interactive testing.

### PROGRAM:
```
# Install dependencies if required:
# !pip install -q gradio diffusers transformers accelerate safetensors pillow

import os
import gradio as gr
import torch

from PIL import Image
from diffusers import StableDiffusionPipeline


# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


# Load Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"

print("Loading Stable Diffusion model...")

pipe = StableDiffusionPipeline.from_pretrained(
    model_id
)

pipe = pipe.to(device)

print("Stable Diffusion model loaded successfully!")


# Image generation function
def generate(prompt):

    if not prompt or not prompt.strip():
        return None

    try:
        image = pipe(
            prompt,
            num_inference_steps=10
        ).images[0]

        return image

    except Exception as e:
        print("Error:", e)
        return None


# Create Gradio interface
gr.close_all()

demo = gr.Interface(
    fn=generate,

    inputs=gr.Textbox(
        label="Enter Your Prompt",
        placeholder="Describe the image you want to generate...",
        lines=3
    ),

    outputs=gr.Image(
        label="Generated Image"
    ),

    title="Image Generation Using Stable Diffusion",

    description=(
        "Enter a text prompt to generate an image "
        "using the Stable Diffusion model."
    ),

    examples=[
        ["A baby in a cradle"],
        ["A futuristic city at sunset"],
        ["A cute robot exploring space"],
        ["A peaceful mountain landscape"],
        ["A cyberpunk city at night"]
    ]
)

# Launch application
demo.launch(share=True)
```
### OUTPUT:

<img width="1054" height="532" alt="image" src="https://github.com/user-attachments/assets/2dee1e25-4401-423e-84ea-b9c1a4976c21" />

### RESULT:
Thus, a prototype image generation application using the Stable Diffusion model and Gradio framework was successfully developed and deployed. The application accepts text prompts from the user and generates corresponding images interactively through the Gradio interface.



