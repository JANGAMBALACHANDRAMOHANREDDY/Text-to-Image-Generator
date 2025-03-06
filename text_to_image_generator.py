import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

def generate_image(prompt, output_file='generated_image.png'):
    # Load Stable Diffusion model
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float16)
    pipe = pipe.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate image
    image = pipe(prompt).images[0]
    
    # Save the image
    image.save(output_file)
    print(f"Image saved as {output_file}")
    return output_file

# Example usage
if __name__ == '__main__':
    user_input = input("Enter a text prompt for image generation: ")
    generate_image(user_input)
