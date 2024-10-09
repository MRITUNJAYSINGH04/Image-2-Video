import spaces
import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
import gradio as gr

pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        "THUDM/CogVideoX-5b-I2V",
        torch_dtype=torch.bfloat16
    )
    
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

@spaces.GPU(duration=250)
def generate_video(prompt, image):
    video = pipe(
        prompt=prompt,
        image=image,
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=49,
        guidance_scale=6,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).frames[0]
    
    video_path = "output.mp4"
    export_to_video(video, video_path, fps=8)
    
    return video_path

# Interface Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Image to Video Generation")
    
    with gr.Row():
        # Entrada de texto para o prompt
        prompt_input = gr.Textbox(label="Prompt", value="A little girl is riding a bicycle at high speed. Focused, detailed, realistic.")
        
        # Upload de imagem
        image_input = gr.Image(label="Upload an Image", type="pil")
    
    # Botão para gerar o vídeo
    generate_button = gr.Button("Generate Video")
    
    # Saída do vídeo gerado
    video_output = gr.Video(label="Generated Video")
    
    # Ação ao clicar no botão
    generate_button.click(fn=generate_video, inputs=[prompt_input, image_input], outputs=video_output)

# Rodar a interface
demo.launch()
