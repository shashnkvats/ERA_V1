import gradio as gr
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import clip
import os
from tqdm import tqdm
from PIL import Image

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

state = {
    'video_embedding': None,
    'text_embedding': None,
    'similarity_graph': None,
    'last_video_path': None  # Add this line to store the last processed video file path
}


def process_video(video_file):
    video_file_path = os.path.abspath(video_file.name)
    state['last_video_path'] = video_file_path 

    cap = cv2.VideoCapture(video_file_path)

    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_file}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    image_vectors = torch.zeros((frame_count, 512), device=device)
    for i in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if ret:
            with torch.no_grad():
                image_vectors[i] = model.encode_image(
                    preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
                )
        else:
            print(f"Failed to read frame {i}")
            break

    state['video_embedding'] = image_vectors
    calculate_similarity()


def process_text(query_text):
    text_inputs = torch.cat([clip.tokenize([query_text]).to(device)])
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    state['text_embedding'] = text_features  #
    calculate_similarity()


def calculate_similarity(video_file=None, query_text=None):
    if video_file:
        video_file_path = os.path.abspath(video_file.name)
        # Only process the video if the file path has changed
        if video_file_path != state['last_video_path']:
            process_video(video_file)
    if query_text:
        process_text(query_text)

    image_vectors = state['video_embedding']
    text_features = state['text_embedding']
    if image_vectors is None or text_features is None:
        return "Please provide both video and text input"  # or return an error image

    image_vectors /= torch.norm(image_vectors, dim=1, keepdim=True)
    similarities = (image_vectors @ text_features.T).squeeze(1)
    closest_idx = similarities.argmax().item()

    frame_count = image_vectors.shape[0]
    fps = state.get('fps', 30) 
    time_in_seconds = np.arange(frame_count) / fps
    similarity_scores = similarities.cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.plot(time_in_seconds, similarity_scores, label='Similarity Score', linestyle='-', color='blue')
    plt.axvline(x=closest_idx/fps, color='red', linestyle='--', label=f'Closest Match at {closest_idx/fps:.2f} seconds')
    plt.xticks(np.arange(0, time_in_seconds[-1] + 10, 10))
    plt.xlabel('Video Time (seconds)')
    plt.ylabel('Similarity Score')
    plt.legend(loc='upper right')
    plt.title('Similarity Score vs Video Time')
    plt.grid(True)

    plt.savefig("output_plot.png")  # Save the plot to a file
    plt.close()  # Close the plot to free up memory

    state['similarity_graph'] = "output_plot.png"  # Save graph to state
    return "output_plot.png", None

def get_similarity_graph():
    return state['similarity_graph']  # Return the saved graph

# Define Gradio interface
iface = gr.Interface(
    fn=calculate_similarity, 
    inputs=[gr.inputs.File(label="Upload a video"), gr.Textbox(label="Enter text")], 
    outputs=[gr.outputs.Image(type="filepath", label="Similarity Graph"), gr.outputs.Textbox(label="Error Message")]
)
iface.launch()