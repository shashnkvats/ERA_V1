# CLIPMatch
<p><bold>CLIPMatch</bold> is a gradio-based application that facilitates the exploration of specific visual content within a video by describing it in words. It employs the capabilities of OpenAI's CLIP model to analyze the similarity between video content and textual descriptions. Users can upload a video file and enter a text query, which then generates a similarity graph over time, making it possible to pinpoint the segments where the described content is most likely to appear.</p>

[![Watch the video](https://img.youtube.com/vi/r_7WYsTZWPA/0.jpg)](https://youtu.be/r_7WYsTZWPA)

## Features
<ul>
  <li><b>Video Upload:</b> Upload a video file to be analyzed.</li>
  <li><b>Text Query:</b> Enter a text description to search for specific visual content within the video.</li>
  <li><b>Similarity Graph:</b> Generate a similarity graph showing the correlation between the video and text over time, identifying the instances where the described content is visually represented.</li>
  <li><b>Closest Match Identification:</b> The graph highlights the point of highest similarity between the video and text, aiding in locating the described visual content.
</li>
</ul>

## Installation
<p>Ensure you have the necessary libraries installed using the following:</p>

`pip install -r requirements.txt`

## Usage
<ul>
  <li>Clone the repository to your local machine.</li>
  <li>Navigate to the project directory in the terminal.</li>
  <li>Run the following command to launch the Gradio interface:</li>
</ul>

`python main.py`

<ul>
  <li>Open the Gradio interface in your web browser (the URL will be displayed in the terminal).</li>
  <li>Upload a video file and enter a text query to search for specific visual content within the video.</li>
  <li>View the similarity graph and analyze the results to find the closest match for your query.</li>
</ul>

## Dependencies
<ul>
  <ol>Gradio</ol>
  <ol>Matplotlib</ol>
  <ol>NumPy</ol>
  <ol>OpenCV</ol>
  <ol>Torch</ol>
  <ol>Torchvision</ol>
  <ol>Clip</ol>
  <ol>Matplotlib</ol>
  <ol>PIL (Pillow)</ol>
  <ol>tqdm</ol>
</ul>






