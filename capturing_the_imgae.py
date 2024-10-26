# Import required libraries
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
import cv2
import numpy as np
from PIL import Image

# Function to capture an image from the webcam in Colab
def capture_image():
    display(Javascript('''
        async function captureImage() {
            const div = document.createElement('div');
            document.body.appendChild(div);
            const video = document.createElement('video');
            div.appendChild(video);

            // Prompt for webcam access
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            await video.play();

            // Create a canvas to capture the image
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            stream.getVideoTracks()[0].stop();
            div.remove();

            // Return the image as base64
            return canvas.toDataURL('image/png');
        }
        captureImage();
    '''))
    image_data_url = eval_js("captureImage()")
    image_data = b64decode(image_data_url.split(",")[1])
    with open("captured_image.png", "wb") as f:
        f.write(image_data)

    # Load image with OpenCV
    img = cv2.imread("captured_image.png")
    return img

# Capture the image
img = capture_image()

# Display the captured image
if img is not None:
    from matplotlib import pyplot as plt
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
else:
    print("No image captured.")
