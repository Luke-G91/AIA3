from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# Load the model
model = tf.keras.models.load_model('models/game-classifier.keras')
img_height = 256
img_width = 256
class_names = ['Among Us', 'Apex Legends', 'Fortnite', 'Forza Horizon', 'Free Fire', 'Genshin Impact', 'God of War', 'Minecraft', 'Roblox', 'Terraria']

# Preprocess the image
def process_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((img_height, img_width))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

# Create a Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file'].read()
        image = Image.open(io.BytesIO(file))
        img = image.resize((img_height, img_width))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        result = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
        return render_template('prediction.html', prediction=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)