from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# Constants
IMG_HEIGHT = 256
IMG_WIDTH = 256

# Load models and their classes
models_info = {
    "graphic_style": {
        "model": load_model('models/Graphic-Style-classifier-best-model.keras'),
        "classes": ['Cartoon', 'Fantasy', 'Low-Poly', 'Pixel-Art', 'Realistic'],
    },
    "perspective": {
        "model": load_model('models/Perspective-classifier-best-model.keras'),
        "classes": ['First-Person', 'Isometric', 'Third-Person', 'Top-Down'],
    },
    "game": {
        "model": load_model('models/Game-classifier-best-model.keras'),
        "classes": ['BattleBit', 'CounterStrike', 'DarkSouls', 'EuropaUniversalis', 'Fortnite', 'GrandTheftAuto', 'Grayzone', 'Hades', 'HeartsOfIron', 'LeagueOfLegends', 'LethalCompany', 'Minecraft', 'Overwatch', 'RuneScape', 'Rust', 'SeaOfThieves', 'Terraria', 'TotalWar', 'VRising', 'Warno'],
    },
}

# Image preprocessing
def process_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((IMG_HEIGHT, IMG_WIDTH))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

# Predict function
def predict_image(image, model_info):
    predictions = model_info["model"].predict(image)
    score = tf.nn.softmax(predictions[0])
    predicted_class = model_info["classes"][np.argmax(score)]
    confidence = 100 * np.max(score)
    return f"This image most likely belongs to {predicted_class} with a {confidence:.2f} percent confidence."

# Create Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part', 400
        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400
        try:
            image = Image.open(io.BytesIO(file.read()))
            img_array = process_image(image)

            # Predictions
            graphic_result = predict_image(img_array, models_info["graphic_style"])
            game_result = predict_image(img_array, models_info["game"])
            perspective_result = predict_image(img_array, models_info["perspective"])

            return render_template('prediction.html', graphic_prediction=graphic_result, game_prediction=game_result, perspective_prediction=perspective_result)

        except Exception as e:
            return str(e), 500

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
