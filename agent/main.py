from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
from PIL import Image
import numpy as np
import io

img_height = 256
img_width = 256

# Load the models
graphic_style_model = tf.keras.models.load_model('models/Graphic-Style-classifier.keras')
graphic_style_classes = ['Cartoon', 'Fantasy', 'Low-Poly', 'Pixel-Art', 'Realistic']

perspective_model = tf.keras.models.load_model('models/Perspective-classifier.keras')
perspective_classes = ['First-Person', 'Isometric', 'Third-Person', 'Top-Down']

game_model = tf.keras.models.load_model('models/Game-classifier.keras')
game_classes = ['BattleBit', 'CounterStrike', 'DarkSouls', 'EuropaUniversalis', 'Fortnite', 'GrandTheftAuto', 'Grayzone', 'Hades', 'HeartsOfIron', 'LeagueOfLegends', 'LethalCompany', 'Minecraft', 'Overwatch', 'RuneScape', 'Rust', 'SeaOfThieves', 'Terraria', 'TotalWar', 'VRising', 'Warno']

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

        graphic_predictions = graphic_style_model.predict(img_array)
        graphic_score = tf.nn.softmax(graphic_predictions[0])
        graphic_result = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(graphic_style_classes[np.argmax(graphic_score)], 100 * np.max(graphic_score))
        
        game_predictions = game_model.predict(img_array)
        game_score = tf.nn.softmax(game_predictions[0])
        game_result = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(game_classes[np.argmax(game_score)], 100 * np.max(game_score))
        
        perspective_predictions = perspective_model.predict(img_array)
        perspective_score = tf.nn.softmax(perspective_predictions[0])
        perspective_result = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(perspective_classes[np.argmax(perspective_score)], 100 * np.max(perspective_score))
        
        return render_template('prediction.html', graphic_prediction=graphic_result, game_prediction=game_result, perspective_prediction=perspective_result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)