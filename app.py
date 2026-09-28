from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelBinarizer
import requests

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = "food_classifier_mobilenetv2.h5"
model = load_model(MODEL_PATH)

# Define categories (ensure it matches the order used during training)
CATEGORIES = ['waffles', 'pancakes', 'cup_cakes', 'pizza', 'donuts', 'apple_pie', 'baklava', 'ceviche',
              'fried_rice', 'ice_cream', 'macarons', 'mussels', 'omelette', 'dumplings', 'edamame', 'falafel']

label_binarizer = LabelBinarizer()
label_binarizer.fit(CATEGORIES)

# Edamam API credentials
EDAMAM_APP_ID = "33c76272"  # Replace with your Edamam APP ID
EDAMAM_APP_KEY = "d7efcb03a5b460bad5bc7667b864681c"  # Replace with your Edamam APP KEY

# Predict the food item from an uploaded image
def predict_food_item(image_path):
    IMAGE_SIZE = (128, 128)
    image = load_img(image_path, target_size=IMAGE_SIZE)
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions)
    predicted_class = label_binarizer.classes_[predicted_class_index]
    return predicted_class

# Fetch nutritional data using Edamam API
def get_nutritional_data(food_name):
    if len(food_name.split()) == 1:
        food_name = f"1 serving {food_name}"

    url = "https://api.edamam.com/api/nutrition-data"
    params = {
        "app_id": EDAMAM_APP_ID,
        "app_key": EDAMAM_APP_KEY,
        "ingr": food_name
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        calories = data.get("calories", 0)
        protein = data.get("totalNutrients", {}).get("PROCNT", {}).get("quantity", 0)
        fat = data.get("totalNutrients", {}).get("FAT", {}).get("quantity", 0)
        carbohydrates = data.get("totalNutrients", {}).get("CHOCDF", {}).get("quantity", 0)

        return {
            "calories": calories,
            "protein": protein,
            "fat": fat,
            "carbohydrates": carbohydrates
        }
    else:
        return None

# Fetch tips and warnings based on food item and calories
def get_tips_and_warnings(food_item, calories):
    tips = {
        "waffles": [
            "Use whole-grain batter for more fiber.",
            "Top with fresh fruits instead of syrup.",
            "Reduce portion size to avoid excess calories."
        ],
        "pancakes": [
            "Try using oat flour or almond flour.",
            "Add protein-rich toppings like yogurt or nuts.",
            "Limit sugary syrups or powdered sugar."
        ],
        "cup_cakes": [
            "Opt for smaller-sized cupcakes to control portions.",
            "Use healthier substitutes like applesauce for oil.",
            "Avoid heavy frosting or use whipped cream instead."
        ],
        "pizza": [
            "Choose thin crust or whole-grain options.",
            "Add lots of veggies for extra nutrients.",
            "Go easy on high-fat toppings like cheese and pepperoni."
        ],
        "donuts": [
            "Limit to one serving as they are high in sugar.",
            "Pair with a protein source to stabilize blood sugar.",
            "Opt for baked donuts instead of fried ones."
        ],
        "apple_pie": [
            "Use less sugar in the filling or opt for natural sweeteners like honey.",
            "Try a whole-grain crust for added fiber.",
            "Control portion size to avoid excess calorie intake."
        ],
        "baklava": [
            "Limit serving size due to high sugar and fat content.",
            "Opt for a smaller slice or share with others.",
            "Balance with a lighter meal afterward."
        ],
        "ceviche": [
            "Use fresh lime juice and a variety of vegetables for added flavor.",
            "Pair with whole-grain crackers or a light salad.",
            "Ensure seafood is fresh to avoid contamination."
        ],
        "fried_rice": [
            "Use brown rice for added fiber and nutrients.",
            "Add lean protein like chicken, tofu, or shrimp.",
            "Limit oil and soy sauce to reduce calories and sodium."
        ],
        "ice_cream": [
            "Choose sorbet or frozen yogurt as a lower-calorie option.",
            "Control portion size to half a cup.",
            "Top with fresh fruits instead of chocolate syrup."
        ],
        "macarons": [
            "Limit serving size to avoid excess sugar intake.",
            "Pair with unsweetened tea or water to balance the sweetness.",
            "Avoid artificial coloring or flavors for healthier options."
        ],
        "mussels": [
            "Steam or grill instead of frying.",
            "Add lemon and herbs for flavor instead of butter.",
            "Pair with whole-grain bread or a light salad."
        ],
        "omelette": [
            "Use egg whites to reduce fat and cholesterol.",
            "Add veggies like spinach, tomatoes, or mushrooms for nutrients.",
            "Limit cheese or use low-fat alternatives."
        ],
        "dumplings": [
            "Choose steamed over fried dumplings.",
            "Add more vegetables to the filling for fiber.",
            "Limit the amount of soy sauce to control sodium intake."
        ],
        "edamame": [
            "Sprinkle with minimal salt or seasoning.",
            "Enjoy as a high-protein snack in moderation.",
            "Avoid processed or pre-packaged versions with added flavors."
        ],
        "falafel": [
            "Bake instead of deep-frying for fewer calories.",
            "Serve with hummus or tzatziki instead of heavy sauces.",
            "Pair with a salad or whole-grain pita bread for a balanced meal."
        ]
    }

    
    warnings = []
    if calories > 200:
        warnings.append("Warning: This food item is high in calories. Consider sharing or saving some for later.")
    
    return tips.get(food_item, []), warnings

# Route to serve static image files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

# Route to render the upload page
@app.route('/')
def index():
    return render_template('predict.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    # Save the uploaded image
    img_path = os.path.join("uploads", file.filename)
    file.save(img_path)

    # Predict the food item
    predicted_food = predict_food_item(img_path)

    # Fetch nutritional data
    nutritional_data = get_nutritional_data(predicted_food)
    
    if nutritional_data:
        tips, warnings = get_tips_and_warnings(predicted_food, nutritional_data['calories'])
        result = {
            "food": predicted_food,
            "calories": nutritional_data['calories'],
            "protein": nutritional_data['protein'],
            "fat": nutritional_data['fat'],
            "carbohydrates": nutritional_data['carbohydrates'],
            "food_image": f"/uploads/{file.filename}",
            "tips": tips,
            "warnings": warnings
        }
        return jsonify(result)
    else:
        return jsonify({"error": "Failed to retrieve nutritional data."})

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
