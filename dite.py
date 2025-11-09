from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # allow React frontend

# Load model + encoders only once (faster)
model = joblib.load("diet_model.pkl")
encoders = joblib.load("encoders.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        cuisine = data.get("Cuisine_top1")
        spice = int(data.get("Spice_Tolerance"))
        budget = float(data.get("Food_Budget"))
        sweet = int(data.get("Sweet_Tooth_Level"))
        eating_out = int(data.get("Eating_Out"))

        # Validate cuisine
        if cuisine not in encoders["Cuisine_top1"].classes_:
            return jsonify({"prediction": "Unknown cuisine. Please enter a valid cuisine."})

        # Encode cuisine
        cuisine_encoded = encoders["Cuisine_top1"].transform([cuisine])[0]

        # Prepare input dataframe
        input_data = pd.DataFrame({
            "Cuisine_top1": [cuisine_encoded],
            "Spice Tolerance": [spice],
            "Food_Budget": [budget],
            "Sweet_Tooth_Level": [sweet],
            "Eating Out Per week": [eating_out]
        })

        # Predict
        pred = model.predict(input_data)[0]
        decoded_pred = encoders["Dietary Preference"].inverse_transform([pred])[0]

        return jsonify({"prediction": decoded_pred})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)




