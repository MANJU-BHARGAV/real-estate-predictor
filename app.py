from flask import Flask, request, jsonify, render_template
import json
import pickle
import numpy as np

app = Flask(__name__)

# Load model and column data
with open("artifacts/banglore_home_price_prediction.pickle", "rb") as f:
    model = pickle.load(f)

with open("artifacts/columns.json", "r") as f:
    data_columns = json.load(f)['data_columns']
    locations = data_columns[3:]  # Assuming first 3 columns are bhk, sqft, bath

@app.route("/")
def home():
    return render_template("home.html", locations=locations)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        location = request.form.get("location")
        sqft = float(request.form.get("sqft"))
        bath = int(request.form.get("bath"))
        bhk = int(request.form.get("bhk"))

        # Create input array for model
        x = np.zeros(len(data_columns))
        x[0] = bhk
        x[1] = sqft
        x[2] = bath
        if location in locations:
            loc_index = data_columns.index(location)
            x[loc_index] = 1

        # Predict
        predicted_price = round(model.predict([x])[0], 2)

        return render_template("home.html", prediction=f"Estimated Price: ₹{predicted_price} Lakhs", locations=locations)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

