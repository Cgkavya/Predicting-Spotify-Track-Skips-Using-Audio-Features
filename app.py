import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load saved model, encoder, scaler, and feature names
model = pickle.load(open("best_model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_names = pickle.load(open("model_features.pkl", "rb"))


@app.route("/")
def welcome():
    return render_template("welcome.html")


@app.route("/home")
def home():
    genres = encoder.classes_.tolist()
    return render_template("home.html", genres=genres)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract user inputs
        user_input = {
            "danceability": float(request.form["danceability"]),
            "energy": float(request.form["energy"]),
            "speechiness": float(request.form["speechiness"]),
            "valence": float(request.form["valence"]),
            "instrumentalness": float(request.form["instrumentalness"]),
            "duration_ms": int(request.form["duration_ms"]),
            "popularity": int(request.form["popularity"]),
            "genre": request.form["genre"],
        }

        # Encode genre
        genre_encoded = encoder.transform([user_input["genre"]])[0]
        user_input["genre"] = genre_encoded

        # Arrange features in correct order
        input_data = [user_input[feat] for feat in feature_names]
        input_array = np.array([input_data])

        # Scale the input data
        input_array = scaler.transform(input_array)

        # Predict using probability and apply threshold
        proba = model.predict_proba(input_array)[0]
        prediction = 1 if proba[1] > 0.3 else 0

        print("Probabilities:", proba)
        print("Prediction:", prediction)

        result = (
            "🎧 The user is likely to SKIP this song."
            if prediction == 1
            else "🎵 The user is likely to LISTEN to this song."
        )

        return render_template("result.html", prediction_text=result)

    except Exception as e:
        return render_template("result.html", prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
