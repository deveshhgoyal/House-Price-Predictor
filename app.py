from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("county_encoder.pkl", "rb"))
counties = list(encoder.classes_)

@app.route("/")
def home():
    return render_template("index.html", counties=counties)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form

    try:
        MedInc = float(data["MedInc"])
        HouseAge = float(data["HouseAge"])
        AveRooms = float(data["AveRooms"])
        AveBedrms = float(data["AveBedrms"])
        Population = float(data["Population"])
        AveOccup = float(data["AveOccup"])

        # ✅ Validation check (no zero or negative values)
        if any(val <= 0 for val in [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup]):
            raise ValueError("All inputs must be positive values.")

        County = encoder.transform([data["County"]])[0]

        x = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                       Population, AveOccup, County]])

        prediction = model.predict(x)[0]
        price = prediction * 100000
        price_k = round(price / 1000, 2)
        final_output = f"{price_k} K"

        return render_template("index.html", counties=counties, result=final_output)

    except ValueError as e:
        # ⚠ Error handling – show message on page
        return render_template("index.html", counties=counties, result=f"⚠ Error: {str(e)}")



if __name__ == "__main__":
    app.run(debug=True)

# inp_1 -> 8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556