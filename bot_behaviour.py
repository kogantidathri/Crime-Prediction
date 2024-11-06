from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model_final.pkl', 'rb') as file:
    model = pickle.load(file)

# Define routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/inspect")
def inspect():
    return render_template("inspect.html")

@app.route("/output", methods=["GET", "POST"])
def output():
    if request.method == 'POST':
        # Retrieve input data from the form
        var1 = request.form["STATE/UT"]
        var2 = float(request.form["YEAR"])  # Ensure year is converted to float
        var3 = float(request.form["MURDER"])  # Convert all numerical inputs to float
        var4 = float(request.form["RAPE"])
        var5 = float(request.form["ROBBERY"])
        var6 = float(request.form["THEFT"])
        var7 = float(request.form["AUTO THEFT"])
        var8 = float(request.form["RIOTS"])
        var9 = float(request.form["CRIMINAL BREACH OF TRUST"])
        var10 = float(request.form["COUNTERFIETING"])
        var11 = float(request.form["DOWRY DEATHS"])
        var12 = float(request.form["CRUELTY BY HUSBAND OR HIS RELATIVES"])
        var13 = float(request.form["CAUSING DEATH BY NEGLIGENCE"])

        # Convert the input data into a numpy array
        predict_data = np.array([var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13]).reshape(1, -1)

        # Use the loaded model to make predictions
        prediction = model.predict(predict_data)

        # Render the output template with the regression prediction
        return render_template('output.html', prediction=prediction[0])

    return render_template("output.html")

if __name__ == "__main__":
    app.run(debug=False)
