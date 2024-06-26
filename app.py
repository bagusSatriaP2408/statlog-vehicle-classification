import pickle
from flask import Flask, request, render_template
import numpy as np

app = Flask(__name__)

# Load the SVM model and scaler
with open('svm_classifier.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        # Mengambil data input dari form
        input_data = [
            float(request.form['compactness']),
            float(request.form['circularity']),
            float(request.form['distance_circularity']),
            float(request.form['radius_ratio']),
            float(request.form['pr_axis_aspect_ratio']),
            float(request.form['max_length_aspect_ratio']),
            float(request.form['scatter_ratio']),
            float(request.form['elongatedness']),
            float(request.form['pr_axis_rectangularity']),
            float(request.form['max_length_rectangularity']),
            float(request.form['scaled_variance_along_major_axis']),
            float(request.form['scaled_variance_along_minor_axis']),
            float(request.form['scaled_radius_of_gyration']),
            float(request.form['skewness_about_major_axis']),
            float(request.form['skewness_about_minor_axis']),
            float(request.form['kurtosis_about_minor_axis']),
            float(request.form['kurtosis_about_major_axis']),
            float(request.form['hollows_ratio'])
        ]

        # Melakukan normalisasi input data
        input_data_normalized = scaler.transform([input_data])

        # Melakukan prediksi dengan model SVM
        prediction = svm_model.predict(input_data_normalized)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(port=5001)
