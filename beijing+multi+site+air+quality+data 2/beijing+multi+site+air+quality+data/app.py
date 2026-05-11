from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)


# LOAD MODEL AND SCALER


model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')



# HOME PAGE


@app.route('/')
def home():
    return render_template('index.html')



# PREDICTION


@app.route('/predict', methods=['POST'])
def predict():

    try:

        temp = float(request.form['TEMP'])
        pres = float(request.form['PRES'])
        dewp = float(request.form['DEWP'])
        rain = float(request.form['RAIN'])
        wspm = float(request.form['WSPM'])

        features = np.array([
            [temp, pres, dewp, rain, wspm]
        ])

        scaled_features = scaler.transform(features)

        prediction = model.predict(scaled_features)

        output = round(prediction[0], 2)

        return render_template(
            'index.html',
            prediction_text=f'Predicted PM2.5 Level: {output}'
        )

    except Exception as e:

        return render_template(
            'index.html',
            prediction_text=f'Error: {str(e)}'
        )



# RUN APPLICATION


if __name__ == '__main__':
    app.run(debug=True)
