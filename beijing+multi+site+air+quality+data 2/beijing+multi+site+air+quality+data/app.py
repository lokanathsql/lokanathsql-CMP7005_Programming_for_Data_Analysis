from flask import Flask, render_template, request
import numpy as np
import joblib
from datetime import datetime

app = Flask(__name__)


# LOAD MODEL AND SCALER


model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')


# AQI CATEGORY FUNCTION


def get_aqi_category(pm25):

    if pm25 <= 50:
        return {
            "category": "Good",
            "color": "#00e400",
            "health": "Air quality is satisfactory and safe.",
            "icon": "😊"
        }

    elif pm25 <= 100:
        return {
            "category": "Moderate",
            "color": "#ffcc00",
            "health": "Sensitive people should reduce prolonged outdoor exposure.",
            "icon": "😐"
        }

    elif pm25 <= 150:
        return {
            "category": "Unhealthy",
            "color": "#ff7e00",
            "health": "People may begin experiencing breathing discomfort.",
            "icon": "⚠️"
        }

    else:
        return {
            "category": "Hazardous",
            "color": "#ff0000",
            "health": "Avoid outdoor activities and wear protective masks.",
            "icon": "☠️"
        }


# ENVIRONMENTAL INSIGHTS


def generate_environmental_insights(
    temp,
    rain,
    wspm,
    pres,
    dewp,
    pm25
):

    insights = []

    if temp > 30:
        insights.append(
            "High temperature conditions may increase ozone formation."
        )

    if rain > 0:
        insights.append(
            "Rainfall can help reduce airborne particulate matter."
        )

    if wspm > 5:
        insights.append(
            "Strong wind speed may help disperse pollutants."
        )

    if pres > 1020:
        insights.append(
            "High atmospheric pressure may trap pollutants near ground level."
        )

    if dewp > 20:
        insights.append(
            "High dew point indicates humid environmental conditions."
        )

    if pm25 > 100:
        insights.append(
            "Air pollution levels are elevated and may affect respiratory health."
        )

    if len(insights) == 0:
        insights.append(
            "Environmental conditions appear stable."
        )

    return insights


# HOME PAGE


@app.route('/')
def home():

    return render_template(
        'index.html',
        current_year=datetime.now().year
    )


# PREDICTION ROUTE


@app.route('/predict', methods=['POST'])
def predict():

    try:


        # GET INPUT VALUES


        temp = float(request.form['TEMP'])
        pres = float(request.form['PRES'])
        dewp = float(request.form['DEWP'])
        rain = float(request.form['RAIN'])
        wspm = float(request.form['WSPM'])


        # FEATURE ARRAY


        features = np.array([
            [temp, pres, dewp, rain, wspm]
        ])


        # SCALE FEATURES


        scaled_features = scaler.transform(features)


        # MODEL PREDICTION


        prediction = model.predict(scaled_features)

        pm25 = round(prediction[0], 2)


        # SIMULATED ENVIRONMENTAL METRICS


        pm10 = round(pm25 * 1.25, 2)
        no2 = round((pres - 980) * 1.8, 2)
        co = round((pm25 / 50), 2)
        o3 = round((temp * 2.1), 2)
        so2 = round((pm25 / 8), 2)


        # AQI INFORMATION


        aqi_info = get_aqi_category(pm25)


        # ENVIRONMENTAL INSIGHTS


        insights = generate_environmental_insights(
            temp,
            rain,
            wspm,
            pres,
            dewp,
            pm25
        )


        # MODEL PERFORMANCE DATA


        model_results = {
            "Linear Regression": {
                "RMSE": 70.77,
                "R2": 0.15
            },
            "Decision Tree": {
                "RMSE": 78.04,
                "R2": -0.02
            },
            "Random Forest": {
                "RMSE": 57.18,
                "R2": 0.44
            }
        }


        # FEATURE IMPORTANCE DATA


        feature_importance = {
            "TEMP": 0.302,
            "DEWP": 0.297,
            "PRES": 0.238,
            "WSPM": 0.145,
            "RAIN": 0.016
        }

        # RENDER DASHBOARD

        return render_template(

            'index.html',

            # MAIN PREDICTION
            prediction=pm25,

            # AQI INFO
            category=aqi_info["category"],
            color=aqi_info["color"],
            health=aqi_info["health"],
            icon=aqi_info["icon"],

            # MULTI POLLUTANT DATA
            pm25=pm25,
            pm10=pm10,
            no2=no2,
            co=co,
            o3=o3,
            so2=so2,

            # WEATHER DATA
            temp=temp,
            pres=pres,
            dewp=dewp,
            rain=rain,
            wspm=wspm,

            # AI INSIGHTS
            insights=insights,

            # MODEL METRICS
            model_results=model_results,

            # FEATURE IMPORTANCE
            feature_importance=feature_importance,

            # FOOTER
            current_year=datetime.now().year
        )

    except Exception as e:

        return render_template(

            'index.html',

            error=f"Error occurred: {str(e)}",

            current_year=datetime.now().year
        )


# RUN APPLICATION


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
