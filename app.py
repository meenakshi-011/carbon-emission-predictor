from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# load model and columns
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('columns.pkl', 'rb') as f:
    columns = pickle.load(f)

countries = ['India', 'United States', 'China', 'Germany', 'Brazil']

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None

    if request.method == 'POST':
        year = int(request.form['year'])
        country = request.form['country']

        input_df = pd.DataFrame(
            np.zeros((1, len(columns))),
            columns=columns
        )

        input_df['Year'] = year

        country_col = f'Country_{country}'
        if country_col in input_df.columns:
            input_df[country_col] = 1

        prediction = model.predict(input_df)[0]

    # 🔹 Pass prediction_text for template
    return render_template(
        'index.html',
        prediction_text=f"Predicted CO₂ Emission: {prediction:,.2f}" if prediction is not None else None,
        countries=countries
    )

if __name__ == '__main__':
    app.run(debug=True)
