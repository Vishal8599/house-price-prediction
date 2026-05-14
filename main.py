import pandas as pd
import pickle

from flask import Flask, render_template, request

app = Flask(__name__)

# Load data
data = pd.read_csv('cleaned_data.csv')

# Load trained model
pipe = pickle.load(open('ridgeModel.pkl', 'rb'))


@app.route('/')
def index():

    locations = sorted(data['location'].unique())

    return render_template(
        'index.html',
        locations=locations
    )


@app.route('/predict', methods=['POST'])
def predict():

    location = request.form.get('location')
    bhk = int(request.form.get('bhk'))
    bath = int(request.form.get('bath'))
    sqft = float(request.form.get('sqft'))

    input_data = pd.DataFrame(
        [[location, sqft, bath, bhk]],
        columns=['location', 'total_sqft', 'bath', 'bhk']
    )

    prediction = pipe.predict(input_data)[0]

    return str(round(prediction, 2))


if __name__ == '__main__':
    app.run(debug=True)