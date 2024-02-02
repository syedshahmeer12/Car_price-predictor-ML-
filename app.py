
from flask import Flask, render_template , request
import pandas as pd
import pickle
import  numpy as np


app = Flask(__name__)
car = pd.read_csv('Cleaned_Car_data.csv')
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    car_models = sorted(car['name'].unique())
    companies = sorted(car['company'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()  # Fix the typo here
    return render_template('web.html', companies=companies, years=year, car_models=car_models, fuel_types=fuel_type)

@app.route('/predict' , methods = ['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kilo_driven'))
    print(company , car_model , fuel_type , year , kms_driven)
    # prediction = model.predict(pd.DataFrame([[car_model , company, year , kms_driven , fuel_type]] , columns=['name' , 'company' , 'year' , 'kms_driven , fuel_type']))
    prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'], data=np.array([car_model, company, year, kms_driven, fuel_type]).reshape(1, 5)))
    print(prediction)
    return str(np.round(prediction[0],2))
# if __name__ == "__main__":
#     app.run(debug=True)
