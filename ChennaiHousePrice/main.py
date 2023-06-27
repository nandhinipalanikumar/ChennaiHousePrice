from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import sklearn

app = Flask(__name__)
data= pd.read_csv('HousingPredic_Clean.csv')
reg_new = pickle.load(open("Chennai_HousePrice_LiRegModel.pickle", 'rb'))

@app.route('/')
def index():
    area = sorted(data['AREA'].unique())
    zone = sorted(data['MZZONE'].unique())
    buildtype = sorted(data['BUILDTYPE'].unique())
    street = sorted(data['STREET'].unique())
    parking = sorted(data['PARK_FACIL'].unique())
    return render_template('index.html', area=area, zone=zone,  buildtype= buildtype,street= street,parking= parking)







@app.route('/predict', methods=['GET', 'POST'])
def predict():
    AREA = float(request.form['AREA'])
    N_BEDROOM = float(request.form['N_BEDROOM'])
    N_BATHROOM = float(request.form['N_BATHROOM'])
    INT_SQFT = float(request.form['INT_SQFT'])
    PROPERTY_AGE = float(request.form['PROPERTY_AGE'])
    DIST_MAINROAD = float(request.form['DIST_MAINROAD'])
    SALE_COND = float(request.form['SALE_COND'])
    PARK_FACIL = float(request.form['PARK_FACIL'])
    MZZONE = float(request.form['MZZONE'])
    STREET = float(request.form['STREET'])
    BUILDTYPE = float(request.form['BUILDTYPE'])
    N_ROOM = float(request.form['N_ROOM'])


    print(AREA, N_BEDROOM, N_BATHROOM, INT_SQFT, PROPERTY_AGE, DIST_MAINROAD, SALE_COND, PARK_FACIL, MZZONE, STREET, BUILDTYPE, N_ROOM)
    input = pd.DataFrame([[AREA, N_BEDROOM, N_BATHROOM, INT_SQFT, PROPERTY_AGE, DIST_MAINROAD, SALE_COND, PARK_FACIL, MZZONE, STREET, BUILDTYPE, N_ROOM]], columns = ['AREA', 'N_BEDROOM', 'N_BATHROOM', 'INT_SQFT', 'PROPERTY_AGE', 'DIST_MAINROAD', 'SALE_COND', 'PARK_FACIL', 'MZZONE', 'STREET', 'BUILDTYPE', 'N_ROOM'])
    scaler = StandardScaler()
    scaler.fit(input)
    val = scaler.transform(input)
    Predi = reg_new.predict(val)[0]
    Predic = np.exp(Predi)

    return str(Predic)
if __name__ == "__main__":
    app.run(debug=True, port=5001)