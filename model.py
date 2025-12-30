import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def predict_stock_price(data):
    data = data.reset_index()
    data['Days'] = np.arange(len(data))

    X = data[['Days']]
    y = data['Close']

    model = LinearRegression()
    model.fit(X, y)

    future_day = [[len(data) + 1]]
    prediction = model.predict(future_day)

    return round(prediction[0], 2)
