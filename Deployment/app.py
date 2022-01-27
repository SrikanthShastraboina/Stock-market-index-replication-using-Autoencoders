from flask import Flask, render_template, request
import sqlite3
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
app = Flask(__name__)

model = load_model('my_model.h5')

@app.route('/')
def home():
     return render_template('Homepage.html')

@app.route('/input',methods=['POST'])
def predict():
    x = int(request.form['n_stocks'])
    con = sqlite3.connect('fbmklci_data.db')
    c= con.cursor()
    c.execute('SELECT * FROM stocks')
    Data1 = c.fetchall()
    Data = pd.DataFrame(Data1)
    con.close()
    
    stocks_data = Data.iloc[ : , 2:]
    index_data = Data.iloc[ : , 1]
    stocks_data.columns = ['3816.KL', '1155.KL', '6012.KL', '1082.KL', '6033.KL', '3034.KL',
           '5225.KL', '1023.KL', '1066.KL', '2445.KL', '4863.KL', '5819.KL',
           '3182.KL', '5183.KL', '4715.KL', '5347.KL', '4197.KL', '1295.KL',
           '6947.KL', '4707.KL', '5681.KL', '8869.KL', '1961.KL', '4065.KL',
           '5285.KL', '5168.KL', '6888.KL', '7277.KL', '7113.KL']
    assets_names = stocks_data.columns.values
    scaler = MinMaxScaler([0, 1])
    stocks_data = scaler.fit_transform(stocks_data)
    scaler_index = MinMaxScaler([0, 1])
    index_data = scaler_index.fit_transform(index_data[:, np.newaxis])

    pred = model.predict(stocks_data)
    
    error = np.mean(np.abs(stocks_data - pred)**2, axis=0)
    ind = np.argsort(error)
    portfolio = pred[:, ind[:x]]
    tracked_index = np.mean(portfolio, axis=1)
    corr = np.corrcoef(index_data.squeeze(), tracked_index)[0, 1]
    z = ['3816.KL', '1155.KL', '6012.KL', '1082.KL', '6033.KL', '3034.KL',
           '5225.KL', '1023.KL', '1066.KL', '2445.KL', '4863.KL', '5819.KL',
           '3182.KL', '5183.KL', '4715.KL', '5347.KL', '4197.KL', '1295.KL',
           '6947.KL', '4707.KL', '5681.KL', '8869.KL', '1961.KL', '4065.KL',
           '5285.KL', '5168.KL', '6888.KL', '7277.KL', '7113.KL']
    y = ['MISC', 'Malayan Banking Berhad', 'Maxis Berhad',
           'Hong Leong Financial Group Berhad', 'Petronas Gas Bhd',
           'Hap Seng Consolidated Berhad', 'IHH Healthcare Berhad',
           'CIMB Group Holdings Berhad', 'RHB Capital Berhad',
           'Kuala Lumpur Kepong Berhad', 'Telekom Malaysia Berhad',
           'Hong Leong Bank Berhad', 'Genting Berhad', 'PCHEM',
           'Genting Malaysia Berhad', 'Tenaga Nasional Berhad',
           'Sime Darby Berhad', 'Public Bank Berhad', 'DiGi.Com Berhad',
           'Nestlé (Malaysia) Berhad', 'Petronas Dagangan Bhd',
           'Press Metal Bhd', 'IOI Corp.Bhd', 'PPB Group Berhad', 'SIMEPLT',
           'Hartalega Holdings Berhad', 'Axiata Group Berhad',
           'Dialog Group Berhad', 'Top Glove Corporation Berhad']
    df = np.array(pd.DataFrame({'stock_symbol': z, 'stock_name':y}))
    output = pd.DataFrame(df[ind[:x], : ])
    output.columns = ['Stocks Symbol', 'Stocks Name']
    output.index +=1
    return render_template('Output.html' , output = [output.to_html(classes = "data")], header="true")



if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

