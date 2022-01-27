
import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
from datetime import date
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model
import sqlite3

# Stocks symbols
df = pd.read_excel('C:/Users/srika/Desktop/DS_Project_Team_61/FBMKLCI_STOCKS.xlsx', header=None)
stocks_symbols = np.squeeze(df[0]).tolist()
#df[1].values
# Index symbol
index_symbol = '^KLSE;1=9'

# Dates
start_date = '2017-11-30'
end_date = date.today()

# Download index data
data = pd.DataFrame()    # Empty dataframe
data['index'] = pdr.DataReader(index_symbol, 'yahoo', start_date, end_date)['Adj Close']
import time
i = 0
while i < len(stocks_symbols):
    print('Downloading.... ', i, stocks_symbols[i])
    try:
        # Extract the desired data from Yahoo!
        data[stocks_symbols[i]] = pdr.DataReader(stocks_symbols[i], 'yahoo', start_date, end_date)['Adj Close']
        i +=1      
    except:
        print ('Error with connection. Wait for 1 minute to try again...')
        # Wait for 30 seconds
        time.sleep(30)
        continue 



############################                  SQL connection with monthly data           ################################################ 
# Connect to the database
FBMKLCI = create_engine('sqlite:///fbmklci_data.db', echo=False)
data.to_sql('stocks', con=FBMKLCI,if_exists='replace')
# Data = pd.read_sql('select * from stocks',FBMKLCI )

con = sqlite3.connect('fbmklci_data.db')
c= con.cursor()
c.execute('SELECT * FROM stocks')
Data1 = c.fetchall()
Data = pd.DataFrame(Data1)
con.close()

#Data = Data.drop('5296.KL', axis =1 )
Data = Data.dropna()

stocks_data = Data.iloc[ : , 2:]
index_data = Data.iloc[ : , 1]
stocks_data.columns = stocks_symbols
assets_names = stocks_data.columns.values
stocks_data.columns


data_assets = stocks_data # Making Duplicated of the original data 
data_index = index_data # Making Duplicated of the original data 

print("Stocks data (time series) shape: {shape}".format(shape=stocks_data.shape))
print("Index data (time series) shape: {shape}".format(shape=index_data.shape))

# Split data
n_train = int(data_assets.shape[0]*0.8)
# Stocks data
X_train = data_assets.values[:n_train, :]
X_test = data_assets.values[n_train:, :]
# Index data
index_train = data_index[:n_train]
index_test = data_index[n_train:]
# Normalize data
scaler = MinMaxScaler([0, 1])
# Stocks data
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
# Index data
scaler_index = MinMaxScaler([0, 1])
index_train = scaler_index.fit_transform(index_train[:, np.newaxis])
index_test = scaler_index.fit_transform(index_test[:, np.newaxis])
## Autoencoder - Keras
# Network hyperparameters
n_inputs = X_train.shape[1]
# Training hyperparameters
epochs = 50
batch_size = 1
# Define model
input = Input(shape=(n_inputs,))
# Encoder Layers
encoded = Dense(8, input_shape=(n_inputs,), activation='relu')(input)
encoded = Dense(4, activation='relu')(encoded)
decoded = Dense(8, activation='relu')(encoded)
decoded = Dense(n_inputs, activation='sigmoid')(decoded)

# Encoder
encoder = Model(input, encoded)

# Autoencoder
model = Model(input, decoded)

# Compile autoencoder
model.compile(loss='mse', optimizer='adam')
model.summary()

# Fit the model
history = model.fit(X_train, X_train,epochs=epochs,batch_size=batch_size,shuffle=True,verbose=1)

# Visualize loss history
plt.figure()
plt.plot(history.history['loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# Evaluate model
score_train = model.evaluate(X_train, X_train, batch_size=batch_size)
score_test = model.evaluate(X_test, X_test, batch_size=batch_size)

print('Training MSE: %.8f' %score_train)
print('Testing MSE: %.8f' %score_test)

# Obtain reconstruction of the stocks
X_train_pred = model.predict(X_train)
X_test_pred = model.predict(X_test)

error = np.mean(np.abs(X_train - X_train_pred)**2, axis=0)
print('Training MSE: %.8f' %np.mean(error))

error_test = np.mean(np.abs(X_test - X_test_pred)**2, axis=0)
print('Testing MSE: %.8f' %np.mean(error_test))

# Sort stocks by reconstruction error
ind = np.argsort(error)
sort_error = error[ind]
sort_assets_names = assets_names[ind]


# Barplot
plt.figure()
plt.barh(2*np.arange(len(error[:30])), error[ind[:30]], tick_label=assets_names[ind[:30]])
plt.xlabel('MSE')
plt.show()

# Plot stock
i= 0
plt.figure()
plt.plot(X_train[:, ind[i]], label=assets_names[ind[i]] + ' Stock')
plt.plot(X_train_pred[:, ind[i]], label=assets_names[ind[i]] + ' AE')
plt.legend()
plt.xlabel('Time (days)')
plt.ylabel('Normalized price')
plt.show()




##################################################### Main ###############################
# Identify stocks
n = 5

portfolio_train = X_train_pred[:, ind[:n]]
portfolio_test = X_test_pred[:, ind[:n]]

# Create portfolio in-sample
tracked_index_insample = np.mean(portfolio_train, axis=1)

# Create portfolio out-sample
tracked_index_outofsample = np.mean(portfolio_test, axis=1)
# In-sample
plt.figure()

plt.plot(index_train, label='FBMKLCI Index')
plt.plot(tracked_index_insample, label='Tracked Index')
plt.legend()
plt.xlabel('Time (days)')
plt.ylabel('Normalized price')
plt.show()

# Correlation coefficient (in-sample)
corr_train = np.corrcoef(index_train.squeeze(), tracked_index_insample)[0, 1]
print('Correlation coefficient (in-sample): %.8f' %corr_train)

# Plot tracked index (out-of-sample)
plt.figure()
plt.plot(index_test, label='FBMKLCI Index')
plt.plot(tracked_index_outofsample, label='Tracked Index')
plt.legend()
plt.xlabel('Time (days)')
plt.ylabel('Normalized price')
plt.show()

# Correlation coefficient (out-of-sample)
corr_test = np.corrcoef(index_test.squeeze(), tracked_index_outofsample)[0, 1]
print('Correlation coefficient: %.8f' %corr_test)


# Correlation coefficient (in-sample): 0.91387505
# Correlation coefficient: 0.48544168
# Correlation coefficient (in-sample): 0.31845217
# Correlation coefficient: 0.73132484
# Correlation coefficient (in-sample): 0.13326888
# Correlation coefficient: 0.91990301
 
model.save('my_model.h5')
new_model = load_model('my_model.h5')
#/Output





'''
<div class="text">
    <p>{{text}}</p>
</div>
'''