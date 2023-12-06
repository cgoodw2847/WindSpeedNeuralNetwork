import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error
#import os

# Load data and parse date
df = pd.read_csv("C:/Users/camer/OneDrive/Desktop/Python/Neural Network Scripts/silon.csv", parse_dates=["Date and hour"])

# Format Data
dataset = df
dataset["Month"] = df["Date and hour"].dt.month
dataset["Day"] = df["Date and hour"].dt.day
dataset["Year"] = df["Date and hour"].dt.year
dataset["Date"] = df["Date and hour"].dt.date
dataset["Time"] = df["Date and hour"].dt.time
dataset["Day of Week"] = df["Date and hour"].dt.day_name()
dataset = df.set_index("Date and hour")
dataset.index = pd.to_datetime(dataset.index)

dates= [6671,	7031,
15431,	15791,
24215,	24575,
32975,	33335,
41831,	41855]

y2 = df.iloc[dates[8]:dates[9], 1]

#%% Upload to MQTT
import time
from Adafruit_IO import Client, Feed, RequestError

run_count = y2

# Set to your Adafruit IO key.
ADAFRUIT_IO_KEY = "aio_QXAX87ERI7xOfUubqzcBm03ZFdp0"

# Set to your Adafruit IO username.
ADAFRUIT_IO_USERNAME = "ckg33"

# Create an instance of the REST client.
aio = Client(ADAFRUIT_IO_USERNAME, ADAFRUIT_IO_KEY)

try:
    foo = aio.feeds('foo1')
except RequestError: # Doesn't exist, create a new feed
    feed = Feed(name="foo1")
    foo = aio.create_feed(feed)

while True:
    a = dates[8]
    while a<dates[9]:
        a = a+1
        print('sending count: ', run_count[a])
        aio.send_data('foo1', run_count[a])
    
        time.sleep(3)
    
#%%
# Down-sampling and split into training and testing sets

train_data = list(range(dates[0], dates[1])) + list(range(dates[2], dates[3])) + list(range(dates[4], dates[5])) + list(range(dates[6], dates[7]))
test_data = list(range(dates[8], dates[9]))

X_Train = df.iloc[train_data, 1:2] # Selecting wind speed column for training
Y_Test = df.iloc[test_data, 1:2]  # Selecting wind speed column for testing

# Pre-Processing
sc = MinMaxScaler(feature_range=(0, 1))
X_Train_scaled = sc.fit_transform(X_Train)
Y_Test_scaled = sc.transform(Y_Test)

X_train = []
Y_train = []

for i in range(70, X_Train_scaled.shape[0]):
    X_train.append(X_Train_scaled[i-70:i, 0])
    Y_train.append(X_Train_scaled[i, 0])

X_train, Y_train = np.array(X_train), np.array(Y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#%% Neural Network Model Formatting

# MLP Model
regressor = Sequential()

# Layer 1 - Training Data
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Layer 2
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Layer 3
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Layer 4
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))
regressor.compile(optimizer='adam', loss='mean_squared_error')


#%% Training the Model

#Set number of Epochs - Training Data
regressor.fit(X_train, Y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# Testing the Data
inputs = pd.concat((dataset[["Avg Avg Wind Speed @ 80m [m/s]"]], Y_Test), axis=0)  # Convert NumPy array to DataFrame
inputs = inputs[len(inputs) - len(Y_Test) - 70:].values

# Reshape and Normalize
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)


X_test = []
for i in range(70, len(inputs)):
    X_test.append(inputs[i-70:i, 0]) 

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#%% Prediction

# Pass to Model
predicted_speed = regressor.predict(X_test)

# Inverse Transformation to get Values
predicted_speed = sc.inverse_transform(predicted_speed)

# Create a matrix with test data dates, real data, and predicted data
result_matrix = pd.DataFrame({
    "Date": df.loc[test_data, "Date and hour"],
    "Real Data": Y_Test.values.flatten(),
    "Predicted Data": predicted_speed.flatten()
})

print(result_matrix)

#%% Real vs. Predicted Comparison Plot

rec_data = list(range(dates[8] - 1, dates[9] - 1))
Rec_Test = df.iloc[rec_data, 1:2]

# Calculate Mean Squared Error
mse = mean_squared_error(Rec_Test, predicted_speed)
print(f"Mean Squared Error: {mse}")

# Comparison Plot
plt.figure(figsize=(9, 6))
plt.plot(df.loc[test_data, "Date and hour"], Rec_Test.values, color="green", label="Recorded Data")
plt.plot(df.loc[test_data, "Date and hour"], predicted_speed, color="red", label="Predicted Data")
plt.xlabel('Date and Hour')
plt.ylabel("Speed")
plt.title(f'October 10th - Comparison of Recorded and Predicted Wind Speed (MSE={mse:.4f})')
plt.legend()
plt.show()

#%% Predicted Data Distribution Plot

# Create histogram
plt.figure(figsize=(9, 6))
sns.histplot(predicted_speed, bins=range(0,18), kde=False, label="Predicted Data")

plt.title('October 10th - Distribution of Predicted Wind Speed')
plt.xlabel('Wind Speed')
plt.ylabel('Hours')
plt.legend()
plt.show()

# Forming a Density Matrix
hist_data, bin_edges = np.histogram(predicted_speed, bins=range(0,11))
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

density_matrix = pd.DataFrame({
    "Wind Speed": bin_centers, "Density": hist_data})

#%% Daily Power Output Calculation

# Power Curve for Wind Turbine
wind_speed = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
power_kWe = np.array([0, 0, 3.8, 10.2, 18.8, 33.2, 51.2, 69.2, 81.4, 87.3, 89.6, 90.0, 89.1, 87.4, 85.5, 83.3, 81.5, 79.6, 78.3, 76.9])

degree = 7
coefficients = np.polyfit(wind_speed, power_kWe, degree)
poly_function = np.poly1d(coefficients)
power_fit = poly_function(wind_speed)

# Plotting Power Curve w/ Fitted Polynomial
#plt.scatter(wind_speed, power_kWe, label='Original Data')
#plt.plot(wind_speed, power_fit, label=f'Polynomial Fit (Degree {degree})', color='red')
#plt.xlabel('Wind Speed')
#plt.ylabel('Power (kWh)')
#plt.legend()
#plt.show()

# Power Calculation
bin_wind_speeds = density_matrix.iloc[:,0]
power_generated = poly_function(bin_wind_speeds)
power_generated[power_generated < 0] = 0

bin_density = density_matrix.iloc[:,1]

daily_generation = np.dot(power_generated,bin_density)
print(f'Daily Power Generated: {daily_generation:.2f} kWh')
