import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2

en_data = pd.read_csv("109_tmp_data_en.csv")
gm_data = pd.read_csv("109_tmp_data_gm.csv")

en_data.rename(columns={'year': 'Year', 'month': 'Month', 'day': 'Day', 'traffic_stream': 'Traffic_stream', 'hours': 'Hour'}, inplace=True)

def process_hours(hours):
    first_part = hours.split('_')[0].lstrip('0')
    return first_part

en_data["Hour"] = en_data["Hour"].astype(str)
en_data['Hour'] = en_data['Hour'].apply(process_hours)

en_data['Date'] = pd.to_datetime(en_data[['Year', 'Month', 'Day', 'Hour']], format='%Y-%m-%d %H')
gm_data['Date'] = pd.to_datetime(gm_data[['Year', 'Month', 'Day', 'Hour']], format='%Y-%m-%d %H')


en_data_sorted = en_data.sort_values(by='Date')
en_data_sorted = en_data_sorted[['klk_EN', 'Date', 'Traffic_stream']]

gm_data_sorted = gm_data.sort_values(by='Date')
gm_data_sorted = gm_data_sorted[['total_count', 'Date', 'Traffic_stream']]

merged_data = pd.merge(en_data_sorted, gm_data_sorted, on=['Date', 'Traffic_stream'], how='inner')

merged_data_true = merged_data[merged_data['Traffic_stream'] == 1]

training_data = merged_data_true[['klk_EN', 'total_count']].values

merged_data_true = merged_data[merged_data['Traffic_stream'] == 1]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_training_data = scaler.fit_transform(training_data)

train_size = int(len(scaled_training_data) * 0.8)
train_data, test_data = scaled_training_data[0:train_size], scaled_training_data[train_size:len(scaled_training_data)]


def create_dataset(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

time_steps = 10

X_train, y_train = create_dataset(train_data, time_steps)
X_test, y_test = create_dataset(test_data, time_steps)
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(units=2))

optimizer = Adam(learning_rate=0.001)

model.add(Dense(units=2, kernel_regularizer=l2(0.01)))

model.compile(optimizer=optimizer, loss='mean_squared_error')

model.fit(X_train, y_train, epochs=200, batch_size=64)


work_hours = pd.date_range(start='2024-04-01 08:00:00', end='2024-04-30 20:00:00', freq='H')
forecast_hours = work_hours[(work_hours.hour >= 8) & (work_hours.hour < 20)]

forecast_input = train_data[-time_steps:].reshape(1, time_steps, 2)
forecast_output = []

for _ in range(len(forecast_hours)):
    forecast_value = model.predict(forecast_input)[0]
    forecast_output.append(forecast_value)
    forecast_input = np.append(forecast_input[:, 1:, :], [[forecast_value]], axis=1)

forecast_output = np.array(forecast_output)

forecast_output = scaler.inverse_transform(forecast_output)

forecast_df = pd.DataFrame({'Date': forecast_hours,
                            'klk_EN_forecast': forecast_output[:, 0],
                            'total_count_forecast': forecast_output[:, 1]})

forecast_df.to_excel('forecast_april_2024_lstm.xlsx', index=False)
