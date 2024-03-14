
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense


en_data = pd.read_csv("109_tmp_data_en.csv")
gm_data = pd.read_csv("109_tmp_data_gm.csv")
en_data.rename(columns={
    'year': 'Year', 'month': 'Month', 'day': 'Day', 'traffic_stream': 'Traffic_stream', 'hours': 'Hour'
}, inplace=True)


def process_hours(hours):
    first_part = hours.split('_')[0].lstrip('0')
    return first_part

en_data["Hour"] = en_data["Hour"].astype(str)

en_data['Hour'] = en_data['Hour'].apply(process_hours)
en_data['Date'] = pd.to_datetime(en_data[['Year', 'Month', 'Day', 'Hour']], format='%Y-%m-%d %H')
gm_data['Date'] = pd.to_datetime(gm_data[['Year', 'Month', 'Day', 'Hour']], format='%Y-%m-%d %H')

en_data_sorted = en_data.sort_values(by='Date')
gm_data_sorted = gm_data.sort_values(by='Date')

en_data_grouped = en_data_sorted.groupby(['Date']).agg({'klk_EN': 'sum'}).reset_index()[['Date', 'klk_EN']]
gm_data_grouped = gm_data_sorted.groupby(['Date']).agg({'total_count': 'sum'}).reset_index()[['Date', 'total_count']]

en_data_task = en_data_grouped[en_data_grouped['Date'] < '2021-08-18 11:00:00']
X_task = en_data_task['klk_EN'].values.reshape(-1, 1)
en_data = en_data_grouped[['Date', 'klk_EN']]
gm_data = gm_data_grouped[['Date', 'total_count']]

merged_data = pd.merge(en_data, gm_data, on='Date', how='inner')

X = merged_data['klk_EN'].values.reshape(-1, 1)
y = merged_data['total_count'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

loss = model.evaluate(X_test, y_test)
print('Mean Squared Error on Test Set:', loss)

predictions = model.predict(X_task)

predictions_df = pd.DataFrame(predictions, columns=['Predicted_total_count'])
merged_df = pd.concat([en_data_task, predictions_df], axis=1)


merged_df.to_excel('prediction_data.xlsx', index=False)
# print(en_data_grouped.iloc[-1])
# print(gm_data_grouped.iloc[-1])
print(merged_data.loc[0])