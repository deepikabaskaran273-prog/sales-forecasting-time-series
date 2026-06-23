import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load dataset
data = pd.read_csv("sales_data.csv")

# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Set Date as index
data.set_index('Date', inplace=True)

# Plot sales trend
plt.figure(figsize=(10, 5))
plt.plot(data['Sales'])
plt.title('Sales Trend')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

# Build forecasting model
model = ExponentialSmoothing(
    data['Sales'],
    trend='add',
    seasonal=None
).fit()

# Forecast next 12 periods
forecast = model.forecast(12)

print("Future Sales Forecast:")
print(forecast)

# Plot forecast
plt.figure(figsize=(10, 5))
plt.plot(data['Sales'], label='Actual Sales')
plt.plot(forecast, label='Forecast', color='red')
plt.title('Sales Forecast')
plt.legend()
plt.show()
