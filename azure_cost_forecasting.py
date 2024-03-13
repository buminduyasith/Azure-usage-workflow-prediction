import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv("AzureUsageData.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Filter data for a specific year (e.g., 2024)
selected_year = 2024
df_year = df[df['Date'].dt.year == selected_year]

# Prepare data for forecasting
service_costs = df_year.groupby(['Date', 'ServiceName'])['Cost'].sum().reset_index()

# Perform time series forecasting for each service
services = df_year['ServiceName'].unique()

for service in services:
    service_data = service_costs[service_costs['ServiceName'] == service]
    service_data = service_data.groupby('Date')['Cost'].sum().reset_index()
    service_data.columns = ['ds', 'y']

    order = (1, 1, 1) 
    seasonal_order = (1, 1, 1, 7) 
    model = sm.tsa.statespace.SARIMAX(service_data['y'], order=order, seasonal_order=seasonal_order)
    results = model.fit(disp=False)

    # Forecast future values
    forecast_periods = 30  
    forecast = results.get_forecast(steps=forecast_periods)
    forecast_index = pd.date_range(start=service_data['ds'].min(), periods=len(service_data) + forecast_periods, freq='D')
    forecast_values = forecast.predicted_mean

    # Plot the original data and forecasted values for each service
    plt.figure(figsize=(10, 5))
    plt.plot(service_data['ds'], service_data['y'], label='Actual')
    plt.plot(forecast_index[-forecast_periods:], forecast_values[-forecast_periods:], label='Forecast', color='red', linestyle='--')
    plt.title(f'Cost Forecast for {service} in {selected_year}')
    plt.xlabel('Date')
    plt.ylabel('Total Cost')
    plt.legend()
    plt.show()