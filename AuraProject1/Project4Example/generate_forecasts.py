
def generate_forecasts(model, start_sequence, n_forecast):
    forecasted = start_sequence[-sequence_length:].tolist()  # Start with the last known sequence
    for _ in range(n_forecast):
        if len(forecasted) > sequence_length:
            current_sequence = forecasted[-sequence_length:]
        else:
            current_sequence = forecasted
        current_sequence = np.array(current_sequence).reshape(1, sequence_length, 1)
        next_step = model.predict(current_sequence)
        forecasted.append(next_step[0, 0])
    return scaler.inverse_transform(np.array(forecasted[sequence_length:]).reshape(-1, 1))

# Forecast future EM field patterns
n_forecast = 200  # Number of steps to forecast into the future
forecasted_data = generate_forecasts(model, data_scaled.tolist(), n_forecast)

# Plotting the forecasted EM field patterns
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(data)), scaler.inverse_transform(data_scaled), label='Original Data')
plt.plot(np.arange(len(data), len(data) + n_forecast), forecasted_data, label='Forecasted Data', linestyle='--')
plt.title("Forecasted EM Field Patterns")
plt.xlabel("Time Step")
plt.ylabel("Field Intensity")
plt.legend()
plt.show()
