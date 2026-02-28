import joblib

# Load trained model
model = joblib.load("models/wait_time_model.pkl")

print("🔮 Queue Wait Time Predictor")

# User input
arrival_rate = float(input("Enter arrival rate: "))
service_time = float(input("Enter service time: "))
counters = int(input("Enter number of counters: "))
queue_length = int(input("Enter queue length: "))

# Prediction
prediction = model.predict([[arrival_rate, service_time, counters, queue_length]])

print(f"\n⏱️ Predicted Wait Time: {prediction[0]:.2f} minutes")