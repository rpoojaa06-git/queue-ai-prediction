import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def load_data():
    data = pd.read_csv("data/queue_data.csv")
    X = data.drop("wait_time", axis=1)
    y = data["wait_time"]
    return X, y


def train_model(X, y):
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create model
    model = LinearRegression()

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("📉 Mean Squared Error:", mse)
    print("📊 R2 Score:", r2)

    return model


if __name__ == "__main__":
    X, y = load_data()
    model = train_model(X, y)

    # Test prediction
    # arrival_rate, service_time, counters, queue_length
    sample = pd.DataFrame(
    [[7, 3, 2, 15]],
    columns=["arrival_rate", "service_time", "counters", "queue_length"]
    )
    prediction = model.predict(sample)

    print("\n🔮 Predicted Wait Time:", prediction[0])
    import joblib

joblib.dump(model, "models/wait_time_model.pkl")
print("✅ Model saved!")