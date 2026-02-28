import pandas as pd

def load_data(path):
    # Load CSV file
    data = pd.read_csv(path)
    print("✅ Data Loaded Successfully")
    print(data.head())
    return data


def preprocess_data(data):
    # Check for missing values
    print("\n🔍 Checking missing values:")
    print(data.isnull().sum())

    # For now, drop missing values (simple approach)
    data = data.dropna()

    # Separate features and target
    X = data.drop("wait_time", axis=1)
    y = data["wait_time"]

    print("\n✅ Preprocessing Done")
    return X, y


if __name__ == "__main__":
    # Path to dataset
    path = "data/queue_data.csv"

    data = load_data(path)
    X, y = preprocess_data(data)

    print("\n📊 Features:")
    print(X.head())

    print("\n🎯 Target:")
    print(y.head())