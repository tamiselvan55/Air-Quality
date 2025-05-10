Original file is located at 
     https://colab.research.google.com/drive/1vvk7ywiqR5vxL2ZjudKeKkn_j9BIQN8t?usp=sharing

# Step 1: Upload the dataset file from your local machine to Colab
from google.colab import files
uploaded = files.upload()

# Step 2: Load the uploaded CSV file into a pandas DataFrame
import pandas as pd
df = pd.read_csv("Air_Quality.csv")
df.head()

# Step 3: View dataset structure, statistics, and column names
df.info()
df.describe()
df.columns

# Step 4: Check for missing values and duplicate rows in the dataset
print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

# Step 5: Plot the distribution of NO2 levels to understand the data visually
import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(df["Data Value"], kde=True)
plt.title("Distribution of NO2 Levels")
plt.show()

# Step 6: Convert Start_Date to datetime and extract Year and Month as new features
df["Start_Date"] = pd.to_datetime(df["Start_Date"])
df["Year"] = df["Start_Date"].dt.year
df["Month"] = df["Start_Date"].dt.month

# Step 7: Convert categorical columns into numerical format using one-hot encoding
df = pd.get_dummies(df, columns=["Geo Type Name", "Geo Place Name"])

# Step 8: Define input features and target variable, and remove irrelevant columns
from sklearn.preprocessing import StandardScaler
features = df.drop(columns=["Data Value", "Message", "Start_Date", "Time Period", "Name", "Measure Info", "Measure"])
target = df["Data Value"]

# Step 9: Normalize the feature values using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Step 10: Split the data into training and testing sets for model evaluation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=0.2, random_state=42)

# Step 11: Train a Random Forest Regressor model using the training data
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Step 12: Evaluate the model using MAE and R² score on the test data
from sklearn.metrics import mean_absolute_error, r2_score
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Step 13: Make a sample prediction using one test sample
sample = X_test[0].reshape(1, -1)
prediction = model.predict(sample)
print("Predicted Value:", prediction)

# Step 14: Install Gradio for building a simple web app interface
!pip install gradio

# Step 15: Define a Gradio-compatible function that takes input and returns a prediction
import gradio as gr
def predict_air_quality(geo_id, year, month):
    import numpy as np
    input_data = pd.DataFrame([[geo_id, year, month]], columns=["Geo Join ID", "Year", "Month"])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return f"Predicted NO2 Level: {prediction[0]:.2f} ppb"

# Step 16: Build and launch the Gradio web interface for real-time predictions
interface = gr.Interface(
    fn=predict_air_quality,
    inputs=["number", "number", "number"],
    outputs="text",
    title="Air Quality Predictor"
)
interface.launch()
