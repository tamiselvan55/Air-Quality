original file is located at
           https://colab.research.google.com/drive/1vvk7ywiqR5vxL2ZjudKeKkn_j9BIQN8t?usp=sharing#scrollTo=80c6133

from google.colab import files
uploaded = files.upload()

import pandas as pd

df = pd.read_csv("Air_Quality.csv")
df.head()

df.info()
df.describe()
df.columns

print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df["Data Value"], kde=True)
plt.title("Distribution of NO2 Levels")
plt.show()

df["Start_Date"] = pd.to_datetime(df["Start_Date"])
df["Year"] = df["Start_Date"].dt.year
df["Month"] = df["Start_Date"].dt.month

df = pd.get_dummies(df, columns=["Geo Type Name", "Geo Place Name"])

from sklearn.preprocessing import StandardScaler

features = df.drop(columns=["Data Value", "Message", "Start_Date", "Time Period", "Name", "Measure Info", "Measure"])
target = df["Data Value"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error, r2_score

y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

sample = X_test[0].reshape(1, -1)
prediction = model.predict(sample)
print("Predicted Value:", prediction)

!pip install gradio

import gradio as gr
import numpy as np
import pandas as pd

# Save column names used during training
feature_columns = features.columns

def predict_air_quality(geo_id, year, month):
    # Reconstruct an input row with all columns set to 0
    input_data = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)

    # Insert the actual inputs into the correct columns
    if 'Geo Join ID' in input_data.columns:
        input_data.at[0, 'Geo Join ID'] = geo_id
    if 'Year' in input_data.columns:
        input_data.at[0, 'Year'] = year
    if 'Month' in input_data.columns:
        input_data.at[0, 'Month'] = month

    # Predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    return f"Predicted NO₂ Level: {prediction[0]:.2f} ppb"

# Launch Gradio
interface = gr.Interface(
    fn=predict_air_quality,
    inputs=[
        gr.Number(label="Geo ID"),
        gr.Number(label="Year"),
        gr.Number(label="Month")
    ],
    outputs=gr.Text(label="Predicted NO₂ Level"),
    title="Air Quality Predictor"
)

interface.launch(share=True)


## 🎓 Air Quality Predictor Complete!
You’ve built a full machine learning pipeline and deployed a simple interactive app.