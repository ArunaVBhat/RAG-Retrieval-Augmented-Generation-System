import pandas as pd
import os
import plotly.express as px
from dash import Dash, dcc, html
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

# Folder containing multiple CSV files
folder_path = r"kagglehub/datasets/boltzmannbrain/nab/versions/1/artificialWithAnomaly/artificialWithAnomaly"
for file_name in os.listdir(folder_path):
    print(f"Processing file: {file_name}")
    file_path = os.path.join(folder_path, file_name)
    temp_df = pd.read_csv(file_path)
    print(f"Columns in {file_name}: {list(temp_df.columns)}")

# Combine all CSV files into a single DataFrame
all_data = pd.DataFrame()

for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):  # Only process CSV files
        file_path = os.path.join(folder_path, file_name)
        temp_df = pd.read_csv(file_path)

        # Check if required columns exist
        if 'timestamp' in temp_df.columns and 'value' in temp_df.columns:
            temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'], errors='coerce')  # Handle invalid timestamps
            temp_df.dropna(subset=['timestamp', 'value'], inplace=True)  # Drop rows with invalid timestamps or values
            temp_df['file_name'] = file_name  # Add a column to track source file
            all_data = pd.concat([all_data, temp_df], ignore_index=True)
        else:
            print(f"Skipping {file_name} - Missing required columns 'timestamp' or 'value'")

# Check if all_data is empty after processing
if all_data.empty:
    raise ValueError("No valid data found. Ensure all CSV files have 'timestamp' and 'value' columns.")

# Preprocess data
all_data = all_data.sort_values(by='timestamp')  # Sort by timestamp

# Set up an anomaly detection threshold
threshold = all_data['value'].mean() + 1 * all_data['value'].std()
all_data['anomaly_flag'] = all_data['value'].apply(lambda x: 'Anomaly' if x > threshold else 'Normal')

# Simulate ground truth labels for evaluation
all_data['ground_truth_label'] = all_data['value'].apply(lambda x: 'Anomaly' if x > 60 else 'Normal')

# Train Isolation Forest model
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(all_data[['value']])
all_data['predicted_labels'] = model.predict(all_data[['value']])
all_data['predicted_labels'] = all_data['predicted_labels'].map({1: 'Normal', -1: 'Anomaly'})

# Evaluate the model
y_true = all_data['ground_truth_label'].map({'Normal': 0, 'Anomaly': 1})
y_pred = all_data['predicted_labels'].map({'Normal': 0, 'Anomaly': 1})

precision = precision_score(y_true, y_pred, zero_division=1)
recall = recall_score(y_true, y_pred, zero_division=1)
f1 = f1_score(y_true, y_pred, zero_division=1)

# Initialize Dash app
app = Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("Multi-File Anomaly Detection Dashboard", style={'textAlign': 'center'}),
    dcc.Graph(
        id='dataset-graph',
        figure=px.scatter(
            all_data,
            x='timestamp',
            y='value',
            color='anomaly_flag',
            color_discrete_map={'Anomaly': 'red', 'Normal': 'blue'},
            title="Anomalies in Combined Dataset",
            labels={'anomaly_flag': 'Status'},
            hover_data=['file_name']  # Show the source file in the hover tooltip
        )
    ),
    html.Div([
        html.H3("Model Metrics"),
        html.P(f"Precision: {precision:.2f}"),
        html.P(f"Recall: {recall:.2f}"),
        html.P(f"F1 Score: {f1:.2f}"),
    ], style={'textAlign': 'center', 'marginTop': '20px'})
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
