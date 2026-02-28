# AI Queue Prediction System

## Overview
This project is an AI-driven digital twin for queue management. 
It predicts wait times at service counters based on historical or simulated queue data and identifies peak hours using K-Means clustering. 
The system provides an interactive dashboard with dynamic sliders for real-time predictions and visualizations.

## Features
- Wait time prediction using regression ML model
- Peak hour detection using K-Means clustering
- Dynamic sliders for user input prediction
- Colored scatter plot of queue length vs wait time
- Integrated Streamlit web interface
- Traffic level classification: Low, Medium, Peak

## Run Instructions
1. **Install dependencies**  
   Open a terminal in the project folder and run:

   ```bash
   pip install -r requirements.txt

Run the web UI
After installing dependencies, run:

python -m streamlit run app/ui.py

Use the interactive sliders in the web UI to input:

Arrival Rate

Service Time

Number of Counters

Queue Length

View outputs:

Predicted wait time (updates automatically)

Traffic level (Low / Medium / Peak)

Historical data table and colored scatter plot showing clusters

Notes

The queue_data.csv file currently contains simulated data.

Once Member 1 provides real queue data, replace the CSV to see predictions on actual data.

The system demonstrates beginner-friendly AI concepts combined with an interactive digital twin.