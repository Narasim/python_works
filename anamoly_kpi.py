import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque

start_time ='12/25/2024'
end_time ='12/31/2024'
freq = 'min'
anomaly_prob = 0.3
anomaly_duration = 50
anomaly_magnitude = 150
features = 10


def generate_anomaly_data(start_time, 
                          end_time, 
                          freq, 
                          features, 
                          anomaly_prob, 
                          anomaly_magnitude, 
                          anomaly_duration):
    time_range = pd.date_range(start = start_time, end = end_time, freq = freq)
    num_points = time_range.shape[0]
    data = pd.DataFrame(time_range, columns=['date'])
    for i in range(features):
        base_value = np.random.uniform(50, 100)
        noise = np.random.normal(0, 5, num_points)
        anomalies = np.zeros(num_points)  
        anomaly_starts = np.random.choice(range(num_points), size=int(num_points * anomaly_prob), replace=False)
        for start in anomaly_starts:
            end = min(start + anomaly_duration, num_points)
            anomalies[start:end] = 1  
        anomaly_values = np.random.uniform(anomaly_magnitude * -1, anomaly_magnitude, num_points) * anomalies
        kpi_values = base_value + noise + anomaly_values
        data[f"kpi_{i+1}"] = kpi_values

    return data



def ewma_anomaly(data, alpha, threshold_multiplier):
    df = pd.DataFrame(data)
    ewma = df.ewm(alpha=alpha).mean()
    std = df.ewm(alpha=alpha).std()
    threshold_upper = ewma + threshold_multiplier * std
    threshold_lower = ewma - threshold_multiplier * std
    anomalies = (df > threshold_upper) | (df < threshold_lower)
    return anomalies

anom_data = generate_anomaly_data(start_time, 
                          end_time, 
                          freq, 
                          features, 
                          anomaly_prob, 
                          anomaly_magnitude, 
                          anomaly_duration)





def detect_anomalies(anom_data, t, temporal_threshold, alpha, threshold_multiplier):
    data_buffers = {}
    anomaly_counts = {}
    for index, row in anom_data.iterrows():
        timestamp = row['date']
        for indicator_name in anom_data.columns[1:]:  
            value = row[indicator_name]
            if indicator_name not in data_buffers:
                data_buffers[indicator_name] = deque()
                anomaly_counts[indicator_name] = 0
            data_buffers[indicator_name].append((timestamp, value))

            while data_buffers[indicator_name] and data_buffers[indicator_name][0][0] < timestamp - t:
                old_timestamp, old_data = data_buffers[indicator_name].popleft()
                if ewma_anomaly({indicator_name: [old_data]}, alpha, threshold_multiplier)[indicator_name][0]:
                    anomaly_counts[indicator_name] -= 1
            values = [v for _, v in data_buffers[indicator_name]]
            if len(values) > 0:
                if ewma_anomaly({indicator_name: values}, alpha, threshold_multiplier)[indicator_name].iloc[-1]:
                    anomaly_counts[indicator_name] += 1

                
                if anomaly_counts[indicator_name] >= temporal_threshold:
                    print(f"Temporal Anomaly Alert: {timestamp}, Indicator: {indicator_name}, Message: Temporal anomaly detected for last {t} with {anomaly_counts[indicator_name]} anomalies")
                    anomaly_counts[indicator_name] = 0




anom_data = pd.DataFrame(anom_data)
anom_data['date'] = pd.to_datetime(anom_data['date'])

t = timedelta(minutes=1)
temporal_threshold = 3
alpha = 0.2
threshold_multiplier = 2

print(anom_data.head())

detect_anomalies(anom_data, t, temporal_threshold, alpha, threshold_multiplier)