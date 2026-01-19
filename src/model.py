from sklearn.ensemble import IsolationForest

def detect_anomalies(df):
    # Select features that should logically follow a pattern
    features = ['Demo Age 5 17', 'Bio Age 5 17', 'Migration_Velocity']
    
    # Contamination=0.02 means we expect 2% of data to be anomalous (fraud/errors)
    iso_forest = IsolationForest(contamination=0.02, random_state=42)
    
    df['Is_Anomaly'] = iso_forest.fit_predict(df[features])
    # Map -1 to "Anomalous" and 1 to "Normal" for the UI
    df['Status'] = df['Is_Anomaly'].map({1: 'Normal', -1: 'Anomalous'})
    
    return df