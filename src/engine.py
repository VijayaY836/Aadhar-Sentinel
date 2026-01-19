import pandas as pd

def calculate_metrics(df):
    # Trend 1: The Migration Velocity
    # High demographic updates relative to low new enrollments suggests a transient population.
    df['Migration_Velocity'] = (df['Demo Age 17'] + df['Demo Age 5 17']) / (df['Age 0 5'] + 1)
    
    # Trend 2: The Policy Compliance Gap
    # People updating demographics but ignoring mandatory biometrics.
    df['Biometric_Compliance_Rate'] = (df['Bio Age 5 17'] + df['Bio Age 17']) / \
                                      (df['Demo Age 5 17'] + df['Demo Age 17'] + 1)
    
    # Trend 3: Urban Saturation Index
    # High volume of adult enrollments in specific pincodes signals a labor hub.
    df['Workforce_Saturation'] = df['Age 18 Greater'] / (df['District'].map(df.groupby('District')['Age 18 Greater'].mean()) + 1)
    
    return df