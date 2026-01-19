import pandas as pd

def get_unified_data(enrol_path, demo_path, bio_path):
    # Load sources
    df_enrol = pd.read_csv(enrol_path)
    df_demo = pd.read_csv(demo_path)
    df_bio = pd.read_csv(bio_path)
    
    file_names = ['enrol.csv', 'demo.csv', 'bio.csv']
    
    for i, df in enumerate([df_enrol, df_demo, df_bio]):
        # REX'S ULTIMATE FIX: Force first letter of every column to be Capitalized
        # This turns 'date' into 'Date' and 'pincode' into 'Pincode' automatically.
        df.columns = [col.strip().capitalize() for col in df.columns]
        
        if 'Date' not in df.columns:
            raise ValueError(f"Hey Via! The column 'Date' is still missing in {file_names[i]}!")

        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        
    # Standardize merge keys to Title Case
    merge_keys = ['Date', 'State', 'District', 'Pincode']
    
    # Merge datasets
    merged = pd.merge(df_enrol, df_demo, on=merge_keys, how='outer')
    merged = pd.merge(merged, df_bio, on=merge_keys, how='outer')
    
    # Ensure mandatory biometrics comparison logic works
    # Using .get() prevents crashes if columns are missing
    merged['Update_Lag'] = merged.get('Demo age 17', 0) - merged.get('Bio age 17', 0)
        
    return merged.fillna(0)