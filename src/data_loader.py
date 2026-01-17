import pandas as pd

def load_data(path='data/reviews.csv'):
    try:
        df = pd.read_csv(path)
        # Rename columns for clarity
        df = df.rename(columns={'text_': 'review', 'label': 'label'})
        # Keep only the necessary columns
        df = df[['review', 'label']]
        print(f"[INFO] Loaded {len(df)} reviews.")
        return df
    except FileNotFoundError:
        print("[ERROR] Dataset not found.")
        return None

