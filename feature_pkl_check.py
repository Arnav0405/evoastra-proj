import pickle
import pandas as pd

with open('resized_features.pkl', 'rb') as f:
    features = pickle.load(f)

print(features['3192005501.jpg'])
# Convert dictionary to DataFrame
features_df = pd.DataFrame.from_dict(features, orient='index')
print(f"Loaded features with shape: {features_df.shape}")
print(features_df.head())
# print(features_df['3384742888.jpg'])
features_df.to_csv('resized_features.csv', index=False) 