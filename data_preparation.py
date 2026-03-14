import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load dataset
data = pd.read_csv("spotify_songs.txt")

print("Original Dataset:")
print(data.head())

# Remove missing values
data = data.dropna()

# Standardize column names
data.columns = data.columns.str.lower()

# Select important features
features = ["bpm","energy","valence","instrumentalness","danceability"]
music_features = data[features]

# Normalize the data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(music_features)

normalized_df = pd.DataFrame(normalized_data, columns=features)

# Combine song info with normalized features
clean_data = pd.concat([data[["song","artist"]], normalized_df], axis=1)

print("\nCleaned Dataset:")
print(clean_data.head())

# Save cleaned dataset
clean_data.to_csv("clean_music_dataset.csv", index=False)

print("\nData preparation completed successfully.")