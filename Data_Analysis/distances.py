import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load datasets
file_paths = {
    'outpdserver': '../output/outpdsizeserver.csv',
    'outplserver': '../output/outplserver.csv',
    'outpeserver': '../output/outpeserver.csv',
    'outprpserver': '../output/outprpsize10.csv'
}

average_distances_dict = {}

for name, file_path in file_paths.items():
    df = pd.read_csv(file_path, na_values='nan')

    # Convert tts to numeric
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')

    # Drop NaN values in tts column
    df.dropna(subset=['distance'], inplace=True)


    # Calculate average distances per customer
    average_distances = df.groupby('customers')['distance'].mean()
    average_distances_dict[name] = average_distances

# Plotting
plt.figure(figsize=(12, 6))

label_mapping = {
    'outpdserver': '$F_D$ Average Distance',
    'outplserver': '$F_L$ Average Distance',
    'outpeserver': '$F_E$ Average Distance',
    'outprpserver': '$F_C$ Average Distance'
}

for name, average_distances in average_distances_dict.items():
    plt.plot(average_distances.index, average_distances, label=label_mapping[name])

plt.yscale('log')
plt.xlabel('Kunden', fontsize=14)
plt.ylabel('Durchschnittliche Distanz', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
#plt.title('Average Distances per Customersize for Different Datasets', fontsize=16)
#plt.savefig('../vis/average_distances_comparison.svg', format='svg')
plt.show()