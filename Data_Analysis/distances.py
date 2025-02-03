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

# Read outpdserver separately for the base comparison
base_df = pd.read_csv(file_paths['outpdserver'], na_values='nan')
base_df['distance'] = pd.to_numeric(base_df['distance'], errors='coerce')
base_df.dropna(subset=['distance'], inplace=True)
base_distances = base_df.groupby('customers')['distance'].mean()

percentage_increase_dict = {}

for name, file_path in file_paths.items():
    if name == 'outpdserver':
        continue  # Skip the base model itself

    df = pd.read_csv(file_path, na_values='nan')
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
    df.dropna(subset=['distance'], inplace=True)

    # Apply cutoff for outprpserver at 9 customers
    if name == 'outprpserver':
        df = df[df['customers'] <= 10]

    # Calculate average distances per customer
    average_distances = df.groupby('customers')['distance'].mean()

    # Calculate percentage increase compared to the base model
    common_customers = average_distances.index.intersection(base_distances.index)
    percentage_increase = ((average_distances[common_customers] - base_distances[common_customers]) / base_distances[common_customers]) * 100
    percentage_increase_dict[name] = percentage_increase

# Plotting
plt.figure(figsize=(12, 6))

label_mapping = {
    'outplserver': '$F_L$',
    'outpeserver': '$F_E$',
    'outprpserver': '$F_C$'
}

for name, percentage_increase in percentage_increase_dict.items():
    plt.plot(percentage_increase.index, percentage_increase, label=label_mapping[name])

plt.xlabel('Kunden', fontsize=14)
plt.ylabel('%', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
#plt.title('Percentage Increase in Distances Compared to $F_D$', fontsize=16)
plt.savefig('../vis/percentage_increase_comparison.svg', format='svg')
plt.show()