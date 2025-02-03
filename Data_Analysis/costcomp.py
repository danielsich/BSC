import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load datasets
file_paths = {
    'outpdserver': '../output/outpdsizeserver.csv',
    'outplserver': '../output/outplserver.csv',
    'outpeserver': '../output/outpeserver.csv',
    'outprp10': '../output/outprpsize10.csv'
}

# Read outprp10 separately for the base comparison
base_df = pd.read_csv(file_paths['outprp10'], na_values='nan')
base_df['costs'] = pd.to_numeric(base_df['costs'], errors='coerce')
base_df.dropna(subset=['costs'], inplace=True)
base_costs = base_df.groupby('customers')['costs'].mean()

percentage_increase_dict = {}

for name, file_path in file_paths.items():
    if name == 'outprp10':
        continue  # Skip the base model itself

    df = pd.read_csv(file_path, na_values='nan')
    df['costs'] = pd.to_numeric(df['costs'], errors='coerce')
    df.dropna(subset=['costs'], inplace=True)

    # Calculate average costs per customer
    average_costs = df.groupby('customers')['costs'].mean()

    # Calculate percentage increase compared to the base model
    common_customers = average_costs.index.intersection(base_costs.index)
    percentage_increase = ((average_costs[common_customers] - base_costs[common_customers]) / base_costs[common_customers]) * 100
    percentage_increase_dict[name] = percentage_increase

# Plotting
plt.figure(figsize=(12, 6))

label_mapping = {
    'outpdserver': '$F_D$',
    'outplserver': '$F_L$',
    'outpeserver': '$F_E$'
}

for name, percentage_increase in percentage_increase_dict.items():
    plt.plot(percentage_increase.index, percentage_increase, label=label_mapping[name])

plt.xlabel('Kunden', fontsize=14)
plt.ylabel('%', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
#plt.title('Percentage Increase in Costs Compared to $F_{PRP10}$', fontsize=16)
plt.savefig('../vis/percentage_increase_cost_comparison.svg', format='svg')
plt.show()