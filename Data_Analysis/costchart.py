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

drivers_pay_percentage_dict = {}

for name, file_path in file_paths.items():
    df = pd.read_csv(file_path, na_values='nan')

    # Convert costs and drivers_pay to numeric
    df['costs'] = pd.to_numeric(df['costs'], errors='coerce')
    df['driver_pay'] = pd.to_numeric(df['driver_pay'], errors='coerce')
    df.dropna(subset=['costs', 'driver_pay'], inplace=True)

    # Calculate drivers_pay percentage of costs
    drivers_pay_percentage = (df['driver_pay'] / df['costs']) * 100
    drivers_pay_percentage_dict[name] = drivers_pay_percentage.mean()

# Plotting
plt.figure(figsize=(12, 6))

model_names = list(drivers_pay_percentage_dict.keys())
drivers_pay_percentages = list(drivers_pay_percentage_dict.values())

plt.bar(model_names, drivers_pay_percentages, color='skyblue')

plt.xlabel('Model', fontsize=14)
plt.ylabel('Drivers Pay Percentage of Costs (%)', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, axis='y')
#plt.savefig('../vis/drivers_pay_percentage_of_costs.svg', format='svg')

plt.show()