import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
file_paths = {
    'outpdserver': '../output/outpdsizeserver.csv',
    'outplserver': '../output/outplserver.csv',
    'outpeserver': '../output/outpeserver.csv',
    'outprpserver': '../output/outprpsize10.csv'
}

# Read the base dataset separately for comparison
base_df = pd.read_csv(file_paths['outpdserver'], na_values='nan')
base_df['distance'] = pd.to_numeric(base_df['distance'], errors='coerce')
base_df.dropna(subset=['distance'], inplace=True)

# Prepare a DataFrame to hold customers per vehicle data
customers_per_vehicle_data = []

for name, file_path in file_paths.items():
    df = pd.read_csv(file_path, na_values='nan')
    df['vehicles'] = pd.to_numeric(df['vehicles'], errors='coerce')
    df.dropna(subset=['vehicles'], inplace=True)

    # Apply cutoff for outprpserver at 8 customers
    #if name == 'outprpserver':
    #    df = df[df['customers'] <= 9]
    #df = df[df['customers'] <= 9]
    df = df[df['customers'] <= 25]
    df = df[df['customers'] >= 11]
    # Collect customers per vehicle data
    for _, row in df.iterrows():
        customers_per_vehicle = row['customers'] / row['vehicles']  # Assuming 'distance' is the number of vehicles
        customers_per_vehicle_data.append((name, customers_per_vehicle))

# Create a DataFrame for the boxplot
customers_per_vehicle_df = pd.DataFrame(customers_per_vehicle_data, columns=['Model', 'Customers per Vehicle'])

# Map the model names to labels
label_mapping = {
    'outpdserver': '$F_D$',
    'outplserver': '$F_L$',
    'outpeserver': '$F_E$',
    'outprpserver': '$F_C$'
}

# Replace model names with labels in the DataFrame
customers_per_vehicle_df['Model'] = customers_per_vehicle_df['Model'].map(label_mapping)

# Output the summary statistics to the terminal
summary_stats = customers_per_vehicle_df.groupby('Model')['Customers per Vehicle'].describe(percentiles=[.25, .5, .75])
print(summary_stats[['25%', '50%', '75%']])

# Plotting
plt.figure(figsize=(12, 6))
sns.boxplot(x='Model', y='Customers per Vehicle', data=customers_per_vehicle_df, palette="Set3")

plt.xlabel('Modell', fontsize=14)
plt.ylabel('Kunden pro Fahrzeug', fontsize=14)
plt.xticks(fontsize=14, rotation=45)
plt.yticks(fontsize=14)
plt.grid(True, axis='y')
# plt.savefig('../vis/customers_per_vehicle_boxplot_11_25.svg', format='svg')

plt.show()