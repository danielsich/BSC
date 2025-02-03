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

# Prepare a DataFrame to hold percentage increases
percentage_increase_data = []

for name, file_path in file_paths.items():
    if name == 'outpdserver':
        continue  # Skip the base model itself

    df = pd.read_csv(file_path, na_values='nan')
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
    df.dropna(subset=['distance'], inplace=True)

    # Apply cutoff for outprpserver at 8 customers
    if name == 'outprpserver':
        df = df[df['customers'] <= 10]
    #df = df[df['customers'] <= 9]
    # Calculate percentage increase compared to the base model for each row
    for _, row in df.iterrows():
        customer = row['customers']
        if customer in base_df['customers'].values:
            base_distance = base_df[base_df['customers'] == customer]['distance'].values[0]
            percentage_increase = ((row['distance'] - base_distance) / base_distance) * 100
            if percentage_increase >= 0:  # Filter out negative values
                percentage_increase_data.append((name, customer, percentage_increase))

# Create a DataFrame for the boxplot
percentage_increase_df = pd.DataFrame(percentage_increase_data, columns=['Model', 'Customers', 'Percentage Increase'])

# Map the model names to labels
label_mapping = {
    'outplserver': '$F_L$',
    'outpeserver': '$F_E$',
    'outprpserver': '$F_C$'
}

# Replace model names with labels in the DataFrame
percentage_increase_df['Model'] = percentage_increase_df['Model'].map(label_mapping)

# Output the summary statistics to the terminal
summary_stats = percentage_increase_df.groupby('Model')['Percentage Increase'].describe(percentiles=[.25, .5, .75])
print(summary_stats[['25%', '50%', '75%']])
# Plotting
plt.figure(figsize=(12, 6))
sns.boxplot(x='Model', y='Percentage Increase', data=percentage_increase_df, palette="Set3")

plt.xlabel('Modell', fontsize=14)
plt.ylabel('%', fontsize=14)
plt.xticks(fontsize=14, rotation=45)
plt.yticks(fontsize=14)
plt.grid(True, axis='y')
#plt.savefig('../vis/percentage_increase_boxplot_corr2.svg', format='svg')

plt.show()