import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load datasets
file_paths = {
    'outprpsize10': '../output/outprpsize10.csv',
    'outopt': '../output/outopt.csv'
}

quartiles_dict = {}

for name, file_path in file_paths.items():
    df = pd.read_csv(file_path, na_values='nan')

    # Convert tts to numeric
    df['tts'] = pd.to_numeric(df['tts'], errors='coerce')

    # Fill NaN in tts column based on the file
    if name == 'outprpsize10':
        df['tts'].fillna(900, inplace=True)
    elif name == 'outopt':
        df['tts'].fillna(1800, inplace=True)

    # Filter data to include only customers <= 11
    df_filtered = df[df['customers'] <= 11]

    # Calculate quartiles
    quartiles = df_filtered.groupby('customers')['tts'].quantile([0.25, 0.5, 0.75]).unstack()
    quartiles_dict[name] = quartiles

# Plotting
plt.figure(figsize=(12, 6))
label_mapping = {
    'outprpsize10': '$F_C$',
    'outopt': '$F_C*$'
}

for name, quartiles in quartiles_dict.items():
    plt.plot(quartiles.index, quartiles[0.5], label=f'{label_mapping[name]} Median')
    plt.fill_between(quartiles.index, quartiles[0.25], quartiles[0.75], alpha=0.2)

plt.xlabel('Kunden', fontsize=14)
plt.ylabel('Rechenzeit', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
plt.savefig('../vis/tts_medians_comparison_outprpsize10_outopt.svg', format='svg')

plt.show()