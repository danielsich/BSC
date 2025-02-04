import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = '../eff/eff50.csv'
df = pd.read_csv(file_path)

# Group by the 'eff' column to calculate the average values
eff_averages = df.groupby('eff').mean()

# Adjust the distances and calculate fuel consumption per 100 km
eff_averages['distance_km'] = eff_averages['distance'] / 1000
eff_averages['fuel_per_100km'] = (eff_averages['total_fuel'] / eff_averages['distance_km']) * 100

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:blue'
ax1.set_xlabel('ε', fontsize=14)
ax1.set_ylabel('€', color=color, fontsize=14)
ax1.plot(eff_averages.index, eff_averages['costs'], color=color, label='Kosten')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
color = 'tab:green'
ax2.set_ylabel('(Liter/100 km)', color=color, fontsize=14)  # We already handled the x-label with ax1
ax2.plot(eff_averages.index, eff_averages['fuel_per_100km'], color=color, linestyle='dashed', label='Kraftstoffverbrauch')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # Otherwise, the right y-label is slightly clipped
#plt.title('Effizienz vs Gefahrene Distanz und Kraftstoffverbrauch', fontsize=16)
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
plt.savefig('../vis/eff_vs_costs50.svg', format='svg')

# Show the plot
plt.show()