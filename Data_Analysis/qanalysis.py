import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = '../qi/q8.csv'
df = pd.read_csv(file_path)

# Calculate the customerdemands in kilograms
df['customerdemands'] = df['a'] * 10

# Group by the 'customerdemands' column to calculate the average values
demand_averages = df.groupby('customerdemands').mean()

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:blue'
ax1.set_xlabel('kg', fontsize=14)
ax1.set_ylabel('Fahrzeuge', color=color, fontsize=14)
ax1.plot(demand_averages.index, demand_averages['vehicles'], color=color, label='Durchschnittliche Anzahl der Fahrzeuge')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
color = 'tab:green'
ax2.set_ylabel('km/h', color=color, fontsize=14)  # We already handled the x-label with ax1
ax2.plot(demand_averages.index, demand_averages['averagespeed'], color=color, linestyle='dashed', label='Durchschnittsgeschwindigkeit')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # Otherwise, the right y-label is slightly clipped
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

#plt.savefig('../vis/demand_vs_vehicles_and_speed8.svg', format='svg')
plt.show()