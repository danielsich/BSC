import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
file_path = '../time/outmod.csv'
df = pd.read_csv(file_path)

# Calculate 'totalcost' as the sum of 'costs' and 'driver_pay'
#df['totalcost'] = df['costs'] + df['driver_pay']

# Calculate 'totalcost_per_100km_distance' as (totalcost / distance) * 100000
#df['totalcost_per_100km_distance'] = (df['totalcost'] / df['distance']) * 100000

df['timewindows'] = df['timewindows'] / 60
# Group by 'timewindows' and calculate the average 'totalcost_per_100km_distance' for each group
grouped = df.groupby('timewindows')['totalcost_per_100km_distance']
average_cost_per_timewindow = grouped.median()

# Filter only the groups where more than 25 values are available
counts_per_timewindow = grouped.size()
filtered_average_cost_per_timewindow = average_cost_per_timewindow[counts_per_timewindow > 30]

# Convert to DataFrame for seaborn
filtered_df = filtered_average_cost_per_timewindow.reset_index()

# Plot the data using seaborn
plt.figure(figsize=(12, 6))
sns.lineplot(x='timewindows', y='totalcost_per_100km_distance', data=filtered_df, marker='o', palette='viridis')

# Add labels and title
plt.xlabel('Zeitfenster in Minuten')
plt.ylabel('Durchschnittskosten in € pro 100 Kilometer')
#plt.title('Average Total Cost per 100km Distance for Each Timewindow (More than 25 Entries)')
plt.xticks(rotation=45)

# Show plot
plt.show()