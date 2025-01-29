import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
file_path = '../time/outmod.csv'
df = pd.read_csv(file_path)

# Calculate 'totalcost_per_100km_distance'
#df['totalcost'] = df['costs'] + df['driver_pay']
#df['totalcost_per_100km_distance'] = (df['totalcost'] / df['distance']) * 100000

#in minutes
df['timewindows'] = df['timewindows'] / 60
# Group by 'timewindows' and calculate the average 'totalcost_per_100km_distance' for each group
grouped = df.groupby('timewindows')['totalcost_per_100km_distance']
average_cost_per_timewindow = grouped.mean()

# Filter only the groups where more than 25 values are available
counts_per_timewindow = grouped.size()
filtered_average_cost_per_timewindow = average_cost_per_timewindow[counts_per_timewindow > 30]

# Plot the data
plt.figure(figsize=(10, 6))
filtered_average_cost_per_timewindow.plot(kind='line', color='blue')

# Add labels and title
plt.xlabel('Timewindows')
plt.ylabel('Average Total Cost per 100km Distance')
#plt.title('Average Total Cost per 100km Distance for Each Timewindow (More than 25 Entries)')
plt.xticks(rotation=45)

# Show plot
plt.show()