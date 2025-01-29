import pandas as pd

# Load the CSV file into a DataFrame
file_path = '../time/out.csv'
df = pd.read_csv(file_path)

# Filter out rows where 'customers' has a value and all other columns except 'timewindows' are NaN
filtered_df = df.dropna()

# Calculate 'fuel per 100 000 distance' for each line
filtered_df['fuel_per_100km_distance'] = (filtered_df['total_fuel'] / filtered_df['distance']) * 100000

# Calculate 'totalcost' for each line
#filtered_df['totalcost'] = filtered_df['costs'] + filtered_df['driver_pay']

# Calculate 'drivercost per 100 000 distance' for each line
filtered_df['drivercost_per_100km_distance'] = (filtered_df['driver_pay'] / filtered_df['distance']) * 100000

# Save the final DataFrame to 'outmod.csv'
filtered_df.to_csv('../time/outmod.csv', index=False)