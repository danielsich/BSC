import pandas as pd
import numpy as np

# csv2Dataframe
file_path = '../output/outpdsize.csv'
df = pd.read_csv(file_path)

# nan.tts column with the maximum value of 'tts'
df['tts'].replace('nan', np.nan, inplace=True)
max_tts = df['tts'].max(skipna=True)
df['tts'].fillna(max_tts, inplace=True)

##tts2num
df['tts'] = pd.to_numeric(df['tts'])

quartiles = df.groupby('customers')['tts'].quantile([0.25, 0.5, 0.75]).unstack()

quartiles.columns = ['25%', '50%', '75%']

# Print the quartiles
print(quartiles)