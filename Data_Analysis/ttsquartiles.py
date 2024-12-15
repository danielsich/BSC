import pandas as pd
import numpy as np

# csv2Dataframe
file_path = '../output/outprpsize.csv'
df = pd.read_csv(file_path)

# nan.tts column with 180
df['tts'].replace('nan', np.nan, inplace=True)
df['tts'].fillna(180, inplace=True)

##tts2num
df['tts'] = pd.to_numeric(df['tts'])

quartiles = df.groupby('customers')['tts'].quantile([0.25, 0.5, 0.75]).unstack()

quartiles.columns = ['25%', '50%', '75%']

# Print the quartiles
print(quartiles)