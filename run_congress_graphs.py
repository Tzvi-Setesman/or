from congress_graphs_funcs import *
import os

df = pd.read_excel( r'C:\Users\otiro\Desktop\python stuff\Netivot\experiments\data\\' + os.listdir('Netivot/experiments/data')[0])

metadata = load_us_metadata()
print('Loaded Congress Metadata')

df = format_data_for_graphs(df)
print('Processed DataFrame')

run_graphs(df,metadata)
