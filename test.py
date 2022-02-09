import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql('SELECT * FROM DisasterResponse', con = engine)
X = df['message']
y = df[df.columns.tolist()[2:]]

test = df.groupby('offer').count()['message']
my_string = test.to_string()
split_string = my_string.split('\n')
print(test.to_string())
# print(type(test))