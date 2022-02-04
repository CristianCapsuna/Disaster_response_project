import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('sqlite:///IO_handling/Disaster_response_messenges.db')
df = pd.read_sql('SELECT * FROM Disaster_response_messenges', con = engine)
X = df['message']
y = df[df.columns.tolist()[4:]]

X = X.drop(y[y.isnull().any(axis = 1)].index.values.tolist())
y = y.dropna()

for col in y.columns:
    if len(y[col].unique()) <2:
        y = y.drop(col, axis = 1)

print(y.shape)
print(y.sum())