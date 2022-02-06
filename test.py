import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('sqlite:///disaster_database.db')
# df = pd.read_sql('SELECT * FROM ' + database_filepath, con = engine)
print(engine.table_names())
