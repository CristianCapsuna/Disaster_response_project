import pandas as pd
from sqlalchemy import create_engine, MetaData

engine = create_engine('sqlite:///disaster_database.db')
df = pd.read_sql('SELECT * FROM disaster_database.db', con = engine)

# metadata_obj = MetaData()
# metadata_obj.reflect(bind=engine)
# print(metadata_obj)
