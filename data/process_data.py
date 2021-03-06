import sys
import pandas as pd
from sqlalchemy import create_engine

def switch_two(x):
    """
    One of thecolumns in the labels dataset has more than two values. This function will convert that column into binary values only
    """

    if x == 1 or x == 2:
        return 1
    else:
        return 0


def load_data(messages_filepath, categories_filepath):
    """
    Function to load the data from their CSVs and join them on their IDs
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = pd.merge(messages, categories, on = 'id')

    return df

def clean_data(df):
    """
    Here the folowing procedures are done:
    1) splitting the categories dataset in individual columns
    2) converting the `related` column to binay values
    3) eliminating duplicates
    4) eliminating the child_alone column as there are only 0 labels in it so cannot be used to train a model
    """
    # split the categories data
    df = pd.concat([df[['message', 'genre']] ,df['categories'].str.split(';', expand = True)], axis = 1)

    # take the first row minus the messages column
    row_except_message = df.iloc[0,2:]

    # take only the name of the categories
    category_colnames = ['message', 'genre']
    for name in row_except_message:
        category_colnames.append(name[:-2])

    # assigning the name of the categories as column headers
    df.columns = category_colnames

    # filter the categories to leave just the label state (0 or 1)
    for column in df.columns[2:]:
        # set each value to be the last character of the string
        df[column] = df[column].str[-1]
        
        # convert column from string to numeric
        df[column] = df[column].astype(int)

    df['related'] = df['related'].map(lambda x: switch_two(x))
    
    df = df.drop_duplicates()
    
    columns = df.columns.tolist()
    # removing the child_alone column because it doesn't have a 1 label
    columns.remove('child_alone')

    df = df[columns]
    

    # data has no nans

    return df


def save_data(df, database_filename):
    """
    This function will save the data from one dataframe to a database.
    """
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(database_filename[:-3].split('/')[1], engine, if_exists='replace', index=False)


def main():
    """
    Main function use to orchestrate the importing, cleaning and exporting of the data for this project
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

# python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db