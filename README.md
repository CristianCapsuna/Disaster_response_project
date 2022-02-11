# Disaster Response Pipeline Project

## Requirements

- python 3.10
- libraries provided in requirements.txt

## File descriptions

- disaster_categories.csv and disaster_messages.csv are the data sources for this project containing labels and messages corresponsing to these labels respectively
- process_data.py processes the above files to split the labels into their individual columns and leave 1 or 0. This is followed by a bit of cleaning and finally storage into a database
- train_classifier.py uses the cleaned data, from the databasepreviously mentioned, to train a naive bayes machine learning model from scikit-learn. The data is put through a tokenizing function that splits the string into words, strips out unnecessary words like "is", "the", etc. and changes them to their base forms using lematization. The resulting model is stored in a pickle file using python's standard persistance library with the same name
- run.py deploy's a flask app with plotly visualizations which uses the model stored from the previous script to classify messages and label them according to the pre-programed labels
- ETL Pipeline preparation.ipynb contains the same code as process_data.py and was used to develop the script and inspect it's different stages
- ML Pipeline Preparation.ipynb contains the same code as train_classifier.py and was used to develop the script and inspect it's different stages. It also contains experimentation with hyperparameter tuning using GridSearchCV. The algorithm yielded a set of better parameters but when using them on the test data no significant improvement was observed.

## Motivation

This project was created to experiment with machine learning solutions offered by sci-kit-learn and visualization capabilities offered by plotly and flask.

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Follow the http address given by flask in the CLI to check the visualization
