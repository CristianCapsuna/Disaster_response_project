import sys
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import pickle
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet') # download for lemmatization
nltk.download('omw-1.4')

def load_data(database_filepath):
    engine = create_engine('sqlite:///data/' + database_filepath)
    df = pd.read_sql('SELECT * FROM ' + database_filepath[:-3], con = engine)

    X = df['message'].to_numpy()
    category_names = df.columns.tolist()[1:]
    Y = df[category_names].to_numpy()

    return X, Y, category_names


def tokenize(text):
    cachedStopWords = stopwords.words("english")
    # Normalize text and get ride of punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Actual tokenization
    words = word_tokenize(text)
    # Getting rid of common words
    words = [w for w in words if w not in cachedStopWords]
    # 1st pass at reducing words to base form using nouns
    words_noun_lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    # 2nd pass at reducing words to base form using verbs
    words_verb_lemmed = [WordNetLemmatizer().lemmatize(w, pos = 'v') for w in words_noun_lemmed]
    return words_verb_lemmed


def build_model():

    basic_MultinomialNB = Pipeline([('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])), \
            ('clasification_model', MultiOutputClassifier(MultinomialNB()))])
    
    return basic_MultinomialNB


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)

    with open("models/model_results.txt", "w") as f:

        for index, column in enumerate(category_names):
            f.write('Current column is {}.\n\n'.format(column))
            print('Current column is {}.\n'.format(column))
            report = classification_report(Y_test[:,index], Y_pred[:,index], zero_division = 0)
            f.write(report)
            print(report)


def save_model(model, model_filepath):
    with open('models/' + model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
            'as the first argument and the filepath of the pickle file to '\
            'save the model to as the second argument. \n\nExample: python '\
            'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

# python model/train_classifier.py disaster_database.db my_model.pkl