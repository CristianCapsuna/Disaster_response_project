import json
import plotly
import pandas as pd

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objects import Bar
# from sklearn.externals import joblib # this does not work in the new version of sklearn
import joblib
from sqlalchemy import create_engine
import plotly.express as px
import nltk
import re

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet') # download for lemmatization
nltk.download('omw-1.4')


app = Flask(__name__)

def tokenize(text):
    """
    The tokenization process does the following:
    1) eliminates punctuation
    2) converts words into an indexed vocabulary
    3) eliminates common words like `is`, `the`, etc
    4) brings nouns to their base forms. I think this means the infinitive tense
    5) brings vers to their base forms. I think this means the infinitive tense
    """

    # tokens = word_tokenize(text)
    # lemmatizer = WordNetLemmatizer()

    # clean_tokens = []
    # for tok in tokens:
    #     clean_tok = lemmatizer.lemmatize(tok).lower().strip()
    #     clean_tokens.append(clean_tok)

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

    return words_verb_lemmed

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Takes the master template for the main page and adds 3 visualizations to show some key characteristics of our dataset
    """
    
    graphs = []

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    # Vis 1
    # genre_counts = df.groupby('genre').count()['message']
    # genre_names = list(genre_counts.index)
    genre_counts_df = df.groupby('genre', as_index = False).count()

    genre_names = genre_counts_df['genre'].tolist()

    genre_counts = genre_counts_df['message'].tolist()


    vis1 = px.bar(y = genre_counts
             ,title = 'Distribution of Message Genres'
             ,x = genre_names
             ,labels = {'x':'Genre', 'y':'Counts'}
    )
    vis1.update_layout(title_x=0.5)

    graphs.append(vis1)

    # Vis 2
    column_names = df.columns[3:]
    ratios_list = []
    number_of_rows = df.groupby(column_names[0]).count().iloc[:,0][0]
    for column in column_names:
        counts_of_label_1 = df.groupby(column).count().iloc[:,0][1]
        ratio_of_label_1_to_total = counts_of_label_1/number_of_rows
        ratios_list.append(ratio_of_label_1_to_total)
    
    vis2 = px.bar(y = ratios_list
             ,title = 'Ratios between counts of label 1 and the total row number'
             ,x = column_names
             ,labels = {'x':'Column', 'y':'Ratio'}
             ,height = 800
    )
    vis2.update_layout(title_x=0.5)
    graphs.append(vis2)

    # Vis 3
    # message_stats_labels = ['min', 'mean', 'max']
    lengths  = df['message'].str.len().to_list()
    i = 0
    lengths_up_to_500 = []
    for elem in lengths:
        if elem > 500:
            i += 1
        else:
            lengths_up_to_500.append(elem)

    # message_stats = []
    # message_stats.append(lengths_df.min())
    # message_stats.append(lengths_df.mean())
    # message_stats.append(lengths_df.max())

    vis3 = px.histogram(x = lengths_up_to_500
             ,title = 'Number of characters in messages. {lengths_over} messages above 500 characters'.format(lengths_over = i)
             ,labels = {'x':'Text length', 'y':'Occurances'}
            #  ,height = 600
    )
    vis3.update_layout(title_x=0.5)

    graphs.append(vis3)
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    Defines what happens when we input new text for the app to classify in the provided box
    """
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[2:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """
    Main function to activate the flask app.
    """
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()

# python app/run.py