import json
import plotly
import pandas as pd
import re
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download(['punkt','wordnet','averaged_perceptron_tagger','stopwords'])

from sklearn.base import BaseEstimator, TransformerMixin

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]"," ", text)
    
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    
    lemmatizer = WordNetLemmatizer()
    clean_words = []
    for w in words:
        clean_w = lemmatizer.lemmatize(w).lower().strip()
        clean_words.append(clean_w)
        
    return clean_words

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        
        text = re.sub(r"[^a-zA-Z0-9]"," ", text)
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('CleanData', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    top_classification_count = df.iloc[:,4:].sum().sort_values(ascending=False).head(5)
    top_classification_names = list(top_classification_count.index)
    
    bottom_classification_count = df.iloc[:,4:].sum().sort_values(ascending=False).tail(5)
    bottom_classification_names = list(bottom_classification_count.index)
    
    # create visuals
    graphs = [
         {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    y=top_classification_count,
                    x=top_classification_names
                )
            ],

            'layout': {
                'title': 'Top 5 Message Classifications',
                'yaxis': {
                    'title': "Count of Messages"
                },
                'xaxis': {
                    'title': "Classification"
                }
            }
        },
         {
            'data': [
                Bar(
                    y=bottom_classification_count,
                    x=bottom_classification_names
                )
            ],

            'layout': {
                'title': 'Bottom 5 Message Classifications',
                'yaxis': {
                    'title': "Count of Messages"
                },
                'xaxis': {
                    'title': "Classification"
                }
            }
        }
    ]
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()