import json
import plotly
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

# Download the necessary corpus
# nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

app = Flask(__name__)

def tokenize(text):
    """ Function to handle the messages and return the clean tokens """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('CLEAN_MESSAGES', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """ Function to handle the default requests. This is the entry point to the Web App.
        All necessary Home page visualizations should be implemented here.
    """
    categories = df.iloc[:,4:].sum().sort_values(ascending=False)
    category_names = list(categories.index.str.title().str.replace('_', ' '))
    category_counts = list(categories)

    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index.str.title())

    related_percentage = (df['related']>0).mean().round(2)*100
    related_counts = [related_percentage, 100-related_percentage]
    related_names = ['Related', 'Non Related']
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Top Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=related_names,
                    y=related_counts
                )
            ],

            'layout': {
                'title': 'Percentage of Messages Disaster Related',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Related x Non Related"
                },
                'color': '[Red, Blue]'
            }
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """ Function to handle message processing. The messages will be handled
    by the ML pipeline defined by model """

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
    """ Wrapper function for the Web App inicialization """
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()