# Importing Libraries
import sys
import pandas as pd
import re
import numpy as np
import pickle
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download(['punkt','wordnet','averaged_perceptron_tagger','stopwords'])

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin


def load_data(database_filepath):
    
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('CleanData',engine)  
    #Defining feature and target variables X and Y
    X = df['message']
    Y = df.drop(['id','message','original','genre'],axis=1)
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
        
    """ 
    Transform text into clean tokens
    
    Args: original text
    Return: clean tokens (words)
    
    """
    
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
    
def build_model():
    
    model = Pipeline([
        ('features',FeatureUnion([
            
            ('pipeline', Pipeline([
                ('vect',CountVectorizer(tokenizer=tokenize)),
                ('tfidf',TfidfTransformer())
            ])),
            
            ('starting_verb',StartingVerbExtractor())
        ])),
        
        ('model', MultiOutputClassifier(AdaBoostClassifier(random_state=54)))
    ])
    
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    
    y_pred = model.predict(X_test)

    for i in range(len(category_names)):
        print('\n'+category_names[i])
        print(classification_report(Y_test.iloc[:,i], y_pred[:,i]))


def save_model(model, model_filepath):
    
    pickle.dump(model, open('models/classifier.pkl', 'wb'))


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