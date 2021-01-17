# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import joblib
import nltk


# Download the necessary corpus
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    """ Load the CLEAN_MESSAGES table from the given SQLite Database """

    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("CLEAN_MESSAGES", engine)
    X = df['message']
    Y = df.iloc[:,4:]

    return X, Y, Y.columns


def tokenize(text):
    """ Extract lemmatized tokens from the given text """
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """ Build a multiclass message classifier using 
        RandomForest and the best_params found using GridSearch """
    
    # Define the Pipeline structure
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf',  MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Define GridSearch parameters
    parameters = {'clf__estimator__n_estimators': range(100, 200, 100),
                  'clf__estimator__min_samples_split': range(50, 100, 10)}

    # Instantiate GridSearch object
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """ Print the Classification Report on test set (X_test, Y_test) for each category """

    # Predict using the trained model with the best parameters
    Y_pred = model.predict(X_test)
    
    # Print the classification report for for each column
    for i, column in enumerate(category_names):
        print("Columns: ", column)
        print(classification_report(Y_test.values[:,i], Y_pred[:,i], zero_division=0))
        print()    


def save_model(model, model_filepath):
    """ Save the given model to a Python Pickle file"""
    # Save the model to pickl file
    #pickled_model = pickle.dumps(cv.best_estimator_)
    #print("Model Size (Mb): ", sys.getsizeof(pickled_model)/(1024**2))
    #file.write(pickled_model)
    joblib.dump(model, model_filepath, compress=5)


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