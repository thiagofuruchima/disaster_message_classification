# Disaster Message Classification Project

A ML tool for classifying messages related to natural disasters.

### Table of Contents

1. [Project Motivation](#motivation)
2. [Installation](#installation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Motivation<a name="motivation"></a>

This project uses Data Pipelines, Natural Language Processing (NLP) Pipelines and Machine Learning Pipelines to build a message classification Web App.

The WebApp classifies messages on 36 categories, as being which could help better address a disaster response.

Similar approuches could be used for many domain specific problems, like "crime related messages", "job applications", etc.

## Installation <a name="installation"></a>

This project uses the following libraries:

pandas, json, plotly, nltk, flask, joblib, sqlalchemy, sklearn.
 
To run the WebApp, follow these steps:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## File Descriptions <a name="files"></a>

The app folder contains the WebApp related files.

The data folder contains the data pipeline related files.

The model folder contains the machine learning pipeline related files.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Credits to [Figure Eight](https://www.figure-eight.com/) for the data. 

This is a student's project, take a look at the [MIT Licence](LICENSE) before using elsewhere.
