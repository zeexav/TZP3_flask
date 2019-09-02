# TZP3_flask

This Repository contains all the requirements to create a Flask App of the TZP3 Movie Recommender System.
The datasets were significantly shrunk to be able to run the app when deployed to Heroku.

About the Data:
28,929 user ratings from 17,664 unique users. 
3,094 unique movies

Datasets used in App:
movies_demo.csv
movies_bow_demo.csv
preds_demo.h5
ratings_demo.h5
ratings_matrix_demo.h5

Running the Collaborative Filtering Machine learning model using SciPy SVDs is resource intensive.By preprocessing the results of SVD Machine Learning model we reduced the memory requirement of the app. We also utilized file compression using Pandas dataframe_to_hdf to compress all of the large data files. This allowed us to get the slug size to under the 500mb requirement for Heroku and avoid exceeding the 512mb ram, memory quota.