from flask import Flask, render_template, request
from flask_table import Table, Col
import pandas as pd
from flask import *
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import tables



app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analysis')
def analysis():
    return render_template('analysis.html')


@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html')


@app.route("/send", methods=["GET", "POST"])
def send():
    if request.method == "POST":
        movie_name = request.form['title']
        user = request.form['userId']
        user = int(user)
        
        data = pd.read_csv('Data/movies_bow_demo.csv')
        movies = pd.read_csv('Data/movies_demo.csv')
        print(os.getcwd())
        #Begin Content Filtering process
        indices = pd.Series(data.index, index=data['Title'])
        idx = indices[movie_name]
        count = CountVectorizer()
        count_matrix = count.fit_transform(data['bag_of_words'])
        cosine_sim = cosine_similarity(count_matrix, count_matrix)
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:51]
        movie_indices = [i[0] for i in sim_scores]
        content_results = pd.DataFrame(data['movieId'].iloc[movie_indices])

        # files for collaborative filtering model
        ratings = pd.read_hdf('Data/ratings_demo.h5')
        ratings_demo_matrix = pd.read_hdf('Data/ratings_demo_mx.h5')

        U = joblib.load('Data/U.pkl')
        Vt = joblib.load('Data/Vt.pkl')
        sigma = joblib.load('Data/sigma.pkl')

        # normalize the data by each users mean and convert it from a dataframe to a numpy array.
        R = ratings_demo_matrix.values
        user_ratings_mean = np.mean(R, axis = 1)

        # The last step of normalization is not needed in the function since it is used when U, Vt and sigma were previously calculated
        # Ratings_demeaned = R - user_ratings_mean.reshape(-1, 1) 
        

        #calculate prediction scores
        user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
        preds_demo = pd.DataFrame(user_predicted_ratings, columns = ratings_demo_matrix.columns)

        #map the userId column to the predictions
        id = ratings_demo_matrix.index
        preds_demo['id'] = id

        
        #begin collaborative filtering process
        
        user = preds_demo.loc[preds_demo['id']== user].index[0]
        sorted_user_predictions = preds_demo.iloc[user].sort_values(ascending=False) 
        sorted_user_predictions = pd.DataFrame(sorted_user_predictions[1:]).reset_index()

        # Get the movies the user originally rated
        user_data = ratings[ratings.userId == user]
        user_full = (user_data.merge(movies, how = 'left', left_on = 'movieId', right_on = 'movieId').
                     sort_values(['rating'], ascending=False))
        #grab only needed columns
        user_full = user_full[['userId', 'movieId', 'Title_x', 'rating', 'genres', 'Actors', 'Director', 'Plot', 'Poster']].\
            rename(columns = {'Title_x': 'Title'})

        # set number of items to return from collaborative filter process
        num_recommendations = 100

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        movie_preds = (movies[~movies['movieId'].isin(user_full['movieId'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'movieId',
               right_on = 'movieId').
         rename(columns = {user: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations])

        #Find the similarity scores for the movies returned from the content system. 
        # Display only the highest ranked ones

       
        movie_recs = pd.merge(content_results, movie_preds, how='left', on='movieId').\
        sort_values('Predictions', ascending=False).dropna()
        top10_df = pd.DataFrame(movie_recs[['Title','genres', 'Plot']][:10])
        results = top10_df.to_dict('records')
        columnNames = top10_df.columns.values

        return render_template('recommendation.html', records = results, colnames = columnNames)




if __name__ == '__main__':
    app.debug = True
    app.run()