
import pandas as pd
from flask import Flask, render_template,request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_data(x):
    return str.lower(x.replace(" ", ""))

def create_soup(x):
    return x['title']+ ' ' + x['director'] + ' ' + x['cast'] + ' ' +x['listed_in']+' '+ x['description']

def get_recommendations(title, cosine_sim):
    global result
    title=title.replace(' ','').lower()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    result =  netflix_overall['title'].iloc[movie_indices]
    result = result.to_frame()
    result = result.reset_index()
    del result['index']    
    return result
    
netflix_overall = pd.read_csv('netflix_titles.csv')
netflix_data = pd.read_csv('netflix_titles.csv')
netflix_data = netflix_data.fillna('')

new_features = ['title', 'director', 'cast', 'listed_in', 'description']
netflix_data = netflix_data[new_features]
for new_features in new_features:
    netflix_data[new_features] = netflix_data[new_features].apply(clean_data)
netflix_data['soup'] = netflix_data.apply(create_soup, axis=1)
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(netflix_data['soup'])
global cosine_sim2 
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
netflix_data=netflix_data.reset_index()
indices = pd.Series(netflix_data.index, index=netflix_data['title'])
#get_recommendations('PK', cosine_sim2)



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/about',methods=['POST'])
def getvalue():
    moviename = request.form['moviename']
    get_recommendations(moviename,cosine_sim2)
    df=result
    return render_template('result.html',  tables=[df.to_html(classes='data')], titles=df.columns.values)

if __name__ == '__main__':
    app.run(debug=False)
