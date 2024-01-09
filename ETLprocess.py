import pandas as pd
import numpy as np
from ast import literal_eval
import matplotlib.pyplot as plt
import seaborn as sns 

#Data import
data_netflix = pd.read_csv('Axity_case\data_netflix.csv')
data_netflix

top_peliculas = pd.read_csv('Axity_case\mejores_peliculas_netflix.csv')
top_peliculas

top_shows = pd.read_csv('Axity_case\mejores_shows_netflix.csv')
top_shows

actores = pd.read_csv('Axity_case\Actores.csv')
actores

#Checker function for datasets 

def dataset_checker(dataset):
    columnas = dataset.columns
    for i in columnas:
        print('Column '+ i)
        print(dataset[i].unique())
        print("-"*80)
    print(dataset.describe())
    print("-"*80)
    print(dataset.info())


#Review 1 dataset: data_netflix
dataset_checker(data_netflix)

data_netflix_modified = data_netflix

#Change data types 
data_netflix_modified['id'] = data_netflix_modified['id'].apply(lambda x: str(x) if pd.notna(x) else '')
data_netflix_modified['title'] = data_netflix_modified['title'].apply(lambda x: str(x) if pd.notna(x) else '')
data_netflix_modified['type'] = data_netflix_modified['type'].apply(lambda x: str(x) if pd.notna(x) else '')

data_netflix_modified['seasons'] = data_netflix_modified['seasons'].fillna(0) 
data_netflix_modified['seasons'] = data_netflix_modified['seasons'].astype(int)

data_netflix_modified['type'] = data_netflix_modified['type'].apply(lambda x: x.title())

data_netflix_modified['imdb_id'] = data_netflix_modified['imdb_id'].apply(lambda x: str(x) if pd.notna(x) else '')

#Genre transformation (get main_genre)
data_netflix_modified['genres'] = data_netflix_modified['genres'].apply(literal_eval)
data_netflix_modified['main_genre'] = data_netflix_modified['genres'].apply(lambda x: x[0] if x else None)
data_netflix_modified['main_genre']

data_netflix_modified.columns

#Production country transformation (get main_production_country)
data_netflix_modified['production_countries'] = data_netflix_modified['production_countries'].apply(literal_eval)
data_netflix_modified['main_prod_country'] = data_netflix_modified['production_countries'].apply(lambda x: x[0] if x else None)
data_netflix_modified['main_prod_country']

data_netflix_modified.columns

#Add description of age_certification code and age_group movie
def age_cert_descrip(code):
    if code == 'TV-Y': resultado = 'All children'
    elif code == 'TV-Y7' : resultado = 'Directed to Older Children'
    elif code == 'TV-G' : resultado = 'General Audience'
    elif code == 'TV-PG' : resultado = 'Parental Guidance Suggested'
    elif code == 'TV-14' : resultado = 'Parents Strongly Cautioned'
    elif code == 'TV-MA' : resultado = 'Mature Audience Only'
    elif code == 'G' : resultado = 'General Audience'
    elif code == 'PG' : resultado = 'Parental Guidance Suggested'
    elif code == 'PG-13' : resultado = 'Parents Strongly Cautioned'
    elif code == 'R' : resultado = 'Restricted'
    elif code == 'NC-17' : resultado = 'Clearly Adult'
    else: resultado = None
    return resultado

data_netflix_modified['age_cert_descrip'] = data_netflix_modified['age_certification'].apply(age_cert_descrip)
data_netflix_modified['age_cert_descrip'].unique()

def age_group(code):
    if code in ['TV-Y','TV-Y7'] : resultado = 'Kids'
    elif code in ['PG-13','TV-14', 'TV-PG', 'PG']: resultado = 'Teens'
    elif code in ['TV-MA','NC-17', 'R']: resultado = 'Adults'
    elif code in ['TV-G', 'G']: resultado = 'All ages'
    else: resultado = None
    return resultado

data_netflix_modified['movie_age_group'] = data_netflix_modified['age_certification'].apply(age_group)
data_netflix_modified['movie_age_group'].unique()

#Review movies near 0 runtime
runtime_check_movies = data_netflix_modified[['title', 'type','runtime']].sort_values(by='runtime', ascending=False)
runtime_check_movies = runtime_check_movies[(runtime_check_movies['runtime'] >= 0) & (runtime_check_movies['runtime'] < 15) & (runtime_check_movies['type'] == 'MOVIE')]
runtime_check_movies

#Review series near 0 runtime
runtime_check_shows = data_netflix_modified[['title', 'type','runtime']].sort_values(by='runtime', ascending=False)
runtime_check_shows = runtime_check_shows[(runtime_check_shows['runtime'] > 0) & (runtime_check_shows['runtime'] < 10) & (runtime_check_shows['type'] == 'SHOW')]
runtime_check_shows

#Add IMDb score measure (categorical)
data_score_qt = data_netflix['imdb_score'].quantile([0.25, 0.75]).to_dict()

def score_cat(score):
    if score <= data_score_qt[0.25]: return 'Low score'
    elif (score > data_score_qt[0.25]) & (score < data_score_qt[0.75]): return 'Moderate score'
    elif score >= data_score_qt[0.75]: return 'High score'
    else: return None

data_netflix_modified['score_group'] = data_netflix_modified['imdb_score'].apply(score_cat)

data_netflix_modified

#Review 2 dataset: mejores_peliculas_netflix
dataset_checker(top_peliculas)
top_peliculas

#Review 3 dataset: mejores_shows_netflix
dataset_checker(top_shows)
top_shows

#Review 4 dataset: actores
dataset_checker(actores)
actores_modified = actores
actores_modified['person_id'] = actores_modified['person_id'].apply(lambda x: str(x) if pd.notna(x) else '')

#Add column if the movie is in mejores_peliculas_netflix or in mejores_shows_netflix (categorized as top movies/shows)
#Movie checker
len(top_peliculas)
mejores_peliculas_checker = pd.merge(data_netflix, top_peliculas, left_on=['title', 'release_year'], right_on=['TITLE', 'RELEASE_YEAR'], how='inner')
mejores_peliculas_checker_list = mejores_peliculas_checker[['title', 'release_year']]
mejores_peliculas_checker_list['concatenate'] = mejores_peliculas_checker_list[['title', 'release_year']].apply(lambda x: str(x['title']) + str(x['release_year']), axis=1)
mejores_peliculas_checker_list

len(mejores_peliculas_checker_list)

#Show checker
len(top_shows)
mejores_shows_checker = pd.merge(data_netflix, top_shows, left_on=['title', 'release_year'], right_on=['TITLE', 'RELEASE_YEAR'], how='inner')
mejores_shows_checker_list = mejores_shows_checker[['title', 'release_year']]
mejores_shows_checker_list['concatenate'] = mejores_shows_checker_list[['title', 'release_year']].apply(lambda x: str(x['title']) + str(x['release_year']), axis=1)
mejores_shows_checker_list
len(mejores_shows_checker_list)

#Add column
print(len(mejores_peliculas_checker_list)+len(mejores_shows_checker_list))

data_netflix['concatenate'] = data_netflix[['title', 'release_year']].apply(lambda x: str(x['title']) + str(x['release_year']), axis=1)
data_netflix['best_movies/shows'] = np.where(data_netflix['concatenate'].isin(mejores_peliculas_checker_list['concatenate'])==True, 'Top movies/shorts', 
                                             np.where(data_netflix['concatenate'].isin(mejores_shows_checker_list['concatenate'])==True, 'Top shows', None))

len(data_netflix[data_netflix['best_movies/shows'].isnull()==False])
data_netflix[data_netflix['best_movies/shows'].isnull()==False]
data_netflix.columns

#Add actors/directors count for id to netflix dataset
film_id_actors = actores[actores['role']=='ACTOR'].groupby('id')['person_id'].agg({'count'}).reset_index()
film_id_actors = film_id_actors.rename(columns={'count':'actor_count'})
film_id_directors = actores[actores['role']=='DIRECTOR'].groupby('id')['person_id'].agg({'count'}).reset_index()
film_id_directors = film_id_directors.rename(columns={'count':'director_count'})

data_netflix_merged = pd.merge(pd.merge(data_netflix, film_id_actors, on = 'id', how='left'), film_id_directors, on='id', how='left')


#Export dataset
data_netflix_final = data_netflix_merged[['id', 'title', 'type', 'release_year', 'age_certification', 'movie_age_group', 'age_cert_descrip','runtime', 'main_genre', 'main_prod_country', 'imdb_score',
       'imdb_votes', 'best_movies/shows', 'seasons','actor_count', 'director_count', 'score_group']]
data_netflix_final.columns

data_netflix_final.to_csv('Axity_case\data_netflix_modified.csv', index=False)



#EDA data_netflix
len(data_netflix_final)

#Netflix films count by type
sns.countplot(data=data_netflix_final, x='type')
plt.title('Movie count by type')
plt.show()

#Films count by type throughout the years
movies_by_year = data_netflix_final.groupby(['release_year', 'type'])['id'].agg({'count'}).reset_index()
movies_by_year
sns.lineplot(data=movies_by_year, x='release_year', y='count', hue='type')
plt.title('Films count by type over the years')
plt.show()

#Films count by type that year released > 1990
movies_after_1990 = data_netflix_final[data_netflix_final['release_year']>1990]
movies_by_year_1990 = movies_after_1990.groupby(['release_year', 'type'])['id'].agg({'count'}).reset_index()
sns.lineplot(data=movies_by_year_1990, x='release_year', y='count', hue='type')
plt.title('Film count by type over the years (>1990)')
plt.show()

#Films count by age group
sns.countplot(data=data_netflix_final, x='movie_age_group')
plt.title('Film count by Audience age group')
plt.show()

#Films count by age group throughout the years 
movies_by_year_age = movies_after_1990.groupby(['release_year', 'movie_age_group'])['id'].agg({'count'}).reset_index()
sns.lineplot(data=movies_by_year_age, x='release_year', y='count', hue='movie_age_group')
plt.title('Film count by age group over the years (>1990)')
plt.show()

#Boxplot for identifying outliers with film type breakdown
runtime_netflix_data = data_netflix_final[data_netflix_final['runtime']>0]
sns.boxplot(data=runtime_netflix_data, x='type', y='runtime')
plt.title('Runtime by Film type') 
plt.show()

#Boxplot for identifying outliers with film type breakdown of High score only
runtime_netflix_data_highscore = data_netflix_final[(data_netflix_final['runtime']>0)&(data_netflix_final['score_group']=='High score')]
sns.boxplot(data=runtime_netflix_data_highscore, x='type', y='runtime')
plt.title('Runtime by Film type with High score') 
plt.show()

sns.barplot(data=runtime_netflix_data_highscore[runtime_netflix_data_highscore['type']=='Show'], x='seasons', y='imdb_votes')
plt.title('IMDB votes by season with film with high score') 
plt.show()

#Countplot of film count by genre
sns.countplot(data=data_netflix_final, x='main_genre', order=data_netflix_final['main_genre'].value_counts().index)
plt.title('Genres count')
plt.xticks(rotation = 45)
plt.show()

#Films count by main_genre throughout the years
movies_by_genre_1990 = movies_after_1990.groupby(['release_year', 'main_genre'])['id'].agg({'count'}).reset_index()
sns.lineplot(data=movies_by_genre_1990, x='release_year', y='count', hue='main_genre')
plt.title('Film count by main genre over the years (>1990)')
plt.show()

#Countplot of films by production country
top_countries = data_netflix_final['main_prod_country'].value_counts().nlargest(10).index
sns.countplot(data=data_netflix_final, x='main_prod_country', order=top_countries)
plt.title('Top 10 Production country count')
plt.show()

#Boxplot for identifying outliers of IMDBb scores by film type
imdb_scores_general = data_netflix_final[(data_netflix_final['imdb_score']>0) & (data_netflix_final['best_movies/shows'].isnull()==False)]
sns.boxplot(data=imdb_scores_general, x='best_movies/shows', y='imdb_score') 
plt.show()

#Histogram for imdb_score distribution (best movies/shows breakdown also)
imdb_scores_df = data_netflix_final.dropna(subset=['imdb_score'])
imdb_scores_df
imdb_top_ms = imdb_scores_df[imdb_scores_df['best_movies/shows']=='Top movies/shorts']
imdb_top_shows = imdb_scores_df[imdb_scores_df['best_movies/shows']=='Top shows']

plt.figure(figsize=(12, 6))
sns.histplot(data=imdb_scores_df, x='imdb_score', bins=20, label='IMDb Score General', color='blue', alpha=0.5)
sns.histplot(data=imdb_top_ms, x='imdb_score', bins=20, label='IMDb Score Best Movies', color='yellow', alpha=0.5)
sns.histplot(data=imdb_top_shows, x='imdb_score', bins=20, label='IMDb Score Best Shows', color='green', alpha=0.5)
plt.title('Histograms IMDb scores')
plt.xlabel('IMDb')
plt.ylabel('Frequency')
plt.legend()
plt.show()

#Barplot for imdb_votes distribution by genres
imdb_votes_df = data_netflix_final.dropna(subset=['imdb_votes'])
imdb_votes_grouped = imdb_votes_df.groupby('main_genre')['imdb_votes'].agg({'sum'}).reset_index().sort_values(by='sum', ascending=False)
imdb_votes_grouped

sns.barplot(data=imdb_votes_grouped, x='main_genre', y='sum')
plt.title('IMDb votes distribution by genre')
plt.xticks(rotation=45)
plt.show()

#Correlation between multiple variables
correlation_matrix = data_netflix_final[data_netflix_final['imdb_score']>0].dropna().corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation matrix')
plt.show()

#Films with IMDb score over 7
films_over_7 = data_netflix_final[data_netflix_final['imdb_score']>=7].sort_values(by='imdb_score', ascending=False)
sns.countplot(data=films_over_7, x='type')
plt.title('Film count with score over 7')
plt.show()

#Correlation between multiple variables with film over score 7
correlation_matrix_over7 = films_over_7.corr()
sns.heatmap(correlation_matrix_over7, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation matrix')
plt.show()


#EDA actores
copy_actores = actores

actores_count = copy_actores[copy_actores['role']=='ACTOR']['name'].value_counts().reset_index().rename(columns={'index':'name', 'name': 'films_count'})
actores_count

directors_count = copy_actores[copy_actores['role']=='DIRECTOR']['name'].value_counts().reset_index().rename(columns={'index':'name', 'name': 'films_count'})
directors_count.head(10)
