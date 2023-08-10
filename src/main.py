# main.py
import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder

# Load data and perform encoding
data = pd.read_csv('../data/lens_tmdb/ratings_small.csv')
ratings_df = data[['userId', 'movieId', 'rating']].copy()  # Create a copy to avoid pandas warnings
ratings_df.loc[:, 'rating'] = ratings_df['rating'].apply(lambda x: 1 if x >= 4 else 0)
user_enc = LabelEncoder()
ratings_df.loc[:, 'userId'] = user_enc.fit_transform(ratings_df['userId'].values)
item_enc = LabelEncoder()
ratings_df.loc[:, 'movieId'] = item_enc.fit_transform(ratings_df['movieId'].values)

# Determine number of unique users and items
num_users = ratings_df['userId'].nunique()
num_items = ratings_df['movieId'].nunique()


# Define the AutoEncoder architecture (as given above)
class AutoEncoder(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_dim):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, users, items):
        user_emb = self.user_emb(users)
        item_emb = self.item_emb(items)
        x = torch.cat([user_emb, item_emb], dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.sigmoid(x).squeeze()


# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder(num_users, num_items, embedding_dim=50, hidden_dim=100).to(device)
model.load_state_dict(torch.load('../data/lens_tmdb/model_weights.pth'))
model.eval()

# Load data and perform encoding
data = pd.read_csv('../data/lens_tmdb/ratings_small.csv')
ratings_df = data[['userId', 'movieId', 'rating']]
ratings_df['rating'] = ratings_df['rating'].apply(lambda x: 1 if x >= 4 else 0)
user_enc = LabelEncoder()
ratings_df['userId'] = user_enc.fit_transform(ratings_df['userId'].values)
item_enc = LabelEncoder()
ratings_df['movieId'] = item_enc.fit_transform(ratings_df['movieId'].values)

movies_df = pd.read_csv('../data/lens_tmdb/cleaned/df_all.csv')


def get_movie_details(movie_id):
    original_id = item_enc.inverse_transform([movie_id])
    movie_row = movies_df[movies_df['movieId'] == original_id[0]]
    movie_name = movie_row['title'].values[0]
    genres = movie_row['genre'].values[0]
    vote_average = movie_row[movie_row['movieId'] == original_id[0]]['vote_average'].values[0]
    return movie_name, genres, vote_average


# Function to get movieId from movie name
def get_movie_id_from_name(movie_name):
    movie_row = movies_df[movies_df['title'] == movie_name]
    if not movie_row.empty:
        return int(movie_row['movieId'].values[0])
    else:
        return None


# Function to recommend movies based on a given movie's embedding
def recommend_based_on_movie(movie_name, model, num_items):
    movie_id = get_movie_id_from_name(movie_name)
    if movie_id is None:
        st.warning('Movie not found in the database!')
        return []

    encoded_movie_id = item_enc.transform([movie_id])[0]
    item_embedding = model.item_emb(torch.tensor([encoded_movie_id]).to(device))

    all_movie_embeddings = model.item_emb(torch.tensor(range(num_items)).to(device))
    similarities = torch.mm(item_embedding, all_movie_embeddings.T).squeeze()
    _, indices = torch.topk(similarities, 11)  # Get top 11, which includes the input movie
    top_movies = indices.cpu().numpy()

    # Exclude the input movie from the list
    top_movies = [movie for movie in top_movies if movie != encoded_movie_id][:10]

    movie_details = [get_movie_details(movie_id) for movie_id in top_movies]
    return movie_details


# Streamlit UI
st.title('Movie Recommendation System')
st.write("### Enter a movie name to get similar movie recommendations:")

# Taking movie input
movie_name = st.text_input('', 'Type here...')

if st.button('Get Recommendations'):
    recommendations = recommend_based_on_movie(movie_name, model, num_items)
    if recommendations:
        st.write("### Recommended Movies:")
        for movie_name, genres, vote_average in recommendations:
            st.write("---")  # This will draw a line for separation

            # Creating a card-like structure for each movie
            st.write(f"**{movie_name}**")
            st.write(f"*Genres:* {genres}")
            st.write(f"*Vote Average:* {vote_average}")
    else:
        st.warning('No Recommendations found for the given movie!')
# Run the app using:
# streamlit run main.py
