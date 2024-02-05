import streamlit as st
import pandas as pd

st.title('Simple Movie Recommendation System')

# Data Overview page
st.header('Raw Data Overview')

st.write("Used the Kaggle Dataset found [here](https://www.kaggle.com/datasets/nicoletacilibiu/movies-and-ratings-for-recommendation-system?select=movies.csv)")
movies_file = 'movies.csv'
ratings_file = 'ratings.csv'

movies_data = pd.read_csv(movies_file)
ratings_data = pd.read_csv(ratings_file)

# Display movies data
st.subheader('Movies Data:')
st.dataframe(movies_data.sample(n=10))

# Display ratings data
st.subheader('Ratings Data:')
st.dataframe(ratings_data.sample(n=10))

# Feature Enhancement page
st.header('Feature Enhancement')
st.write("Changes made:")
st.markdown("""
- Split genres and One Hot Encoded each of them 
- Converted User ID and Movie ID to strings from integer 
- Using the now Object type User ID and Movie ID Encoded them to be used by the neural network
""")

feature_enhanced_data_file = 'feature_enhanced_data.csv'

st.subheader('Enhanced Data:')
feature_enhanced_data = pd.read_csv(feature_enhanced_data_file)
st.dataframe(feature_enhanced_data.sample(n=10))

# Test Train Split Overview page

st.header('Test Train Split Overview')

st.write('To ensure there was no bias towards a specific user, I employed stratified sampling while splitting the test and train datasets, ensuring an even distribution of user IDs in both sets.')
# Sample code to split the data
code_to_split_data = """
# Split the data into training and testing sets, maintaining the class distribution
train_data, test_data = train_test_split(data, test_size=0.2, stratify=data['userId'], random_state=29)
"""
# Display the code to split the data
st.code(code_to_split_data, language='python')

# Recommendation Abstract page
st.header('Recommendation Abstract')
st.write("The selection of a neural network (NN) model was based on several considerations:")
st.write("""
1. Matrix optimization methods, including collaborative filtering and content-based filtering, were evaluated. However, the inclusion of metadata features such as genres proved challenging with these methods.

2. A comparative analysis between NN and Singular Value Decomposition (SVD) was conducted. The NN model exhibited superior performance with lower Mean Squared Error (MSE) and Mean Absolute Error (MAE) compared to SVD:

   - NN: 
     - Mean Squared Error: 0.7606360202823306
     - Mean Absolute Error: 0.6751276128960478
   - SVD: 
     - Mean Squared Error: 3.6333589479005948
     - Mean Absolute Error: 1.557864395630317

Thus, the NN model was chosen as it offered better accuracy and flexibility in incorporating metadata features for recommendation tasks.
""")

# Recommendation Demo page
st.header('Recommendation Demo')
movie_recommendation_file = 'movie_recommendations.csv'
movie_recommendations_df = pd.read_csv(movie_recommendation_file)
user_input = st.text_input('Enter your user id:', '')
result_df = movie_recommendations_df[movie_recommendations_df['userId'] == int(user_input)]
if len(result_df) != 0 :
    st.write('Recommended Movies have the following titles:', result_df['title'].iloc[0])
    st.write('With predicted ratings:', result_df['predicted_rating'].iloc[0] )

st.write('Note: Sum error stats are as mentioned above in Recommendation Abstract for the NN model')