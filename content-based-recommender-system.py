import pandas as pd

# Storing the movie information into a pandas data-frame
movies_df = pd.read_csv('movies.csv')
# Storing the user information into a pandas data-frame
ratings_df = pd.read_csv('ratings.csv')
# Head is a function that gets the first N rows of a data-frame. default is 5.
print(movies_df.head())

# Using regular expressions to find a year stored between parentheses
# We specify the parentheses so we don't conflict with movies that have
# years in their titles
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))', expand=False)

# Removing the parentheses
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)', expand=False)

# Removing the years from the 'title' column
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')

# Applying the strip function to get rid of any ending whitespace
# characters that may have appeared
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
print(movies_df.head())

# Every genre is separated by a | so we simply have to call the split function on |
movies_df['genres'] = movies_df.genres.str.split('|')
print(movies_df.head())

# Copying the movie dataframe into a new one since we won't need to use
# the genre information in our first case.
moviesWithGenres_df = movies_df.copy()

# For every row in the dataframe, iterate through the list of genres
# and place a 1 into the corresponding column
for index, row in movies_df.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index, genre] = 1
# Filling in the NaN values with 0 to show that a movie
# doesn't have that column's genre
moviesWithGenres_df = moviesWithGenres_df.fillna(0)
moviesWithGenres_df.head()

# Take a look on ratings
print(ratings_df.head())

# won't be needing the timestamp column, so let's drop it to save on memory
# Drop removes a specified row or column from a dataframe
ratings_df = ratings_df.drop('timestamp', 1)
print(ratings_df.head())

# Let's input user ratings matrix
userInput = [
    {'title': 'Breakfast Club, The', 'rating': 5},
    {'title': 'Toy Story', 'rating': 3.5},
    {'title': 'Jumanji', 'rating': 2},
    {'title': "Pulp Fiction", 'rating': 5},
    {'title': 'Akira', 'rating': 4.5}
]
inputMovies = pd.DataFrame(userInput)
print(inputMovies)

# Add movie ID to user input
# Filtering out the movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
# Then merging it so we can get the movieId.
# It's implicitly merging it by title.
inputMovies = pd.merge(inputId, inputMovies)
# Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('genres', 1).drop('year', 1)
# Final input dataframe
# If a movie you added in above isn't here,
# then it might not be in the original
# dataframe or it might spelled differently,
# please check capitalization.
print(inputMovies)

# Filtering out the movies from the input, that the user has watched
userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(
    inputMovies['movieId'].tolist())]
print(userMovies)

# only need the actual genre table, so let's clean this up a bit
# by resetting the index and dropping the movieId, title, genres & year columns
# Resetting the index to avoid future issues
userMovies = userMovies.reset_index(drop=True)
# Dropping unnecessary issues due to save memory and to avoid issues
userGenreTable = userMovies.drop('movieId', 1).drop(
    'title', 1).drop('genres', 1).drop('year', 1)
print(userGenreTable)

# Now we're ready to start learning the input's preferences!

# we're going to turn each genre into weights.
# We can do this by using the input's reviews and multiplying them
# into the input's genre table and then summing up the resulting table
# by column. This operation is actually a dot product between a matrix
# and a vector, so we can simply accomplish by calling Pandas's "dot" function
print(inputMovies['rating'])

# Dot product to get weights
userProfile = userGenreTable.transpose().dot(inputMovies['rating'])
# The user profile
print(userProfile)

# Now, we have the weights for every of the user's preferences.
# known as the User Profile.
# Using this, we can recommend movies that satisfy the user's preferences.

# Let's start by extracting the genre table from the original dataframe:
# Now let's get the genres of every movie in our original dataframe
genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
# And drop the unnecessary information
genreTable = genreTable.drop('movieId', 1).drop(
    'title', 1).drop('genres', 1).drop('year', 1)
print(genreTable.head())
print(genreTable.shape)

# With the input's profile and the complete list of movies and their
# genres in hand, we're going to take the weighted average of every movie
# based on the input profile and recommend the top twenty movies
# that most satisfy it.
# Multiply the genres by the weights and then take the weighted average
recommendationTable_df = (
    (genreTable*userProfile).sum(axis=1))/(userProfile.sum())
print(recommendationTable_df.head())

# Sort our recommendations in descending order
recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
# Just a peek at the values
print(recommendationTable_df.head())

# The final recommendation table
movies_df.loc[movies_df['movieId'].isin(
    recommendationTable_df.head(20).keys())]
