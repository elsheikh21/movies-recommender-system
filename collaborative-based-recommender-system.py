import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

# Storing the movie information into a pandas dataframe
movies_df = pd.read_csv('movies.csv')
# Storing the user information into a pandas dataframe
ratings_df = pd.read_csv('ratings.csv')

# Head is a function that gets the first N rows of a dataframe. N's default is 5.
print(movies_df.head())

# Clean the data by removing unwanted columns

# Using regular expressions to find a year stored between parentheses
# We specify the parentheses so we don't conflict with movies
# that have years in their titles
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))', expand=False)
# Removing the parentheses
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)', expand=False)
# Removing the years from the 'title' column
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
# Applying the strip function to get rid of any ending whitespace
# characters that may have appeared
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
# Dropping the genres column
movies_df = movies_df.drop('genres', 1)

# Show cleaned movies data-frame
print(movies_df.head())

# Show ratings dataset
print(ratings_df.head())

# Clean the data, by removing unwanted columns

# Drop removes a specified row or column from a dataframe
ratings_df = ratings_df.drop('timestamp', 1)

# Show the cleaned ratings data-frame
print(ratings_df.head())

'''
Collaborative Filtering, which is also known as User-User Filtering.
As hinted by its alternate name, this technique uses other users to recommend
items to the input user.
It attempts to find users that have similar preferences and opinions as
the input and then recommends items that they have liked to the input.
There are several methods of finding similar users (Even some making use of ML),
and the one we will be using here is going to be based on the
Pearson Correlation Function.


Pearson Correlation Coefficient. It is used to measure
the strength of a linear association between two variables.
The formula for finding this coefficient between sets X and Y 
with N values.

Why Pearson Correlation?

Pearson correlation is invariant to scaling,
i.e. multiplying all elements by a nonzero constant or
adding any constant to all elements.
For example, if you have two vectors X and Y,then,
pearson(X, Y) == pearson(X, 2 * Y + 3).
This is a pretty important property in recommendation systems because
for example two users might rate two series of items totally different in
terms of absolute rates,
but they would be similar users (i.e. with similar ideas)
with similar rates in various scales .



The process for creating a User Based recommendation system is as follows:

1. Select a user with the movies the user has watched
2. Based on his rating to movies, find the top X neighbors
3. Get the watched movie record of the user for each neighbor.
4. Calculate a similarity score using some formula
5. Recommend the items with the highest score
'''

# begin by creating an input user to recommend movies to:
userInput = [
    {'title': 'Breakfast Club, The', 'rating': 5},
    {'title': 'Toy Story', 'rating': 3.5},
    {'title': 'Jumanji', 'rating': 2},
    {'title': "Pulp Fiction", 'rating': 5},
    {'title': 'Akira', 'rating': 4.5}
]
inputMovies = pd.DataFrame(userInput)
print(inputMovies)


# Add movieId to input user

# Filtering out the movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
# Then merging it so we can get the movieId.
# It's implicitly merging it by title.
inputMovies = pd.merge(inputId, inputMovies)
# Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('year', 1)
# Final input dataframe
# If a movie you added in above isn't here,
# then it might not be in the original
# dataframe or it might spelled differently,
# please check capitalization.
print(inputMovies)

# The users who has seen the same movies
# Now with the movie ID's in our input, we can now get the subset of users
# that have watched and reviewed the movies in our input
# Filtering out users that have watched movies
# that the input has watched & storing it
userSubset = ratings_df[ratings_df['movieId'].isin(
    inputMovies['movieId'].tolist())]
print(userSubset.head())

# Group rows by user ID
# Group-by creates several sub data-frames where they all
# have the same value in the column specified as the parameter
userSubsetGroup = userSubset.groupby(['userId'])

# Let's look at user 1130
print(userSubsetGroup.get_group(1130))


# ort these groups so the users that share the most movies in common with the
# input have higher priority. This provides a richer recommendation since
# we won't go through every single user.

# Sorting it so users with movie most in common
# with the input will have priority
userSubsetGroup = sorted(
    userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)

# take a look on the first 3 users
print(userSubsetGroup[0:3])

# Select a subset, so it wont take too much time running
userSubsetGroup = userSubsetGroup[0:100]

# Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient
pearsonCorrelationDict = {}

# we calculate the Pearson Correlation between input user and subset group,
# and store it in a dictionary, where the key is the user Id
# and the value is the coefficient
# For every user group in our subset
for name, group in userSubsetGroup:
    # Let's start by sorting the input and current user group
    # so the values aren't mixed up later on
    group = group.sort_values(by='movieId')
    inputMovies = inputMovies.sort_values(by='movieId')
    # Get the N for the formula
    nRatings = len(group)
    # Get the review scores for the movies that they both have in common
    temp_df = inputMovies[inputMovies['movieId'].isin(
        group['movieId'].tolist())]
    # And then store them in a temporary buffer variable in a list
    # format to facilitate future calculations
    tempRatingList = temp_df['rating'].tolist()
    # Let's also put the current user group reviews in a list format
    tempGroupList = group['rating'].tolist()
    # Now let's calculate the pearson correlation between two users,
    # so called, x and y
    Sxx = sum([i**2 for i in tempRatingList]) - \
        pow(sum(tempRatingList), 2)/float(nRatings)
    Syy = sum([i**2 for i in tempGroupList]) - \
        pow(sum(tempGroupList), 2)/float(nRatings)
    Sxy = sum(i*j for i, j in zip(tempRatingList, tempGroupList)) - \
        sum(tempRatingList)*sum(tempGroupList)/float(nRatings)

    # If the denominator is different than zero, then divide,
    # else, 0 correlation.
    if Sxx != 0 and Syy != 0:
        pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
    else:
        pearsonCorrelationDict[name] = 0

print(pearsonCorrelationDict.items())

# Print out similarity index with their corresponding IDs
pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['userId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))
print(pearsonDF.head())

# get the top 50 users that are most similar to the input.
topUsers = pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
print(topUsers.head())

# let's start recommending movies to the input user.

'''
Rating of selected users to all movies
We're going to do this by taking the weighted average of the ratings of the
movies using the Pearson Correlation as the weight.
But to do this, we first need to get the movies watched by the users
in our pearsonDF from the ratings dataframe and then store their
correlation in a new column called _similarityIndex".
This is achieved below by merging of these two tables.
'''
topUsersRating = topUsers.merge(
    ratings_df, left_on='userId', right_on='userId', how='inner')
print(topUsersRating.head())

'''
Now all we need to do is simply multiply the movie rating by its weight
(The similarity index), then sum up the new ratings & divide it
by the sum of the weights.

We can easily do this by simply multiplying two columns,
then grouping up the dataframe by movieId and then dividing two columns:

It shows the idea of all similar users to candidate movies for the input user:
'''
# Multiplies the similarity by the user's ratings
topUsersRating['weightedRating'] = topUsersRating['similarityIndex'] * \
    topUsersRating['rating']
print(topUsersRating.head())

# Applies a sum to the topUsers after grouping it up by userId
tempTopUsersRating = topUsersRating.groupby(
    'movieId').sum()[['similarityIndex', 'weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex', 'sum_weightedRating']
print(tempTopUsersRating.head())

# Creates an empty dataframe
recommendation_df = pd.DataFrame()
# Now we take the weighted average
recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating'] / \
    tempTopUsersRating['sum_similarityIndex']
recommendation_df['movieId'] = tempTopUsersRating.index
print(recommendation_df.head())


recommendation_df = recommendation_df.sort_values(
    by='weighted average recommendation score', ascending=False)
print(recommendation_df.head(10))
