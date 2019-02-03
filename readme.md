# Recommendation System

To install dependencies used by this repo, run the following in your terminal

> `pip install requirements.txt`

---

## Introduction

Recommender System is a system that seeks to predict or filter preferences according to the user’s choices. Recommender systems are utilized in a variety of areas including movies, music, news, books, research articles, search queries, social tags, and products in general.

There is 2 types of recommendation systems implemented in this repo

- Collaborative Based
- Content Based

---

### Collaborative Based Recommendation System

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
pearson(X, Y) == pearson(X, 2 \* Y + 3).
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

---

### Content Based Recommendation System

Content based filtering, based on a description of the item and a profile of the user’s preferences. keywords are used to describe the items and a user profile is built to indicate the type of item this user likes. In other words, these algorithms try to recommend items that are similar to those that a user liked in the past (or is examining in the present). In particular, various candidate items are compared with items previously rated by the user and the best-matching items are recommended. This approach has its roots in information retrieval and information filtering research.

To abstract the features of the items in the system, an item presentation algorithm is applied. A widely used algorithm is the tf–idf representation (also called vector space representation).

To create a user profile, the system mostly focuses on two types of information:

1. A model of the user's preference.
2. A history of the user's interaction with the recommender system.

The system creates a content-based profile of users based on a weighted vector of item features. The weights denote the importance of each feature to the user and can be computed from individually rated content vectors using a variety of techniques. Simple approaches use the average values of the rated item vector while other sophisticated methods use machine learning techniques such as Bayesian Classifiers, cluster analysis, decision trees, and artificial neural networks in order to estimate the probability that the user is going to like the item.

Direct feedback from a user, usually in the form of a like or dislike button, can be used to assign higher or lower weights on the importance of certain attributes.

---
