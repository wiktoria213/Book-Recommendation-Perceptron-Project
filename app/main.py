import pandas as pd
from perceptron import PerceptronScratch

books = pd.read_csv(
    "app/datasets/Books.csv",
    encoding="latin-1",
    sep=";"
)

ratings = pd.read_csv(
    "app/datasets/Ratings.csv",
    encoding="latin-1",
    sep=";"
)

users = pd.read_csv(
    "app/datasets/Users.csv",
    encoding="latin-1",
    sep=";"
)

ratings = ratings[ratings["Book-Rating"] != 0]
users = users[users["Age"].notnull()]

ratings["Liked"] = ratings["Book-Rating"].apply(
    lambda rating: 1 if rating >= 7 else -1)

print("\nRATINGS")
print(ratings[["User-ID", "ISBN", "Book-Rating", "Liked"]].head(20))

print(ratings["Liked"].value_counts())

data = ratings.merge(users, on="User-ID")
user_rating_counts = data.groupby("User-ID").size().reset_index(name="Ratings-Count")

active_users = user_rating_counts[user_rating_counts["Ratings-Count"] > 1]

data = data[data["User-ID"].isin(active_users["User-ID"])]

user_features = users.merge(active_users, on="User-ID")

print("\nUSER FEATURES")
print(user_features[["User-ID", "Age", "Ratings-Count"]].head(20))
X = data[["Age", "Book-Rating"]].values
y = data["Liked"].values

perceptron = PerceptronScratch(
    lr=0.01,
    epochs=10)

perceptron.fit(X, y)

print("\nWagi perceptronu:")
print(perceptron.w)

print("\nBias:")
print(perceptron.b)

print("\nBłędy w epokach:")
print(perceptron.errors_per_epoch)