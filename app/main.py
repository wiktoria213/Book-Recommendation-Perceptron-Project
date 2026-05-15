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

print("\nBOOKS")
print(books.head())
print(books.columns)

print("\nRATINGS")
print(ratings.head())
print(ratings.columns)

print("\nUSERS")
print(users.head())
print(users.columns)
