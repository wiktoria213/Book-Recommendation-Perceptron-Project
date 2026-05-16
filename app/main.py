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

ratings["Liked"] = ratings["Book-Rating"].apply(
    lambda rating: 1 if rating >= 7 else -1)

print("\nLIKED COUNTS")
print(ratings["Liked"].value_counts())


data = ratings.merge(users, on="User-ID")


user_features = data.groupby("User-ID").agg(
    User_Ratings_Count=("Book-Rating", "count"),
    User_Average_Rating=("Book-Rating", "mean")).reset_index()


book_features = data.groupby("ISBN").agg(
    Book_Ratings_Count=("Book-Rating", "count"),
    Book_Average_Rating=("Book-Rating", "mean")).reset_index()


active_users = user_features[user_features["User_Ratings_Count"] > 10]
popular_books = book_features[book_features["Book_Ratings_Count"] > 10]

data = data.merge(active_users, on="User-ID")
data = data.merge(popular_books, on="ISBN")

print("\nDATA WITH FEATURES")
print(data[
    [
        "ISBN",
        "Book_Ratings_Count",
        "Book_Average_Rating",
        "User-ID",
        "Book-Rating",
        "User_Ratings_Count",
        "User_Average_Rating",
        "Liked",
        
    ]
].head(20))


#dane wejściowe dla perceptrona
X = data[
    [
        "User_Average_Rating",
        "Book_Ratings_Count",
        "Book_Average_Rating",
    ]
].values


y = data["Liked"].values

perceptron = PerceptronScratch(
    lr=0.01,
    epochs=50)

perceptron.fit(X, y)

print("\nWagi perceptronu:")
print(perceptron.w)

print("\nBias:")
print(perceptron.b)

print("\nBłędy w epokach:")
print(perceptron.errors_per_epoch)