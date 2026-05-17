import pandas as pd
from perceptron import PerceptronScratch

# wczytanie danych
books = pd.read_csv("app/datasets/Books.csv", encoding="latin-1", sep=";")

ratings = pd.read_csv("app/datasets/Ratings.csv", encoding="latin-1", sep=";")

users = pd.read_csv("app/datasets/Users.csv", encoding="latin-1", sep=";")

# usunięcie rekordów bez ocen
ratings = ratings[ratings["Book-Rating"] != 0]

# podział klas na: polubi, nie polubi
ratings["Liked"] = ratings["Book-Rating"].apply(lambda rating: 1 if rating >= 7 else -1)

# połączenie tabel z danymi: ratings+users
data = ratings.merge(users, on="User-ID")

# grupowanie rekordów
user_features = (
    data.groupby("User-ID")
    .agg(User_Ratings_Count=("Book-Rating", "count"), User_Average_Rating=("Book-Rating", "mean"))
    .reset_index()
)

book_features = (
    data.groupby("ISBN")
    .agg(Book_Ratings_Count=("Book-Rating", "count"), Book_Average_Rating=("Book-Rating", "mean"))
    .reset_index()
)

# wykluczenie książek oraz użytkowników poniżej 10 opinii
active_users = user_features[user_features["User_Ratings_Count"] > 10]
popular_books = book_features[book_features["Book_Ratings_Count"] > 10]

# selekcja danych
data = data.merge(active_users, on="User-ID")
data = data.merge(popular_books, on="ISBN")
print("\nLiczba rekordów po filtracji:")
print(len(data))
print("\nPodział klas po filtracji danych:")
print(data["Liked"].value_counts())

# losowy dobór rekordów
data = data.sample(frac=1, random_state=42)

# dane wejściowe dla perceptrona
X = data[
    [
        "User_Average_Rating",
        "Book_Ratings_Count",
        "Book_Average_Rating",
    ]
].values


y = data["Liked"].values

# podział danych na treningowe i testowe w proporcjach 80/20
split_index = int(0.8 * len(X))

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

print("\nLiczba rekordów treningowych:")
print(len(X_train))

print("\nLiczba rekordów testowych:")
print(len(X_test))

perceptron = PerceptronScratch(lr=0.01, epochs=50)

# wywołanie funkcji do nauki perceptrona
perceptron.fit(X_train, y_train)

# sprawdzenie predykcji i dokładności modelu
predictions = perceptron.predict(X_test)
accuracy = (predictions == y_test).mean()

print("\nWagi perceptronu:")
print(perceptron.w)

print("\nBias:")
print(perceptron.b)

print("\nBłędy w epokach:")
print(perceptron.errors_per_epoch)

print("\nAccuracy:")
print(round(accuracy, 4))

# zliczenie poprawnych oraz błędnych predykcji perceptronu
true_positive = ((predictions == 1) & (y_test == 1)).sum()
true_negative = ((predictions == -1) & (y_test == -1)).sum()
false_positive = ((predictions == 1) & (y_test == -1)).sum()
false_negative = ((predictions == -1) & (y_test == 1)).sum()

print("\nMacierz pomyłek:")
print("Poprawnie przewidziane - polubi:", true_positive)
print("Poprawnie przewidziane - nie polubi:", true_negative)
print("Błędnie przewidziane - polubi:", false_positive)
print("Błędnie przewidziane - nie polubi:", false_negative)
