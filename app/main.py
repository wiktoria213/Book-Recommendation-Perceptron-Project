import pandas as pd
from perceptron import PerceptronScratch

# wczytanie danych
books = pd.read_csv("app/datasets/Books.csv", encoding="latin-1", sep=";")

ratings = pd.read_csv("app/datasets/Ratings.csv", encoding="latin-1", sep=";")

users = pd.read_csv("app/datasets/Users.csv", encoding="latin-1", sep=";")

# usunięcie użytkowników bez wieku i ograniczenie do wieku 13+
users = users[users["Age"].notnull()]
users = users[users["Age"] >= 13]

# usunięcie rekordów bez ocen
ratings = ratings[ratings["Book-Rating"] != 0]

# podział klas na: polubi, nie polubi
ratings["Liked"] = ratings["Book-Rating"].apply(lambda rating: 1 if rating >= 7 else -1)

# połączenie tabel z danymi: ratings + users
data = ratings.merge(users, on="User-ID")

# utworzenie grup wiekowych
data["Age_Group"] = pd.cut(
    data["Age"],
    bins=[13, 18, 25, 35, 50, 100],
    labels=["13-18", "19-25", "26-35", "36-50", "51+"],
    include_lowest=True,
)

# średnia ocen książki ogólnie
book_features = (
    data.groupby("ISBN")
    .agg(Book_Average_Rating=("Book-Rating", "mean"), Book_Ratings_Count=("Book-Rating", "count"))
    .reset_index()
)

# średnia ocen książki w danej grupie wiekowej
book_age_features = (
    data.groupby(["ISBN", "Age_Group"], observed=True)
    .agg(Book_Average_Rating_By_Age=("Book-Rating", "mean"), Book_Age_Ratings_Count=("Book-Rating", "count"))
    .reset_index()
)

# wykluczenie przypadków, gdzie książka ma zbyt mało ocen w danej grupie wiekowej
reliable_book_age_features = book_age_features[
    book_age_features["Book_Age_Ratings_Count"] > 10
]

# selekcja danych
data = data.merge(book_features, on="ISBN")
data = data.merge(reliable_book_age_features, on=["ISBN", "Age_Group"])

print("\nLiczba rekordów po filtracji:")
print(len(data))

print("\nPodział klas po filtracji danych:")
print(data["Liked"].value_counts())

# losowe wymieszanie rekordów
data = data.sample(frac=1, random_state=42)

# dane wejściowe dla perceptrona
X = data[
    [
        "Age",
        "Book_Average_Rating_By_Age",
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

# rekomendacja dla przykładowego użytkownika
new_user_age = int(input("\nPodaj wiek użytkownika: "))

if new_user_age < 13 or new_user_age > 100:
    print("Wiek użytkownika musi być w zakresie od 13 do 100.")
    exit()

# przypisanie użytkownika do danej grupy wiekowej
new_user_age_group = pd.cut(
    [new_user_age],
    bins=[13, 18, 25, 35, 50, 100],
    labels=["13-18", "19-25", "26-35", "36-50", "51+"],
    include_lowest=True,
)[0]

# zawężenie danych do książek ocenianych przez daną grupę wiekową
recommendation_data = data[data["Age_Group"] == new_user_age_group].copy()

# ustawienie wieku nowego użytkownika
recommendation_data["Age"] = new_user_age

recommendation_data = recommendation_data.merge(
    books[["ISBN", "Book-Title", "Book-Author"]],
    on="ISBN"
)

# dane wejściowe perceptrona dla nowego uzytkownia 
recommendation_x = recommendation_data[
    [
        "Age",
        "Book_Average_Rating_By_Age",
        "Book_Average_Rating",
    ]
].values

recommendation_data["Prediction"] = perceptron.predict(recommendation_x)

# lista książek, które perceptron oznaczył jako 1 czyli "może polubić", posortowane malejąco
recommended_books = recommendation_data[
    recommendation_data["Prediction"] == 1
].sort_values(
    by=["Book_Average_Rating_By_Age", "Book_Average_Rating"],
    ascending=False
)

print("\nPrzykładowy użytkownik:")
print("Age:", new_user_age)
print("Age group:", new_user_age_group)

print("\nProponowana książka:")

if len(recommended_books) > 0:
    selected_book = recommended_books.iloc[0]

    print("Tytuł:", selected_book["Book-Title"])
    print("Autor:", selected_book["Book-Author"])
    print("ISBN:", selected_book["ISBN"])

    print(
        "Średnia ocena w grupie wiekowej:",
        round(selected_book["Book_Average_Rating_By_Age"], 2),
    )

    print(
        "Średnia ocena ogólna:",
        round(selected_book["Book_Average_Rating"], 2),
    )

    print(
        "Liczba ocen w tej grupie wiekowej:",
        selected_book["Book_Age_Ratings_Count"],
    )

else:
    print("Brak rekomendacji dla tej grupy wiekowej.")