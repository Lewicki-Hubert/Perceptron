def load_data(filepath):
    dataset = []
    with open(filepath, 'r') as file:
        for line in file:
            # Zamiana przecinków na kropki i rozdzielenie linii
            row = line.replace(',', '.').strip().split()
            # Konwersja atrybutów na float, etykieta pozostaje jako string
            row = [float(attr) if i < len(row) - 1 else attr for i, attr in enumerate(row)]
            dataset.append(row)
    return dataset


# Funkcja inicjalizująca wagi perceptronu
def initialize_weights(num_attributes):
    weights = [0.0 for i in range(num_attributes + 1)]  # +1 dla biasu
    return weights


# Funkcja aktywacji perceptronu
def predict(row, weights):
    activation = weights[0]  # bias
    for i in range(len(row) - 1):  # Ostatni element to etykieta
        activation += weights[i + 1] * row[i]  # ważona suma cech (bias)
    return 1.0 if activation >= 0.0 else 0.0


# Algorytm delty (trenowanie perceptronu)
def train_weights(train, l_rate, n_epoch):
    weights = initialize_weights(len(train[0]))
    for epoch in range(n_epoch):
        for row in train:
            prediction = predict(row, weights)
            error = (1 if row[-1] == 'Iris-setosa' else 0) - prediction
            weights[0] = weights[0] + l_rate * error  # Aktualizacja biasu
            for i in range(len(row) - 1):  # Aktualizacja wag na podstawie błędu i współczynnika uczenia
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
    return weights


# Funkcja do testowania perceptronu
def perceptron(train, test, l_rate, n_epoch):
    predictions = []
    weights = train_weights(train, l_rate, n_epoch)
    for row in test:
        prediction = predict(row, weights)
        predictions.append(prediction)
    return predictions  # zwraca liste predykcji dla wszystkich wierszy danych testowych.


def user_interface():
    while True:
        # Pytanie o chęć przeprowadzenia nowego eksperymentu
        if input("Czy chcesz przeprowadzić nowy eksperyment? (tak/nie): ").lower() != 'tak':
            break

        # Wczytywanie danych treningowych
        train_data_path = input("Podaj ścieżkę do pliku z danymi treningowymi: ")
        train = load_data(train_data_path)

        # Wczytywanie danych testowych
        test_data_path = input("Podaj ścieżkę do pliku z danymi testowymi: ")
        test = load_data(test_data_path)

        # Ponowne określenie parametrów perceptronu za każdym razem
        l_rate = float(input("Podaj współczynnik uczenia (np. 0.01): "))
        n_epoch = int(input("Podaj liczbę epok: "))

        # Trenowanie perceptronu
        weights = train_weights(train, l_rate, n_epoch)
        print("Perceptron został wytrenowany.")

        # Testowanie perceptronu na zbiorze testowym
        correct = 0
        for row in test:
            prediction = predict(row, weights)
            correct += ((prediction == 1 and row[-1] == 'Iris-setosa') or (
                    prediction == 0 and row[-1] != 'Iris-setosa'))
        accuracy = correct / float(len(test)) * 100.0
        print(f'Liczba prawidłowo zaklasyfikowanych przykładów: {correct}/{len(test)}')
        print(f'Dokładność: {accuracy:.2f}%')

        while True:
            # Wprowadzanie wektorów atrybutów przez użytkownika
            test_row_input = input(
                "Wprowadź wektor atrybutów oddzielony spacjami (bez etykiety) lub wpisz 'koniec', aby zakończyć: ")
            if test_row_input.lower() == 'koniec':
                break
            test_row = [float(attr) for attr in test_row_input.split()]
            prediction = predict([*test_row, ''], weights)  # Dodanie pustej etykiety dla spójności
            predicted_class = 'Iris-setosa' if prediction == 1 else 'Iris-nie-setosa'
            print(f'Przewidziana klasa dla wprowadzonego wektora: {predicted_class}')


user_interface()
