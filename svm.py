from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# Load Iris dataset
irisData = load_iris()

X = irisData.data
Y = irisData.target


# Split into training and testing
X_train_full, X_test, Y_train_full, Y_test = train_test_split(
    X, Y,
    test_size=0.3,
    random_state=42
)


# Split training data into training and validation
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train_full, Y_train_full,
    test_size=0.2,
    random_state=42
)

print("Training data size:", len(Y_train))


# Find the best C parameter
def get_best_para(X_train, X_val, Y_train, Y_val):

    C_range = [2.0 ** i for i in range(-2, 13, 1)]

    best_score = -1
    best_c = None

    for C in C_range:

        svc = SVC(kernel='linear', C=C)

        model = OneVsOneClassifier(svc)

        model.fit(X_train, Y_train)

        val_score = model.score(X_val, Y_val)

        print(f"Parameter: {C}")
        print(f"Validation accuracy: {val_score:.4f}")

        if val_score > best_score:
            best_score = val_score
            best_c = C

    print(f"Best Parameter: {best_c}")
    print(f"Best Validation accuracy: {best_score:.4f}")

    return best_c


# Train final model
def train_final_model(X_trainval, Y_trainval, best_c):

    model = OneVsOneClassifier(
        SVC(kernel='linear', C=best_c)
    )

    model.fit(X_trainval, Y_trainval)

    return model


# Evaluate model
def evaluate_model(model, X_test, Y_test):

    Y_pred = model.predict(X_test)

    acc = accuracy_score(Y_test, Y_pred)

    print(f"Test accuracy: {acc:.4f}")

    print("Classification report:")

    print(classification_report(Y_test, Y_pred))


# Find best C
best_c = get_best_para(
    X_train,
    X_val,
    Y_train,
    Y_val
)


# Train final model using full training data
final_model = train_final_model(
    X_train_full,
    Y_train_full,
    best_c
)


# Evaluate final model
evaluate_model(
    final_model,
    X_test,
    Y_test
)
