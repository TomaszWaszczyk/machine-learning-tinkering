"""Lesson 4 - the same thing with scikit-learn.

Run it with:

    python 04_scikit_learn.py

MLPClassifier ("multi-layer perceptron") is scikit-learn's neural network.
It builds the same kind of network as lesson 3: ReLU hidden layers, a softmax
output, cross-entropy loss, mini-batch training. The numbers differ slightly
because it uses a different split, a different random start and a smarter
optimiser (Adam, which adapts the learning rate for every weight).

scikit-learn is fine for small networks on tabular data. For anything larger
(images, text, GPUs) people use PyTorch, where you describe the forward pass
and the library derives the backward pass for you.
"""

from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

digits = load_digits()
X, y = digits.data / 16.0, digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"layers: {[X.shape[1], *model.hidden_layer_sizes, len(model.classes_)]}")
print(f"stopped after {model.n_iter_} epochs")
print(f"test accuracy = {accuracy_score(y_test, y_pred):.1%}\n")

print("confusion matrix (row = true digit, column = predicted digit)")
print(confusion_matrix(y_test, y_pred))

# predict_proba returns the 10 softmax outputs for each image.
probabilities = model.predict_proba(X_test[:1])[0]
print(f"\nfirst test image is a {y_test[0]}; the network's probabilities:")
for digit, p in enumerate(probabilities):
    print(f"  {digit}: {p:6.1%} {'#' * round(p * 30)}")
