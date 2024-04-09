import numpy as np

weights_dict = {
    'S': np.array([1, 0, 0, 0, 0]),
    'P': np.array([0, 0, 0, 0, 0]),
    'T': np.array([0, 0, 0, 0, 0])
}

feature_vectors = [
    np.array([1, 1, 0, 1, 1]),
    np.array([1, 1, 0, 0, 1]),
    np.array([1, 1, 1, 0, 1])
]
true_labels = ['P', 'P', 'S']

def update_weights(true_label, predicted_label, feature_vector, weights):
    if predicted_label != true_label:
        weights[true_label] += feature_vector
        weights[predicted_label] -= feature_vector

def run_multiclass_perceptron(feature_vectors, true_labels, weights):
    for index, feature_vector in enumerate(feature_vectors):
        scores = {label: np.dot(weights[label], feature_vector) for label in weights}
        prediction = max(scores, key=scores.get)
        
        update_weights(true_labels[index], prediction, feature_vector, weights)
        
        print(f"After sample {index + 1} with true label '{true_labels[index]}':")
        for label in weights:
            print(f"Weight vector for class {label}: {weights[label]}")

run_multiclass_perceptron(feature_vectors, true_labels, weights_dict)
