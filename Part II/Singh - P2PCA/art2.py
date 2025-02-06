import numpy as np

class ART2:
    def __init__(self, num_features, vigilance=0.5, learning_rate=0.1, max_epochs=10, max_clusters=None):
        self.vigilance = vigilance
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.num_features = num_features
        self.weights = []
        self.categories = 0
        self.max_clusters = max_clusters

    def normalize(self, vector):
        norm = np.linalg.norm(vector)
        return vector / norm if norm != 0 else vector

    def train(self, data):
        for epoch in range(self.max_epochs):
            for sample in data:
                sample = self.normalize(sample)
                if not self.weights:
                    # Initialize first category
                    self.weights.append(sample.copy())
                    self.categories += 1
                    continue

                # Calculate similarities
                similarities = [np.dot(sample, w) / (np.linalg.norm(w) + 1e-6) for w in self.weights]
                index = np.argmax(similarities)
                max_similarity = similarities[index]

                # Check vigilance criterion
                if max_similarity >= self.vigilance:
                    # Update weights
                    self.weights[index] = (1 - self.learning_rate) * self.weights[index] + self.learning_rate * sample
                    self.weights[index] = self.normalize(self.weights[index])
                else:
                    # Check if max_clusters is reached
                    if self.max_clusters is None or self.categories < self.max_clusters:
                        # Create new category
                        self.weights.append(sample.copy())
                        self.categories += 1
                    else:
                        # Assign to the closest existing category and update weights
                        self.weights[index] = (1 - self.learning_rate) * self.weights[index] + self.learning_rate * sample
                        self.weights[index] = self.normalize(self.weights[index])

    def predict(self, data):
        labels = []
        for sample in data:
            sample = self.normalize(sample)
            similarities = [np.dot(sample, w) / (np.linalg.norm(w) + 1e-6) for w in self.weights]
            index = np.argmax(similarities)
            labels.append(index)
        return np.array(labels)