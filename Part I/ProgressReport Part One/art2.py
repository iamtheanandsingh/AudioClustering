class Art2:
    def __init__(self, max_clusters, vigilance_threshold, creation_buffer_size):
        self.max_clusters = max_clusters
        self.vigilance_threshold = vigilance_threshold
        self.creation_buffer_size = creation_buffer_size
        self.num_clusters = 0
        self.cluster_weights = []
        self.cluster_creation_buffer = 0
        return

    # Compute Manhattan distance to already existing clusters
    def distance_2_clusters(self, input_features):
        distances = np.sum(
            np.abs(self.cluster_weights - input_features), axis=1)
        return distances

    def process_new_sample(self, input_features):
        if self.num_clusters > 0:
            distances = self.distance_2_clusters(input_features)

            best_cluster_index = np.argmin(distances)
            best_distance = distances[best_cluster_index]

            if best_distance <= self.vigilance_threshold:
                self.update_cluster(best_cluster_index, input_features)
                self.cluster_creation_buffer = 0
            else:
                self.cluster_creation_buffer += 1

        if self.cluster_creation_buffer >= self.creation_buffer_size:
            if self.num_clusters < self.max_clusters:
                self.create_new_cluster(input_features)
            else:
                print("Maximum number of clusters reached.")
            self.cluster_creation_buffer = 0

    def create_new_cluster(self, input_features):
        self.cluster_weights.append(input_features)
        self.num_clusters += 1

    def update_cluster(self, cluster_index, input_features):
        # Simple averaging weight update; you might want to use a more complex method
        self.cluster_weights[cluster_index] = (
            self.cluster_weights[cluster_index] + input_features
        ) / 2

    def cluster_data_as_stream():
        return
