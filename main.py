import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


n = 2708


def create_adjacency_matrix(graph_file_path):
    with open(graph_file_path, 'r') as file:
        edges = file.readlines()
    
    # Determine the size of the matrix
    max_node = 0
    edge_list = []
    for edge in edges:
        node1, node2 = map(int, edge.strip().split(','))
        edge_list.append((node1, node2))
        max_node = max(max_node, node1, node2)
    
    # Initialize the adjacency matrix
    adjacency_matrix = np.zeros((max_node + 1, max_node + 1), dtype=int)
    
    # Fill the adjacency matrix
    for node1, node2 in edge_list:
        adjacency_matrix[node1][node2] = 1
        adjacency_matrix[node2][node1] = 1  # Since the graph is undirected
    print(adjacency_matrix)
    return adjacency_matrix


def create_degree_matrix(adjacency_matrix):
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    return degree_matrix


def compute_laplacian_matrix(adjacency_matrix):
    degree_matrix = create_degree_matrix(adjacency_matrix)
    laplacian_matrix = degree_matrix - adjacency_matrix
    return laplacian_matrix


def compute_normalized_laplacian_matrix(adjacency_matrix):
    degree_matrix = create_degree_matrix(adjacency_matrix)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(degree_matrix)))
    identity_matrix = np.eye(adjacency_matrix.shape[0])
    normalized_laplacian_matrix = identity_matrix - d_inv_sqrt @ adjacency_matrix @ d_inv_sqrt
    return normalized_laplacian_matrix


def compute_embedding(laplacian_matrix, dimensions=1):
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigh(laplacian_matrix)
    
    # Find the indices of the smallest non-zero eigenvalues
    non_zero_eigenvalues = eigenvalues[eigenvalues > 1e-10]
    smallest_non_zero_eigenvalue_indices = np.where(np.isin(eigenvalues, non_zero_eigenvalues[:dimensions]))[0]
    
    # Get the corresponding eigenvectors
    embedding = eigenvectors[:, smallest_non_zero_eigenvalue_indices]
    return embedding


def visualize_embedding_1d(embedding, labels):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(embedding, np.zeros_like(embedding), c=labels, cmap='viridis', s=50)
    plt.colorbar(scatter, label='Class Label')
    plt.xlabel('One-dimensional embedding')
    plt.title('One-dimensional embedding of nodes with class labels')
    plt.show()


def visualize_embedding_2d(embedding, labels):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis', s=50)
    plt.colorbar(scatter, label='Class Label')
    plt.xlabel('First dimension')
    plt.ylabel('Second dimension')
    plt.title('Two-dimensional embedding of nodes with class labels')
    plt.show()


def train_classifier(embedding, labels):
    class_intervals = {}
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        if embedding.ndim == 1:
            class_intervals[label] = (np.min(embedding[labels == label]), np.max(embedding[labels == label]))
        else:
            class_intervals[label] = [(np.min(embedding[labels == label, dim]), np.max(embedding[labels == label, dim])) for dim in range(embedding.shape[1])]
    
    return class_intervals


def predict(class_intervals, embedding):
    predictions = []
    for value in embedding:
        predicted = False
        for label, intervals in class_intervals.items():
            if embedding.ndim == 1:
                low, high = intervals
                if low <= value <= high:
                    predictions.append(label)
                    predicted = True
                    break
            else:
                if all(low <= val <= high for (low, high), val in zip(intervals, value)):
                    predictions.append(label)
                    predicted = True
                    break
        if not predicted:
            predictions.append(-1)  # Assign a default class if no interval matches
    return predictions


def evaluate_classifier(embedding, labels, num_splits=5, test_size=0.2):
    accuracies = []
    
    for _ in range(num_splits):
        X_train, X_test, y_train, y_test = train_test_split(embedding, labels, test_size=test_size, random_state=None)
        
        class_intervals = train_classifier(X_train, y_train)
        y_pred = predict(class_intervals, X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    
    mean_accuracy = np.mean(accuracies)
    variance_accuracy = np.var(accuracies)
    
    return mean_accuracy, variance_accuracy


def row_major(graph_file_path):
    with open(graph_file_path, 'r') as file:
        graph = file.readlines()
    # print(graph)
    graph = [list(map(int, line.strip().split(','))) for line in graph]
    # print(graph)
    # print(len(graph))
    r = [0] * (n+1)
    c = []
    v = []
    last = -1
    i = 0
    for line in graph:
        if line[1] > last:
            r[line[1]] = i
            j = line[1] - last - 1
            for k in range(j):
                r[last + k + 1] = i
            last = line[1]
        i += 1
        c.append(line[0])
        v.append(1)
    r[n] = i
    # print(r)
    # print(c)
    
    return r, c, v


def vm_mult_sparse(r, c, v, vector):
    result = [0] * n
    for i in range(len(r)-1):
        for j in range(r[i], r[i+1]):
            result[i] += v[j] * vector[c[j]]
    return result


def vm_mult_dense(m, v):
    return np.dot(m, v)


def create_random_vector():
    return np.random.rand(n)


def assess_runtime(r, c, v, dense_matrix, num_tests=100):
    sparse_times = []
    dense_times = []
    
    for _ in range(num_tests):
        vector = create_random_vector()
        
        # Time sparse matrix-vector multiplication
        start_time = time.time()
        sparse = vm_mult_sparse(r, c, v, vector)
        sparse_times.append(time.time() - start_time)
        
        # Time dense matrix-vector multiplication
        start_time = time.time()
        dense = vm_mult_dense(dense_matrix, vector)
        dense_times.append(time.time() - start_time)
        if not np.allclose(sparse, dense):
            print("Error: Results do not match!")
            print(sparse)
            print(dense)
            break
    
    sparse_mean = np.mean(sparse_times)
    sparse_variance = np.var(sparse_times)
    
    dense_mean = np.mean(dense_times)
    dense_variance = np.var(dense_times)
    
    return {
        'sparse_mean': sparse_mean,
        'sparse_variance': sparse_variance,
        'dense_mean': dense_mean,
        'dense_variance': dense_variance
    }


def visualize_results(results):
    labels = ['Sparse', 'Dense']
    means = [results['sparse_mean'], results['dense_mean']]
    variances = [results['sparse_variance'], results['dense_variance']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot mean runtimes
    ax[0].bar(x, means, width, label='Mean Runtime')
    ax[0].set_ylabel('Time (seconds)')
    ax[0].set_title('Mean Runtime of Matrix-Vector Multiplication')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels)
    
    # Plot variances
    ax[1].bar(x, variances, width, label='Variance of Runtime')
    ax[1].set_ylabel('Time^2 (seconds^2)')
    ax[1].set_title('Variance of Runtime of Matrix-Vector Multiplication')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels)
    
    plt.tight_layout()
    plt.show()

def visualize_comparison2(results):
    labels = ['2D', '5D', '10D']
    means = [results['mean_2d'], results['mean_5d'], results['mean_10d']]
    variances = [results['variance_2d'], results['variance_5d'], results['variance_10d']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot mean accuracies
    ax[0].bar(x, means, width, label='Mean Accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_title('Mean Accuracy of Classifier')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels, rotation=45, ha='right')
    
    # Plot variances
    ax[1].bar(x, variances, width, label='Variance of Accuracy')
    ax[1].set_ylabel('Variance')
    ax[1].set_title('Variance of Accuracy of Classifier')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()

def visualize_comparison(results):
    labels = ['Unnormalized Laplacian 1D', 'Normalized Laplacian 1D', 'Unnormalized Laplacian 2D', 'Normalized Laplacian 2D']
    means = [results['unnormalized_mean_1d'], results['normalized_mean_1d'], results['unnormalized_mean_2d'], results['normalized_mean_2d']]
    variances = [results['unnormalized_variance_1d'], results['normalized_variance_1d'], results['unnormalized_variance_2d'], results['normalized_variance_2d']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot mean accuracies
    ax[0].bar(x, means, width, label='Mean Accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_title('Mean Accuracy of Classifier')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels, rotation=45, ha='right')
    
    # Plot variances
    ax[1].bar(x, variances, width, label='Variance of Accuracy')
    ax[1].set_ylabel('Variance')
    ax[1].set_title('Variance of Accuracy of Classifier')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()


def main():
    graph_file_path = 'graph.csv'
    labels_file_path = 'nodelabels.csv'
    
    adjacency_matrix = create_adjacency_matrix(graph_file_path)
    laplacian_matrix = compute_laplacian_matrix(adjacency_matrix)
    
    # Load the labels
    with open(labels_file_path, 'r') as file:
        labels = np.array([int(line.strip()) for line in file.readlines()])
    
    # Compute embeddings for k = 2, 5, 10
    embedding_2d = compute_embedding(laplacian_matrix, dimensions=2)
    embedding_5d = compute_embedding(laplacian_matrix, dimensions=5)
    embedding_10d = compute_embedding(laplacian_matrix, dimensions=10)
    
    # Evaluate classifiers
    mean_accuracy_2d, variance_accuracy_2d = evaluate_classifier(embedding_2d, labels)
    mean_accuracy_5d, variance_accuracy_5d = evaluate_classifier(embedding_5d, labels)
    mean_accuracy_10d, variance_accuracy_10d = evaluate_classifier(embedding_10d, labels)
    
    print("2D Embedding:")
    print(f"Mean accuracy: {mean_accuracy_2d}")
    print(f"Variance of accuracy: {variance_accuracy_2d}")
    print("5D Embedding:")
    print(f"Mean accuracy: {mean_accuracy_5d}")
    print(f"Variance of accuracy: {variance_accuracy_5d}")
    print("10D Embedding:")
    print(f"Mean accuracy: {mean_accuracy_10d}")
    print(f"Variance of accuracy: {variance_accuracy_10d}")
    
    results = {
        'mean_2d': mean_accuracy_2d,
        'variance_2d': variance_accuracy_2d,
        'mean_5d': mean_accuracy_5d,
        'variance_5d': variance_accuracy_5d,
        'mean_10d': mean_accuracy_10d,
        'variance_10d': variance_accuracy_10d
    }
    
    visualize_comparison2(results)
    # graph_file_path = 'graph.csv'
    # r, c, v = row_major(graph_file_path)
    # dense_matrix = create_adjacency_matrix(graph_file_path)
    
    # results = assess_runtime(r, c, v, dense_matrix)
    # print("Sparse Matrix-Vector Multiplication:")
    # print(f"Mean runtime: {results['sparse_mean']} seconds")
    # print(f"Variance of runtime: {results['sparse_variance']} seconds^2")
    
    # print("Dense Matrix-Vector Multiplication:")
    # print(f"Mean runtime: {results['dense_mean']} seconds")
    # print(f"Variance of runtime: {results['dense_variance']} seconds^2")
    # visualize_results(results)

    # labels_file_path = 'nodelabels.csv'
    
    # adjacency_matrix = create_adjacency_matrix(graph_file_path)
    # laplacian_matrix = compute_laplacian_matrix(adjacency_matrix)
    # embedding_1d = compute_embedding(laplacian_matrix, dimensions=1)
    # embedding_2d = compute_embedding(laplacian_matrix, dimensions=2)
    
    # # Load the labels
    # with open(labels_file_path, 'r') as file:
    #     labels = [int(line.strip()) for line in file.readlines()]
    
    # visualize_embedding_1d(embedding_1d, labels)
    # visualize_embedding_2d(embedding_2d, labels)
    # unnormalized_mean_accuracy_1d, unnormalized_variance_accuracy_1d = evaluate_classifier(embedding_1d, labels)
    # unnormalized_mean_accuracy_2d, unnormalized_variance_accuracy_2d = evaluate_classifier(embedding_2d, labels)

    # print("Unnormalized Laplacian 1D:")
    # print(f"Mean accuracy: {unnormalized_mean_accuracy_1d}")
    # print(f"Variance of accuracy: {unnormalized_variance_accuracy_1d}")
    # print("Unnormalized Laplacian 2D:")
    # print(f"Mean accuracy: {unnormalized_mean_accuracy_2d}")
    # print(f"Variance of accuracy: {unnormalized_variance_accuracy_2d}")

    # # Normalized Laplacian
    # normalized_laplacian_matrix = compute_normalized_laplacian_matrix(adjacency_matrix)
    # normalized_embedding_1d = compute_embedding(normalized_laplacian_matrix, dimensions=1)
    # normalized_embedding_2d = compute_embedding(normalized_laplacian_matrix, dimensions=2)
    
    # visualize_embedding_1d(normalized_embedding_1d, labels)
    # visualize_embedding_2d(normalized_embedding_2d, labels)
    # normalized_mean_accuracy_1d, normalized_variance_accuracy_1d = evaluate_classifier(normalized_embedding_1d, labels)
    # normalized_mean_accuracy_2d, normalized_variance_accuracy_2d = evaluate_classifier(normalized_embedding_2d, labels)

    # print("Normalized Laplacian 1D:")
    # print(f"Mean accuracy: {normalized_mean_accuracy_1d}")
    # print(f"Variance of accuracy: {normalized_variance_accuracy_1d}")
    # print("Normalized Laplacian 2D:")
    # print(f"Mean accuracy: {normalized_mean_accuracy_2d}")
    # print(f"Variance of accuracy: {normalized_variance_accuracy_2d}")

    # results = {
    #     'unnormalized_mean_1d': unnormalized_mean_accuracy_1d,
    #     'unnormalized_variance_1d': unnormalized_variance_accuracy_1d,
    #     'normalized_mean_1d': normalized_mean_accuracy_1d,
    #     'normalized_variance_1d': normalized_variance_accuracy_1d,
    #     'unnormalized_mean_2d': unnormalized_mean_accuracy_2d,
    #     'unnormalized_variance_2d': unnormalized_variance_accuracy_2d,
    #     'normalized_mean_2d': normalized_mean_accuracy_2d,
    #     'normalized_variance_2d': normalized_variance_accuracy_2d
    # }
    
    # visualize_comparison(results)


if __name__ == '__main__':
    main()