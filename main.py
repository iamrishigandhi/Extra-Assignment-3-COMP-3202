"""
Instructions for running the code:

1. Required Libraries:
    - Make sure you have the following libraries installed:
        - numpy
        - pandas
        - scikit-learn (sklearn)
        - matplotlib
        - scikit-bio (skbio)
    - You can install them using pip:
        ```
        pip install numpy pandas scikit-learn matplotlib scikit-bio
        ```

2. Run the Code:
    - Run this code in a Python environment.

3. Output:
    - The code will compute mean average precision and mean F1 score for different encoding methods (k-mer frequencies and one-hot encoding) in combination with a random forest classifier.
    - It will also plot precision-recall curves for each encoding method.
    - The results will be displayed as a pandas DataFrame showing the encoding method, mean average precision, and mean F1 score.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from skbio.sequence import DNA
import matplotlib.pyplot as plt

# Read the dataset
data = pd.read_csv("extraA3_Data.tsv", sep='\t', header=None)
X = data[0].values  # DNA sequences
y = data[1].values  # Labels

# Function to convert DNA sequences to k-mer frequencies
def get_kmer_features(X, k):
    kmer_features = []
    max_length = 0
    for sequence in X:
        dna = DNA(sequence)
        kmer_freq = dna.kmer_frequencies(k=k, relative=True)
        feature_vector = [kmer_freq.get(kmer, 0) for kmer in sorted(kmer_freq)]
        kmer_features.append(feature_vector)
        max_length = max(max_length, len(feature_vector))
    
    # Pad shorter feature vectors with zeros
    for i in range(len(kmer_features)):
        kmer_features[i] += [0] * (max_length - len(kmer_features[i]))
    
    return np.array(kmer_features)

# Function to convert DNA sequences to one-hot encoding
def get_one_hot_encoding(X):
    bases = 'ACGT'
    one_hot_encoded = []
    max_length = max(len(sequence) for sequence in X)
    for sequence in X:
        encoding = np.zeros((max_length, len(bases)))
        for i, base in enumerate(sequence):
            if base in bases:
                encoding[i, bases.index(base)] = 1
        one_hot_encoded.append(encoding.flatten())
    return np.array(one_hot_encoded)

# Function to calculate nucleotide composition features
def get_nucleotide_composition(X):
    nucleotide_features = []
    for sequence in X:
        length = len(sequence)
        a_count = sequence.count('A') / length
        c_count = sequence.count('C') / length
        g_count = sequence.count('G') / length
        t_count = sequence.count('T') / length
        nucleotide_features.append([a_count, c_count, g_count, t_count])
    return np.array(nucleotide_features)

# Choose a tree-based ensemble classification method
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Define cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Initialize lists to store results including standard deviations
mean_avg_precision_scores = []
mean_f1_scores = []
std_avg_precision_scores = []
std_f1_scores = []

# Define function to compute precision-recall curve and metrics
def compute_metrics(y_true, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    f1 = f1_score(y_true, (y_pred_proba > 0.5).astype(int))
    return precision, recall, avg_precision, f1

# Iterate over encodings
encodings = ['kmer_freq_2', 'kmer_freq_3', 'one_hot_encoding', 'nucleotide_composition']
for encoding in encodings:
    avg_precision_scores = []
    f1_scores = []

    # Transform data based on encoding
    if encoding.startswith('kmer_freq'):
        k = int(encoding.split('_')[-1])
        X_encoded = get_kmer_features(X, k)
    elif encoding == 'one_hot_encoding':
        X_encoded = get_one_hot_encoding(X)
    elif encoding == 'nucleotide_composition':
        X_encoded = get_nucleotide_composition(X)
    else:
        raise ValueError("Invalid encoding")

    # Perform cross-validation
    for train_index, test_index in cv.split(X_encoded, y):
        X_train, X_test = X_encoded[train_index], X_encoded[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train the classifier
        clf.fit(X_train, y_train)

        # Predict probabilities
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        # Compute metrics
        precision, recall, avg_precision, f1 = compute_metrics(y_test, y_pred_proba)
        avg_precision_scores.append(avg_precision)
        f1_scores.append(f1)

    # Store mean average precision score, mean F1 score, and their standard deviations
    mean_avg_precision_scores.append(np.mean(avg_precision_scores))
    mean_f1_scores.append(np.mean(f1_scores))
    std_avg_precision_scores.append(np.std(avg_precision_scores))
    std_f1_scores.append(np.std(f1_scores))

    # Plot precision-recall curve
    precision, recall, _, _ = compute_metrics(y_test, y_pred_proba)
    plt.figure()
    plt.step(recall, precision, where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve ({encoding})')
    plt.show()

# Display results including standard deviations
results_df = pd.DataFrame({
    'Encoding': encodings,
    'Mean Avg Precision': mean_avg_precision_scores,
    'Std Avg Precision': std_avg_precision_scores,
    'Mean F1 Score': mean_f1_scores,
    'Std F1 Score': std_f1_scores
})
print(results_df)
