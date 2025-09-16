# visualize_encodings.py
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# --- Configuration ---
ENCODINGS_FILE = "encodings.pkl"

print(f"Loading encodings from {ENCODINGS_FILE}...")

try:
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)

    names = data["names"]
    encodings = np.array(data["encodings"])

    # --- Perform Dimensionality Reduction using t-SNE ---
    # t-SNE is a powerful technique for visualizing high-dimensional data.
    # We are reducing the 128 dimensions down to 2.
    print("Performing dimensionality reduction using t-SNE...")
    tsne = TSNE(
        n_components=2,     # We want a 2D plot
        perplexity=min(15, len(names)-1),  # A key tuning parameter, default 30 is too high for few points
        random_state=42,    # For reproducible results
        init='pca',
        learning_rate='auto'
    )
    reduced_encodings = tsne.fit_transform(encodings)

    # --- Create the Visualization Plot ---
    print("Creating visualization plot...")
    plt.figure(figsize=(12, 8))
    
    # Create a scatter plot of the reduced data
    scatter = plt.scatter(reduced_encodings[:, 0], reduced_encodings[:, 1])

    # Annotate each point with the person's name
    for i, name in enumerate(names):
        plt.annotate(
            name,
            (reduced_encodings[i, 0], reduced_encodings[i, 1]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center'
        )

    plt.title('2D Visualization of Facial Encodings using t-SNE')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    plt.show()

except FileNotFoundError:
    print(f"Error: The file '{ENCODINGS_FILE}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")