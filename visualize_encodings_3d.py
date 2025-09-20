# visualize_encodings_3d.py
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# This import is necessary for 3D plotting
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration ---
ENCODINGS_FILE = "encodings.pkl"

print(f"Loading encodings from {ENCODINGS_FILE}...")

try:
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)

    names = data["names"]
    encodings = np.array(data["encodings"])

    # --- Perform Dimensionality Reduction using t-SNE ---
    print("Performing dimensionality reduction using t-SNE...")
    tsne = TSNE(
        n_components=3,     # KEY CHANGE: We now want 3 dimensions
        perplexity=min(15, len(names)-1),
        random_state=42,
        init='pca',
        learning_rate='auto'
    )
    reduced_encodings = tsne.fit_transform(encodings)

    # --- Create the Visualization Plot ---
    print("Creating 3D visualization plot...")
    fig = plt.figure(figsize=(13, 10))
    # KEY CHANGE: Create a 3D subplot
    ax = fig.add_subplot(111, projection='3d')
    
    # KEY CHANGE: Create a scatter plot with x, y, and z coordinates
    ax.scatter(reduced_encodings[:, 0], reduced_encodings[:, 1], reduced_encodings[:, 2])

    # Annotate each point with the person's name
    for i, name in enumerate(names):
        # KEY CHANGE: Use ax.text for labeling in 3D
        ax.text(
            reduced_encodings[i, 0],
            reduced_encodings[i, 1],
            reduced_encodings[i, 2],
            f'  {name}',  # Add a little space before the name
            size=10,
            zorder=1,
            color='k'
        )

    ax.set_title('3D Visualization of Facial Encodings using t-SNE')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3') # KEY CHANGE: Add a z-axis label
    plt.show()

except FileNotFoundError:
    print(f"Error: The file '{ENCODINGS_FILE}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")