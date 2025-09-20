import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration ---
LEGACY_ENCODINGS_FILE = "encodings.pkl"
ARCFACE_ENCODINGS_FILE = "encodings_arcface.pkl"

def load_data(filepath):
    """Helper function to load encodings from a pickle file."""
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return data["names"], np.array(data["encodings"])
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None, None

def create_3d_plot(title, names, encodings):
    """
    Creates and displays a 3D t-SNE plot for a given set of encodings.
    """
    if encodings is None or len(encodings) == 0:
        print(f"Skipping plot for '{title}' due to missing data.")
        return

    print(f"\nPerforming 3D dimensionality reduction for {title}...")
    tsne = TSNE(n_components=3, perplexity=min(30, len(names)-1), random_state=42, init='pca', learning_rate='auto')
    reduced_points = tsne.fit_transform(encodings)

    print(f"Creating 3D plot for {title}...")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create a color map for unique names
    unique_names = list(np.unique(names))
    colors = plt.cm.get_cmap('tab10', len(unique_names))
    name_to_color = {name: colors(i) for i, name in enumerate(unique_names)}

    # Plot points with corresponding colors
    for i, name in enumerate(names):
        ax.scatter(reduced_points[i, 0], reduced_points[i, 1], reduced_points[i, 2],
                   color=name_to_color[name], label=name if name not in ax.get_legend_handles_labels()[1] else "")

    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.legend()

# --- 1. Load Both Sets of Encodings ---
print("Loading encodings from both models...")
legacy_names, legacy_encodings = load_data(LEGACY_ENCODINGS_FILE)
arcface_names, arcface_encodings = load_data(ARCFACE_ENCODINGS_FILE)

# --- 2. Create a plot for each model separately ---
create_3d_plot('Legacy Model (128-D)', legacy_names, legacy_encodings)
create_3d_plot('ArcFace Model (512-D)', arcface_names, arcface_encodings)

# --- 3. Show both plots ---
print("\nDisplaying plots. Close each plot window to exit.")
plt.show()