from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import os
from sklearn.cluster import KMeans  # New

# Initialize Flask app
app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create folder if it doesn't exist

def rgb_to_hex(rgb):
    """Convert RGB tuple to HEX string."""
    return '#%02x%02x%02x' % rgb

def get_top_colors(image_path, num_colors=10):
    """
    Extract top visually unique colors from an image using KMeans clustering.

    Args:
        image_path (str): Path to the image.
        num_colors (int): Number of clusters/colors.

    Returns:
        List[Dict]: List of colors with RGB and HEX values.
    """
    with Image.open(image_path) as img:
        img = img.convert('RGB')           # Ensure RGB mode
        img = img.resize((100, 100))       # Resize for performance
        pixels = np.array(img).reshape(-1, 3)

    # Use KMeans to find color clusters
    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init='auto')
    kmeans.fit(pixels)
    centers = np.round(kmeans.cluster_centers_).astype(int)

    return [{
        'rgb': f'rgb({r},{g},{b})',
        'hex': rgb_to_hex((r, g, b)),
    } for r, g, b in centers]

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Main page: Upload image and display extracted top distinct colors.
    """
    colors = []

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            colors = get_top_colors(filepath)  # Extract distinct clustered colors

    return render_template('index.html', colors=colors)

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
