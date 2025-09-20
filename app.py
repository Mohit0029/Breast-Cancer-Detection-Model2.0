from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the trained machine learning model
model = pickle.load(open("model.pkl", "rb"))

feature_names = [
    "clump_thickness", "uniform_cell_size", "uniform_cell_shape",
    "marginal_adhesion", "single_epithelial_size", "bare_nuclei",
    "bland_chromatin", "normal_nucleoli", "mitoses"
]

def generate_thermograph_image(input_data, is_malignant):
    """Generates a more realistic and attractive simulated thermograph."""
    try:
        # Create a high-resolution grid
        grid_size = 128
        x, y = np.mgrid[-1.0:1.0:grid_size*1j, -1.0:1.0:grid_size*1j]

        # Create a circular mask to simulate a breast shape
        radius = np.sqrt(x**2 + y**2)
        circular_mask = radius < 0.9

        # Normalize data (1-10 scale) to use as weights
        weights = np.array(input_data) / 10.0
        
        # Generate several Gaussian "hot spots" with locations and intensities based on input data
        # This creates a more complex and organic pattern
        locations = [(-0.4, 0.4), (0.5, 0.5), (0.0, -0.3), (-0.5, -0.5), (0.3, -0.1)]
        heatmap = np.zeros_like(x)

        for i in range(5):
            mean_x, mean_y = locations[i]
            # Use input features to influence the intensity and spread of blobs
            intensity = weights[i] * 2 + 0.5
            spread = 0.1 + weights[i+1] * 0.2
            g = np.exp(-(((x - mean_x)**2 + (y - mean_y)**2) / (2.0 * spread**2)))
            heatmap += intensity * g

        # If malignant, add a dominant, asymmetrical hot spot
        if is_malignant:
            tumor_intensity = np.max(weights) * 5 + 2.0
            tumor_spread = 0.05 + weights[np.argmax(weights)] * 0.1
            tumor_x, tumor_y = (weights[2]-0.5)*0.8, (weights[1]-0.5)*0.8 # Location based on cell shape/size uniformity
            g_tumor = np.exp(-(((x - tumor_x)**2 + (y - tumor_y)**2) / (2.0 * tumor_spread**2)))
            heatmap += tumor_intensity * g_tumor

        # Apply the circular mask and add subtle background noise
        heatmap = heatmap * circular_mask + np.random.normal(0, 0.05, heatmap.shape) * circular_mask

        fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
        ax.imshow(heatmap, cmap='hot', interpolation='bilinear')
        ax.axis('off')

        plt.tight_layout(pad=0)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig)
        buf.seek(0)
        
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"
    except Exception:
        return None


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_features = [int(request.form.get(name)) for name in feature_names]
        features_df = pd.DataFrame([input_features], columns=feature_names)
        
        prediction = model.predict(features_df)[0]
        prediction_proba = model.predict_proba(features_df)[0]
        
        is_malignant_prediction = (prediction == 4)
        thermograph_image_data = generate_thermograph_image(input_features, is_malignant_prediction)

        if prediction == 4:
            result_class = "result-malignant"
            prediction_text = "Malignant"
            confidence = prediction_proba[1]
        else:
            result_class = "result-benign"
            prediction_text = "Benign"
            confidence = prediction_proba[0]
            
        confidence_score = f"Model Confidence: {confidence * 100:.2f}%"
        
        return jsonify({
            'result_class': result_class,
            'prediction_text': prediction_text,
            'confidence_score': confidence_score,
            'thermograph_image_data': thermograph_image_data,
            'input_data': input_features
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)

