Breast Cancer Diagnostic AI



Overview

This is a web-based tool I built to predict breast cancer using a machine learning model. You can input the cytological features of a tumor, and the app will instantly tell you if it's likely benign or malignant.



It's built as a single-page application, so the results—including a dynamically generated thermograph and a feature analysis chart—appear on the same page without a refresh. The whole project covers the full ML pipeline, from training the model to deploying it in this interactive web interface.



Key Features

Real-Time Predictions: Uses a Random Forest model to give you an immediate prediction.



Generated Thermograph: Instead of a static image, the app creates a unique thermal image for every prediction, highlighting potential hot spots in malignant cases.



Visual Feature Analysis: A radar chart shows you the patient's data, making it easy to see which features are contributing to the result.



No Page Reloads: It's built as a Single-Page App, so results load smoothly without refreshing the page.



Clean UI: The interface is designed to be clean and easy to use, with a medical theme and helpful tooltips.



Full Pipeline Included: The repository contains everything, including the Python script used to train the model.



Tech Stack

Backend: Python, Flask



Machine Learning: Scikit-learn, Pandas, NumPy



Data Visualization: Matplotlib



Frontend: HTML5, CSS3, JavaScript



Charting: Chart.js



Project Structure

Breast Cancer Model/

│

├── app.py              # Main Flask application with prediction logic

├── train\_model.py      # Script to train and save the ML model

├── model.pkl           # (Generated) Trained machine learning model

├── requirements.txt    # Required Python libraries

│

├── static/

│   └── style.css       # All styling for the web interface

│

└── templates/

&nbsp;   └── index.html      # The HTML structure for the web app



&nbsp;Getting Started

To get this running on your own machine, you'll need to do the following:



1\. Clone the Repository

git clone \[https://github.com/your-username/Breast-Cancer-Model.git](https://github.com/your-username/Breast-Cancer-Model.git)

cd Breast-Cancer-Model



2\. Set up a Virtual Environment (Recommended)

\# For Windows

python -m venv venv

venv\\Scripts\\activate



3\. Install the Dependencies

This will install all the Python libraries needed for the project.



pip install -r requirements.txt



Running the App

Step 1: Train the Model

You only need to do this once. Run the training script to generate the model.pkl file.



python train\_model.py



Step 2: Start the Server

With the model file created, you can now start the web app.



python app.py



Step 3: View in Browser

Open your browser and go to http://127.0.0.1:5000. You can now start making predictions.



&nbsp;A Bit About the Model

The prediction model is a RandomForestClassifier I trained using Scikit-learn.



The data comes from the Breast Cancer Wisconsin (Original) dataset from the UCI Machine Learning Repository.



It's trained on 9 key features: Clump Thickness, Uniform Cell Size, Uniform Cell Shape, Marginal Adhesion, Single Epithelial Size, Bare Nuclei, Bland Chromatin, Normal Nucleoli, and Mitoses.



On the test set, the final model achieved an accuracy of about 95%.



&nbsp;Disclaimer

This application is an educational tool created to demonstrate a machine learning pipeline. The predictions are based on a statistical model and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

