# Heart_Disease_Detection

Heart Disease Detection Project

1. Project Description

This project aims to develop a Heart Disease Detection System using both a rule-based expert system (Experta) and a machine learning model (Decision Tree Classifier in Scikit-Learn). The system will analyze patient health indicators to predict heart disease risk.

Additionally, the project includes:

Data Preprocessing

Data Visualization

Rule-Based Expert System Implementation

Machine Learning Model Training & Evaluation

Streamlit UI for User Interaction

2. Technologies Used

Python (3.x)

Pandas, NumPy (Data Processing)

Seaborn, Matplotlib (Data Visualization)

Scikit-Learn (Machine Learning Model)

Experta (Rule-Based Expert System)

Streamlit (Web-based UI)

Joblib (Model Saving)

3. Project Structure
4. Heart_Disease_Detection/
│── data/                 # Contains the dataset (raw & cleaned)
│   ├── raw_data.csv
│   ├── cleaned_data.csv
│── notebooks/            # Jupyter Notebooks for visualization & preprocessing
│   ├── data_analysis.ipynb
│   ├── model_training.ipynb
│── rule_based_system/    # Rule-based system using Experta
│   ├── rules.py
│   ├── expert_system.py
│── ml_model/             # Decision Tree implementation
│   ├── train_model.py
│   ├── predict.py
│── utils/                # Helper functions for data cleaning & processing
│   ├── data_processing.py
│── reports/              # Comparison reports and evaluation
│   ├── accuracy_comparison.md
│── ui/                   # Streamlit UI for user interaction
│   ├── app.py
│── README.md             # Project documentation & setup instructions
│── requirements.txt      # List of dependencies

4. Installation & Setup

Step 1: Clone the Repository
git clone https://github.com/yourusername/Heart_Disease_Detection.git
cd Heart_Disease_Detection

Step 2: Create a Virtual Environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate   # For Mac/Linux
venv\Scripts\activate      # For Windows

Step 3: Install Dependencies
pip install -r requirements.txt

5. Running the Project

Step 1: Data Preprocessing & Visualization

Run the Jupyter Notebooks to preprocess and analyze the dataset:

jupyter notebook notebooks/data_analysis.ipynb
upyter notebook notebooks/data_analysis.ipynb

 step 2: Train the Machine Learning Model

Run the script to train the Decision Tree model:
python ml_model/train_model.py

Step 3: Run the Rule-Based Expert System
python rule_based_system/expert_system.py

step 4: Launch the Streamlit UI

To run the web application, execute:
streamlit run ui/app.py

This will start the web UI where users can input their health indicators and receive a heart disease risk assessment.

6. Features

Data Preprocessing: Handling missing values, normalization, encoding.

Visualization: Statistical summaries, correlation heatmaps, feature importance ranking.

Rule-Based System: At least 10 expert-defined rules for risk assessment.

Machine Learning Model: Decision Tree Classifier with hyperparameter tuning.

Comparison: Evaluating accuracy and explainability between the two approaches.

Interactive UI: Web-based interface for user interaction.

7. Contributors

Your Name - Project Lead

Team Member 1 - Data Preprocessing & Visualization

Team Member 2 - Machine Learning Model Implementation

Team Member 3 - Rule-Based System & UI Development

8. License

This project is licensed under the MIT License - see the LICENSE file for details.
