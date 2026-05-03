# Machine Learning Pipeline for Titanic Survival Prediction

This repository contains a machine learning pipeline for predicting Titanic survival outcomes using a Random Forest classifier. The pipeline includes data preprocessing, model training, evaluation, testing, and generating reports in both Excel and PDF formats. 

## Table of Contents
- [Overview](#overview)
- [Pipeline Components](#pipeline-components)
- [Installation](#installation)
- [Running the Pipeline](#running-the-pipeline)
- [Directory Structure](#directory-structure)
- [Customization and Configuration](#customization-and-configuration)
- [Requirements](#requirements)
- [Contact](#contact)

---

## Overview

The pipeline predicts whether a passenger survived the Titanic disaster based on features such as age, gender, ticket class, fare, etc. The model is trained using a Random Forest classifier, and the pipeline offers the following features:
- **Training the Model**: Preprocessing the data and training a Random Forest model.
- **Testing the Model**: Evaluating model performance on a validation dataset or new/random datasets.
- **Visualization**: Generating visual outputs and performance reports in Excel and PDF formats.
- **Report Generation**: Creating PDF reports that summarize model performance statistics like Precision, Recall, F1-Score, MSE, RMSE, and MAE.

## Pipeline Components

The key components of this pipeline are:
1. **Data Preprocessing**: Data is cleaned and prepared by handling missing values, encoding categorical variables, and dropping irrelevant columns.
2. **Model Training**: A Random Forest model is trained on the Titanic dataset with hyperparameters set through a configuration file.
3. **Testing**: The trained model is tested on new datasets, and evaluation metrics like accuracy, precision, recall, and F1-score are reported.
4. **Excel Output with Color Coding**: The results are saved in an Excel file with columns color-coded for better readability.
5. **PDF Report Generation**: A detailed PDF report is created that includes the model’s performance metrics.
6. **Visualization**: Graphs and visual data summaries are generated for exploratory data analysis (EDA) and model insights.

---

## Installation

To run the pipeline locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8 or higher installed. Install all necessary dependencies using the provided `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

   This will install the following libraries:
   - pandas
   - scikit-learn
   - joblib
   - numpy
   - argparse
   - openpyxl
   - fpdf
   - matplotlib
   - seaborn

3. **Set Up the Directory**:
   Ensure you have the following file structure on your local system:
   ```
   .
   ├── MS_1_Scenario_train.csv       # Training dataset
   ├── MS_1_Scenario_test.csv        # Validation dataset
   ├── config.txt                    # Model configuration file
   ├── eda_outputs/                  # Output folder for EDA results
   ├── generate_pdf.py               # Script for generating PDF reports
   ├── predict_model.py              # Script for making predictions on test datasets
   ├── train_model.py                # Script for training the model
   ├── visualize.py                  # Script for visualizing the results
   ├── pipeline.sh                   # Main pipeline shell script
   └── test CSV/                     # Directory for new/random test datasets
   ```

4. **Configure Model Parameters**:
   Edit the `config.txt` file to adjust the hyperparameters for the Random Forest model (optional):
   ```ini
   [MODEL_PARAMS]
   n_estimators = 80
   max_depth = 8
   min_samples_split = 10
   min_samples_leaf = 4
   random_state = 42

   # Paths
   TRAIN_DATA = MS_1_Scenario_train.csv
   TEST_DATA = MS_1_Scenario_test.csv
   EDA_OUTPUT = eda_outputs
   MODEL_OUTPUT = random_forest_model.pkl
   ```

---

## Running the Pipeline

You can run the entire pipeline using the provided shell script `pipeline.sh`. The pipeline supports three options: training the model, testing on validation datasets, and testing on new/random datasets.

1. **Run the Pipeline**:
   To execute the pipeline, use the following command:
   ```bash
   ./pipeline.sh
   ```

2. **CLI Menu**:
   The pipeline will display a menu with the following options:
   ```bash
   Pipeline CLI Menu:
   1. Train the Model
   2. Test on Available Validation Dataset
   3. Test on New/Random Dataset
   4. Exit
   ```

### Option 1: Train the Model
- This option will preprocess the training dataset, train the Random Forest model, generate training and validation reports, and save the model to a file.
  
### Option 2: Test on Available Validation Dataset
- This option will test the trained model on the provided validation dataset (`MS_1_Scenario_test.csv`) and output the results to both an Excel file and a PDF report.

### Option 3: Test on New/Random Dataset
- This option loops through all CSV files in the `test CSV/` directory, runs the test on each file, and generates individual reports (Excel and PDF) for each dataset.

### Example of Output
- **Excel**: Results are saved in a color-coded Excel file, where predicted survival outcomes are marked in green (for 'Yes') or red (for 'No').
- **PDF**: A PDF file is generated that includes performance metrics like Precision, Recall, F1-Score, MSE, RMSE, and MAE.

---

## Directory Structure

Ensure the following directory structure is maintained for smooth execution:

```plaintext
.
├── pipeline.sh                   # Main shell script for pipeline
├── train_model.py                # Script to train the model
├── predict_model.py              # Script to make predictions
├── generate_pdf.py               # Script to generate PDF reports
├── config.txt                    # Configuration file for model parameters
├── MS_1_Scenario_train.csv       # Training dataset
├── MS_1_Scenario_test.csv        # Validation dataset
├── test CSV/                     # Directory containing new/random test datasets
├── eda_outputs/                  # Output directory for EDA results
├── requirements.txt              # File containing all required libraries
└── README.md                     # This file (readme for the project)
```

---

## Customization and Configuration

1. **Configuring Hyperparameters**:
   Modify the `config.txt` file to customize the hyperparameters of the Random Forest model, such as `n_estimators`, `max_depth`, and `min_samples_split`.

2. **Adding New Test Datasets**:
   To test the model on new datasets, place your CSV files inside the `test CSV/` directory. The pipeline will automatically detect and test all CSV files in this folder when you select Option 3 from the menu.

3. **File Paths**:
   Make sure all file paths are correctly configured in `config.txt`, especially if you are using custom directories for datasets, models, and EDA output files.

---

## Requirements

Make sure you have the following libraries installed (see `requirements.txt` for version details):
- `pandas`
- `scikit-learn`
- `joblib`
- `numpy`
- `argparse`
- `openpyxl`
- `fpdf`
- `matplotlib`
- `seaborn`

Install all dependencies using:
```bash
pip install -r requirements.txt
```

---

## Contact

For questions or feedback, feel free to reach out at [2201512@sit.singaporetech.edu.sg](mailto:lordmuze@gmail.com).

--- 

This README is designed to provide clear instructions for setting up and using the machine learning pipeline, ensuring smooth execution from installation to testing and reporting.
