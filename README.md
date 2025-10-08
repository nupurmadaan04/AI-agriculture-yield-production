# AI in Agriculture: Crop Monitoring and Yield Prediction


[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)
![Status](https://img.shields.io/badge/Status-Active-success)


--- 

## 📖 Table of Contents

- [📘 Overview](#project-overview)
- [🗂️ Dataset](#dataset)
- [🧩 Project Structure](#-project-structure)
- [🛠️ Tech Stack](#tech-stack)
- [🚀 How to Run](#-how-to-run)
- [📝 Notebooks](#-notebooks)
- [🤖 Model & Accuracy](#-model--accuracy)
- [🤝 Contributing](#-contributing)
- [📝 Notes](#-notes)
- [🔮 Future Work / Extensions](#-future-work--extensions)
- [🪪 License](#-license)


---



## Project Overview


This project uses **Machine Learning** to predict **Rice Yield (Kg/ha)** based on factors like:
- Year  
- Area of cultivation  
- Production  
- State  

It includes data preprocessing, outlier removal, model training, and evaluation using **Random Forest Regressor**.
It demonstrates the use of **EDA**, **outlier removal**, **data preprocessing**, **model training**, and **evaluation**.


---


## Dataset


### Source:

- `Crops_data.csv` (original raw data)

### Derived Datasets:

- `rice_data.csv`: Extracted rice-related data
- `rice_data_outlier_removed.csv`: Cleaned rice dataset after removing outliers using IQR


---


## 🧩 Project Structure

```bash
AI-agriculture-yield-production/
├── Code/
│   ├── .github/
│   │   ├── ISSUE_TEMPLATE/
│   │   │   ├── bug_report.yml
│   │   │   ├── config.yml
│   │   │   └── feature_request.yml
│   │   └── pull_request_template.md
│   ├── .streamlit/
│   │   └── config.toml
│   ├── templates/
│   │   └── index.html
│   └── app.py
├── Datasets/
│   ├── Crops_data.csv
│   ├── rice_data.csv
│   └── rice_data_outlier_removed.csv
├── Models/
│   ├── rf_model.pkl
│   ├── scaler.pkl
│   └── test_predictions.csv
├── Notebooks/
│   ├── 01_Data_preprocessing.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_modelling.ipynb
│   ├── 04_Evaluation_Deployment.ipynb
│   ├── 05_boosting_model.ipynb
│   └── 06_Modular_Evaluation.ipynb
├── src/
│   ├── __init__.py
│   └── preprocessing_functions.py
├── tests/
│   ├── README.md
│   ├── run_tests.py
│   ├── test_data_preprocessing_comprehensive.py
│   ├── test_data_preprocessing_old_broken.py.bak
│   ├── test_preprocessing.py
│   └── test_preprocessing_simple.py
├── .gitignore
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── Evaluation_metrics.md
├── LICENSE.md
├── README.md
├── requirements.txt
└── pyproject.toml

```

---



##  Tech Stack

- 🐍 **Language:** `Python`
- 📚 **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- 📝 **IDE / Environment:** `Jupyter Notebook (Anaconda)`


---


## 🚀 How to Run

Follow these steps to set up and execute the project:

### 1️⃣ Fork the Repository
- Click the **Fork** button on the top-right corner of the original repo:   
- This will create a copy under your GitHub account.

### 2️⃣ Clone Your Fork
```bash
git clone https://github.com/your-username/AI-agriculture-yield-production.git
cd AI-agriculture-yield-production
```

### 3️⃣ Create and Activate Virtual Environment
```bash 
# Create a virtual environment
python -m venv env

# Activate it
env\Scripts\activate          # On Windows
source env/bin/activate       # On Mac/Linux
```
### 4️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 5️⃣ Run Jupyter Notebook
```bash
jupyter notebook
```
Open and execute the notebooks **in order**:

1. `01_Data_preprocessing.ipynb`
2. `02_eda.ipynb`
3. `03_modelling.ipynb`
4. `04_Evaluation_Deployment.ipynb`


---


## 📝 Notebooks

| Notebook                         | Description                                                                                           |
| -------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `01_Data_preprocessing.ipynb`         | Loaded original dataset, filtered for rice, handled missing/null values, encoded categorical features |
| `02_eda.ipynb`                   | Performed data visualization, outlier detection and removal (IQR method), and saved the cleaned data  |
| `03_modelling.ipynb`             | Trained Random Forest model, performed scaling, trained/test split, saved model and scaler            |
| `04_Evaluation_Deployment.ipynb` | Evaluated the model (R², MAE, RMSE), plotted results, and saved prediction output                     |


---


### 🤖 Model & Accuracy

| Metric                | Value       |
|-------------------------------|------------|
| Model Used 🌲                  | Random Forest Regressor |
| Train Accuracy ✅              | ~99.2%    |
| Test Accuracy (R²) 📊          | ~95.4%    |
| Mean Absolute Error (MAE) ✏️  | ~107.5    |
| Root Mean Squared Error (RMSE) 📉 | ~56287.4 |



---


## 🤝 Contributing

Contributions are always welcome 💡

Please follow these steps:

```bash
# 1. Fork the repository
# 2. Create your feature branch
git checkout -b feature-name

# 3. Commit your changes
git commit -m "Add your feature description"

# 4. Push to your branch
git push origin feature-name

# 5. Create a Pull Request
```


---

## 📝 Notes

- ⚠️ Outliers were removed using IQR before training.  
- ⚡ Data was scaled using `StandardScaler`.  
- 🔄 Cross-validation was also explored to prevent overfitting.

---


## 🔮 Future Work / Extensions

Here are a few ways this project can be improved further:

- 🌾 Add support for multiple crops (wheat, maize, etc.)  
- 🤖 Implement advanced models   
- ☁️ Deploy model using Streamlit or Flask web app  
- 📈 Build an interactive dashboard for yield insights  


---


## Author

- **Name**: [Nupur Madaan]
- **Internship Project**: AI in Agriculture — Yield Prediction



## 🪪 License

This project is licensed under the [MIT License](LICENSE.md).



