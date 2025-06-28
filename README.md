# 🌾 AI in Agriculture: Crop Monitoring and Yield Prediction

## 📄 Project Overview

This project leverages machine learning techniques to predict **Rice Yield (Kg/ha)** based on agricultural data such as year, area of cultivation, production, and state. It demonstrates the use of **EDA**, **outlier removal**, **data preprocessing**, **model training**, and **evaluation**.

---

## 📊 Dataset

### Source:

- `Crops_data.csv` (original raw data)

### Derived Datasets:

- `rice_data.csv`: Extracted rice-related data
- `rice_data_outlier_removed.csv`: Cleaned rice dataset after removing outliers using IQR

---

## 📚 Notebooks

| Notebook                         | Description                                                                                           |
| -------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `01_Preprocessing.ipynb`         | Loaded original dataset, filtered for rice, handled missing/null values, encoded categorical features |
| `02_EDA.ipynb`                   | Performed data visualization, outlier detection and removal (IQR method), and saved the cleaned data  |
| `03_Modelling.ipynb`             | Trained Random Forest model, performed scaling, trained/test split, saved model and scaler            |
| `04_Evaluation_Deployment.ipynb` | Evaluated the model (R², MAE, RMSE), plotted results, and saved prediction output                     |

---

## 📏 Folder Structure

`AI-agriculture-yield-prediction/`
`│`
`├── Datasets/`
│ ├── Crops_data.csv
│ ├── rice_data.csv
│ └── rice_data_outlier_removed.csv
│
├── Models/
│ ├── rf_model.pkl
│ ├── scaler.pkl
│ └── test_predictions.csv
│
├── Plots/
│ ├── actual_vs_predicted.png
│ └── residuals_distribution.png
│
├── Notebooks/
│ ├── 01_Preprocessing.ipynb
│ ├── 02_EDA.ipynb
│ ├── 03_Modelling.ipynb
│ └── 04_Evaluation_Deployment.ipynb
│
├── README.md
├── LICENSE
├── requirements.txt
└── .gitignore

---

## 📊 Model & Accuracy

- **Model Used**: Random Forest Regressor
- **Train Accuracy**: ~99.2%
- **Test Accuracy (R²)**: ~95.4%
- **MAE**: ~107.5
- **RMSE**: ~56287.4

---

## 🔧 Tech Stack

- **Language**: Python
- **Libraries**: pandas, numpy, matplotlib, seaborn, sklearn
- **IDE**: Jupyter Notebook (Anaconda)

---

## 📢 Notes

- Outliers were removed using IQR before training.
- Data was scaled using `StandardScaler`.
- Cross-validation was also explored to prevent overfitting.

---

## 🚀 How to Run

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run notebooks in sequence: `01` to `04`

---

## 🚑 Author

- **Name**: [Your Name Here]
- **Internship Project**: AI in Agriculture — Yield Prediction

---

## 📅 License

MIT License

