# 📌 ATM Cash Level Prediction

A machine learning system to forecast the optimal cash level for Automated Teller Machines (ATMs), reducing operational costs and improving customer satisfaction by preventing cash shortages and overstocking.

---

## 📖 Project Description

Effective cash management in ATMs is critical to minimize both logistical expenses and customer frustration. This project builds predictive models that estimate the daily cash requirement for each ATM based on historical withdrawal trends, special events, and location.

Key benefits:
- **Cost Reduction:** Fewer emergency cash refills and less tied-up capital.
- **Customer Satisfaction:** Fewer “out-of-cash” events and enhanced trust.
- **Scalable Pipeline:** Automated data preprocessing, feature engineering, model training, and evaluation.

---

## 📂 Dataset

The dataset comprises 6 years of daily transaction data from City Union Bank ATMs. Key columns:
- `atm_name`: Identifier of the ATM.
- `transaction_date`: Date of the transactions.
- `No_Of_Withdrawals`: Total count of withdrawals.
- `total_amount_withdrawn`: Total amount dispensed.
- `weekday`: Day of week (0=Monday, 6=Sunday).
- `working_day`: Binary flag for business days.
- `special_day`: Binary flag for public holidays and cultural events.

---

## 🧹 Preprocessing & Feature Engineering

1. **Data Cleaning**  
   - Removed card-type breakdowns to focus on total withdrawals.  
   - Handled missing values and date parsing.

2. **Holiday & Special Event Encoding**  
   - Used Python’s `holidays` library to mark Indian public holidays and cultural events (e.g., Holi, Diwali).

3. **Encoding & Transformation**  
   - Categorical features (weekday, special_day) encoded as integers or binaries.  
   - Time-based features (month, quarter) extracted from `transaction_date`.

---

## 🧠 Ensemble Modeling

Two ensemble approaches were implemented:

### Bootstrapping Optimistic Tree Construction (BOTC)
- Builds multiple decision trees on bootstrapped subsets.  
- Optimistic split evaluation prevents overfitting by estimating split effectiveness beyond the current sample.

### Bagging Regression
- Trains multiple regression models (e.g., decision trees) on resampled data.  
- Aggregates predictions by averaging to reduce variance.

Both methods are compared to baseline regressors, and the best-performing model is selected.

---

## 📊 Evaluation

Model performance metrics on the test set:
- **RMSE:** 0.36
- **MAE:** 0.28
- **R²:** 0.86

See visualizations in the `plots/` folder:
- `actual_vs_predicted.png`
- `error_distribution.png`

---

## 🛠️ Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/meteahmetyakar/ATM-Cash-Level-Prediction.git
   cd ATM-Cash-Level-Prediction
   ```

2. **Create environment & install**  
   ```bash
   python3 -m venv env
   source env/bin/activate    # Linux/macOS
   env\Scripts\activate      # Windows
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

1. **Prepare data**  
   ```bash
   python scripts/preprocess.py --input data/raw.csv --output data/processed.csv
   ```

2. **Train model**  
   ```bash
   python scripts/train_model.py --data data/processed.csv --model models/best_model.pkl
   ```

3. **Evaluate**  
   ```bash
   python scripts/evaluate.py --model models/best_model.pkl --test data/test.csv --output reports/metrics.json
   ```

---

## 📂 File Structure

```
├── data/
│   ├── raw.csv
│   └── processed.csv
├── plots/
│   ├── actual_vs_predicted.png
│   └── error_distribution.png
├── scripts/
│   ├── preprocess.py
│   ├── train_model.py
│   └── evaluate.py
├── models/
│   └── best_model.pkl
├── requirements.txt
└── README.md
```

---

## 🧩 Troubleshooting

| Issue                             | Solution                                                                |
|-----------------------------------|-------------------------------------------------------------------------|
| Missing dependencies              | `pip install -r requirements.txt`                                       |
| Holiday encoding errors           | Verify `holidays` library version and correct country settings.         |
| Model file not found              | Ensure training completed and `models/best_model.pkl` exists.           |
| Unexpected performance drop       | Check feature distributions and retrain with updated data.              |

---

## 🤝 Acknowledgments

- **Dataset:** City Union Bank transaction logs  
- **Libraries:** scikit-learn, pandas, holidays  
- **Author:** Mete Ahmet Yakar  
- **Advisor:** Dr. Jane Doe  

---
