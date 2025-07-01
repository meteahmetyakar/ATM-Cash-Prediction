# \# 📌 ATM Cash Level Prediction

# 

# A machine learning system to forecast the optimal cash level for Automated Teller Machines (ATMs), reducing operational costs and improving customer satisfaction by preventing cash shortages and overstocking.

# 

# ---

# 

# \## 📖 Project Description

# 

# Effective cash management in ATMs is critical to minimize both logistical expenses and customer frustration. This project builds predictive models that estimate the daily cash requirement for each ATM based on historical withdrawal trends, special events, and location.

# 

# Key benefits:

# \- \*\*Cost Reduction:\*\* Fewer emergency cash refills and less tied-up capital.

# \- \*\*Customer Satisfaction:\*\* Fewer “out-of-cash” events and enhanced trust.

# \- \*\*Scalable Pipeline:\*\* Automated data preprocessing, feature engineering, model training, and evaluation.

# 

# ---

# 

# \## 📂 Dataset

# 

# The dataset comprises 6 years of daily transaction data from City Union Bank ATMs. Key columns:

# \- `atm\_name`: Identifier of the ATM.

# \- `transaction\_date`: Date of the transactions.

# \- `No\_Of\_Withdrawals`: Total count of withdrawals.

# \- `total\_amount\_withdrawn`: Total amount dispensed.

# \- `weekday`: Day of week (0=Monday, 6=Sunday).

# \- `working\_day`: Binary flag for business days.

# \- `special\_day`: Binary flag for public holidays and cultural events.

# 

# ---

# 

# \## 🧹 Preprocessing \& Feature Engineering

# 

# 1\. \*\*Data Cleaning\*\*  

# &nbsp;  - Removed card-type breakdowns to focus on total withdrawals.  

# &nbsp;  - Handled missing values and date parsing.

# 

# 2\. \*\*Holiday \& Special Event Encoding\*\*  

# &nbsp;  - Used Python’s `holidays` library to mark Indian public holidays and cultural events (e.g., Holi, Diwali).

# 

# 3\. \*\*Encoding \& Transformation\*\*  

# &nbsp;  - Categorical features (weekday, special\_day) encoded as integers or binaries.  

# &nbsp;  - Time-based features (month, quarter) extracted from `transaction\_date`.

# 

# ---

# 

# \## 🧠 Ensemble Modeling

# 

# Two ensemble approaches were implemented:

# 

# \### Bootstrapping Optimistic Tree Construction (BOTC)

# \- Builds multiple decision trees on bootstrapped subsets.  

# \- Optimistic split evaluation prevents overfitting by estimating split effectiveness beyond the current sample.

# 

# \### Bagging Regression

# \- Trains multiple regression models (e.g., decision trees) on resampled data.  

# \- Aggregates predictions by averaging to reduce variance.

# 

# Both methods are compared to baseline regressors, and the best-performing model is selected.

# 

# ---

# 

# \## 📊 Evaluation

# 

# Model performance metrics on the test set:

# \- \*\*RMSE:\*\* 0.36

# \- \*\*MAE:\*\* 0.28

# \- \*\*R²:\*\* 0.86

# 

# See visualizations in the `plots/` folder:

# \- `actual\_vs\_predicted.png`

# \- `error\_distribution.png`

# 

# ---

# 

# \## 🛠️ Installation

# 

# 1\. \*\*Clone the repo\*\*  

# &nbsp;  ```bash

# &nbsp;  git clone https://github.com/meteahmetyakar/ATM-Cash-Level-Prediction.git

# &nbsp;  cd ATM-Cash-Level-Prediction

# &nbsp;  ```

# 

# 2\. \*\*Create environment \& install\*\*  

# &nbsp;  ```bash

# &nbsp;  python3 -m venv env

# &nbsp;  source env/bin/activate    # Linux/macOS

# &nbsp;  env\\Scripts\\activate      # Windows

# &nbsp;  pip install -r requirements.txt

# &nbsp;  ```

# 

# ---

# 

# \## ▶️ Usage

# 

# 1\. \*\*Prepare data\*\*  

# &nbsp;  ```bash

# &nbsp;  python scripts/preprocess.py --input data/raw.csv --output data/processed.csv

# &nbsp;  ```

# 

# 2\. \*\*Train model\*\*  

# &nbsp;  ```bash

# &nbsp;  python scripts/train\_model.py --data data/processed.csv --model models/best\_model.pkl

# &nbsp;  ```

# 

# 3\. \*\*Evaluate\*\*  

# &nbsp;  ```bash

# &nbsp;  python scripts/evaluate.py --model models/best\_model.pkl --test data/test.csv --output reports/metrics.json

# &nbsp;  ```

# 

# ---

# 

# \## 📂 File Structure

# 

# ```

# ├── data/

# │   ├── raw.csv

# │   └── processed.csv

# ├── plots/

# │   ├── actual\_vs\_predicted.png

# │   └── error\_distribution.png

# ├── scripts/

# │   ├── preprocess.py

# │   ├── train\_model.py

# │   └── evaluate.py

# ├── models/

# │   └── best\_model.pkl

# ├── requirements.txt

# └── README.md

# ```

# 

# ---

# 

# \## 🧩 Troubleshooting

# 

# | Issue                             | Solution                                                                |

# |-----------------------------------|-------------------------------------------------------------------------|

# | Missing dependencies              | `pip install -r requirements.txt`                                       |

# | Holiday encoding errors           | Verify `holidays` library version and correct country settings.         |

# | Model file not found              | Ensure training completed and `models/best\_model.pkl` exists.           |

# | Unexpected performance drop       | Check feature distributions and retrain with updated data.              |

# 

# ---

# 

# \## 🤝 Acknowledgments

# 

# \- \*\*Dataset:\*\* City Union Bank transaction logs  

# \- \*\*Libraries:\*\* scikit-learn, pandas, holidays  

# \- \*\*Author:\*\* Mete Ahmet Yakar  

# \- \*\*Advisor:\*\* Dr. Jane Doe  

# 

# ---



