# ML Project — Comment Category Prediction

A machine learning project for predicting comment categories using a combination of **textual, behavioral, categorical, and temporal features**.

The project was developed as part of the **Machine Learning Practice (MLP)** project and follows an end-to-end machine learning workflow including exploratory data analysis, preprocessing, feature engineering, TF-IDF representation, model comparison, evaluation, and final prediction.

---

## 📌 Project Overview

The objective of this project is to classify comments into one of **four categories (0, 1, 2, 3)** using both the content of the comment and associated structured features.

The dataset contains:

- **198,000 training samples**
- **102,000 test samples**
- **14 input features**
- **1 target label**
- **4 target classes**

The training dataset contains textual information such as `comment`, along with reaction, emoticon, platform, demographic, and timestamp-related features.

The primary evaluation metric is **Macro F1 Score**, which is appropriate because the target classes are imbalanced. Class 0 represents approximately **57.7%** of the training data, while class 3 represents only about **2.7%**.

---

## 🎯 Objectives

- Perform exploratory data analysis on the dataset
- Handle missing and inconsistent values
- Clean and preprocess comment text
- Engineer useful numerical and temporal features
- Convert text into numerical representations using TF-IDF
- Combine text and structured features
- Compare multiple machine learning algorithms
- Optimize the best-performing model
- Train the final model on the complete training dataset
- Generate predictions for the test dataset
- Create the final `submission.csv`

---

## 📊 Dataset

### Training Data

The training dataset contains **198,000 rows and 15 columns**, including the target `label`.

### Test Data

The test dataset contains **102,000 rows and 14 columns**.

### Features

| Feature | Description |
|---|---|
| `created_date` | Timestamp of the comment |
| `post_id` | Post identifier |
| `emoticon_1` | Emoticon indicator |
| `emoticon_2` | Emoticon indicator |
| `emoticon_3` | Emoticon indicator |
| `upvote` | Number of upvotes |
| `downvote` | Number of downvotes |
| `if_1` | Platform-related feature |
| `if_2` | Platform-related feature |
| `race` | Sensitive-topic indicator |
| `religion` | Sensitive-topic indicator |
| `gender` | Sensitive-topic indicator |
| `disability` | Disability indicator |
| `comment` | Original comment text |
| `label` | Target class |

The dataset contains missing values primarily in `race`, `religion`, `gender`, and one missing `comment`.

---

## 🔍 Exploratory Data Analysis

The project began with analysis of:

- Dataset dimensions
- Data types
- Missing values
- Target distribution
- Numerical feature statistics
- Feature correlations
- Relationship between structured features and target classes

One notable finding was that **`if_2` had the strongest correlation with the target**, with an absolute correlation of approximately **0.233**.

The class distribution was:

| Class | Percentage |
|---:|---:|
| 0 | 57.66% |
| 1 | 8.04% |
| 2 | 31.54% |
| 3 | 2.76% |

Because of this imbalance, **Macro F1** was used instead of relying solely on accuracy.

---

## 🧹 Data Preprocessing

### Missing Values

The following preprocessing steps were applied:

- Missing comments were replaced with empty strings.
- The boolean `disability` feature was converted to an integer.
- Missing categorical values were handled using `"none"` during preprocessing.

### Text Cleaning

The comment text was normalized by:

1. Converting text to lowercase
2. Removing punctuation
3. Removing numerical characters
4. Normalizing whitespace

This produced a cleaned `comment_clean` feature for text vectorization.

---

## 🛠️ Feature Engineering

Several additional features were created from the original data.

### Text Features

- `comment_length`
- `word_count`
- `exclamation_count`
- `question_count`

### Engagement Features

- `upvote_ratio`
- `score_difference`
- `engagement`
- `controversy`

### Emoticon Features

- `total_emoticons`
- `has_emotion`

### Temporal Features

The `created_date` timestamp was converted into:

- `hour`
- `dayofweek`
- `is_weekend`
- `is_sunday`

These engineered features were designed to capture comment characteristics, engagement patterns, emotional indicators, and temporal behavior.

---

## 📝 Text Representation

TF-IDF was used to convert comments into numerical features.

### Word-Level TF-IDF

- N-grams: **1–2**
- Maximum features: **50,000**
- Minimum document frequency: **3**
- English stop-word removal
- Sublinear TF enabled

### Character-Level TF-IDF

- Character n-grams: **3–5**
- Maximum features: **10,000**

The two representations were combined to capture both:

- Word and phrase-level patterns
- Character-level patterns such as spelling variations and typos

The resulting text representation contained **60,000 features**.

---

## 🔗 Structured Feature Processing

Structured features were processed using a `ColumnTransformer`.

### Numerical Pipeline

```text
Numerical Features
       ↓
StandardScaler
```

### Categorical Pipeline

```text
Categorical Features
       ↓
Missing-value Imputation
       ↓
OrdinalEncoder
```

The structured representation contained **25 features**.

The final feature matrix combined:

```text
Word TF-IDF
      +
Character TF-IDF
      +
Structured Features
      ↓
Final Feature Matrix
```

This resulted in **60,025 combined features**. 

---

## 🤖 Models Evaluated

Three classification algorithms were compared:

1. **Logistic Regression**
2. **Linear SVM**
3. **LightGBM**

The training data was split into:

- **80% training**
- **20% validation**

using stratified sampling with `random_state=42`.

---

## 📈 Model Performance

The models were evaluated using **Macro F1 Score**.

| Model | Validation Macro F1 |
|---|---:|
| Logistic Regression | **0.7975** |
| Linear SVM | **0.6593** |
| LightGBM | **0.8137** |

### 🏆 Best Model: LightGBM

LightGBM achieved the highest validation Macro F1 score of:

**0.81365**

It also achieved approximately **92% validation accuracy**.

The per-class F1 scores were:

| Class | F1 Score |
|---:|---:|
| 0 | 0.96 |
| 1 | 0.79 |
| 2 | 0.89 |
| 3 | 0.62 |

Class 3 remained the most challenging class because of its relatively small representation in the dataset.

---

## ⚙️ LightGBM Configuration

The final LightGBM model used:

```python
LGBMClassifier(
    objective="multiclass",
    num_class=4,
    num_leaves=64,
    learning_rate=0.05,
    min_child_samples=20,
    n_estimators=305,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    random_state=42
)
```

Early stopping was used during validation, resulting in **305 optimal trees**. 

---

## 🏁 Final Training

After selecting LightGBM as the best-performing model, it was retrained using the **entire training dataset**.

The final model was trained using:

- **198,000 training samples**
- **305 trees**
- 4 output classes
- The complete engineered feature set

The model then generated predictions for all **102,000 test samples**.

---

## 📤 Submission

Predictions were added to the provided `Sample.csv` file and saved as:

```text
submission.csv
```

The final submission contains the predicted `label` for each test sample.

---

## 📁 Project Structure

```text
ML-Project_Comment-Category-Prediction/
│
├── README.md
├── 24f1000556-notebook-t12026 (50).ipynb
├── train.csv
├── test.csv
├── Sample.csv
└── submission.csv
```

> Dataset files and submission files may be excluded from the Git repository depending on repository size and competition restrictions.

---

## 🧰 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- LightGBM
- Matplotlib
- Seaborn

### Machine Learning Techniques

- Exploratory Data Analysis
- Data preprocessing
- Feature engineering
- TF-IDF
- N-gram modeling
- Sparse feature matrices
- ColumnTransformer
- Logistic Regression
- Linear SVM
- LightGBM
- Early stopping
- Stratified train-validation split
- Macro F1 evaluation

---

## 📌 Key Takeaways

- Combining **textual and structured features** provided a strong representation of the dataset.
- Character-level TF-IDF complemented word-level TF-IDF by capturing spelling and character-level patterns.
- Feature engineering introduced useful behavioral and temporal signals.
- **LightGBM performed better than Logistic Regression and Linear SVM** on the validation set.
- The imbalance between the four classes made **Macro F1** a more informative metric than accuracy alone.
- The minority class, class 3, remained the primary challenge for the final model.

---

## 📚 Project Milestones

The notebook also contains the work completed across the project milestones:

- **Milestone 1:** Dataset exploration and basic analysis
- **Milestone 2:** Text processing and TF-IDF-based feature extraction
- **Milestone 3:** Preprocessing and baseline classification experiments
- **Milestone 4:** Dimensionality reduction and additional ML experiments
- **Milestone 5:** Additional model evaluation and hyperparameter experiments
- **Final:** Feature engineering, TF-IDF combination, model comparison, LightGBM optimization, and final submission

The milestone implementations are retained in the notebook appendix.

---

## 👤 Author

**Nikhilesh**

IIT Madras — BS in Data Science and Applications

---

## 📄 License

This project is intended for **academic and educational purposes**.
