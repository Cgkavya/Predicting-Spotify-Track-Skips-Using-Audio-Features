# %%
# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# Load the data
df = pd.read_csv(r"C:\Users\Kavya\OneDrive\Documents\Data sets\spotifydata.csv")

# %%
df.head(10)

# %%
df.shape

# %%
df.columns.tolist()

# %%
df.info()

# %%
df.describe()

# %%
df.isnull().sum()

# %%
df = df.drop_duplicates()
df = df.dropna()
df = df.drop(['Unnamed: 0', 'artist_name', 'track_name', 'track_id'], axis=1)

# %%
def rule_based_skip(row):
    score = 0
    
    # Audio feature rules
    if row['danceability'] < 0.4:
        score += 1
    if row['energy'] < 0.4:
        score += 1
    if row['speechiness'] > 0.5:
        score += 1
    if row['valence'] < 0.3:
        score += 1
    if row['instrumentalness'] > 0.7:
        score += 1
    if row['duration_ms'] < 90000 or row['duration_ms'] > 300000:
        score += 1

    # Popularity-based rule
    if row['popularity'] < 40:
        score += 1
    elif row['popularity'] > 70:
        score -= 1

    return 1 if score > 0 else 0

# %%
#  Create the 'skipped' column
df['skipped'] = df.apply(rule_based_skip, axis=1)

# %%
df['skipped'].value_counts()

# %%
df.isnull().sum()

# %%
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="viridis")
plt.title("Correlation Matrix")
plt.show()

# %%
df

# %%
num_data = df.select_dtypes(include = "number")
cat_data = df.select_dtypes(include = "object")

# Identify categorical and numerical columns
num_cols = num_data.columns.tolist()
cat_cols = cat_data.columns.tolist()

print("numerical columns: ", num_cols)
print("categorical columns: ",cat_cols)

# %%
# Perform an outlier detection analysis on numerical variables (e.g., using the IQR method).
num_data.boxplot()
plt.xticks(rotation=45)
plt.show()

# %%
def remove_outliers(data, column_name):
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR
    data[column_name] = data[column_name].clip(upper = upper_bound)
    data[column_name] = data[column_name].clip(lower = lower_bound)
    return data[column_name]

# %%
for col in num_cols:
      num_data[col] = remove_outliers(num_data, col)

# %%
df.isnull().sum()

# %%
#Distribution plot of a few features
features_to_plot = ['danceability', 'energy', 'valence', 'popularity']
for col in features_to_plot:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# %%
#Skip Class Distribution
plt.figure(figsize=(5, 4))
sns.countplot(x='skipped', hue='skipped', data=df, palette='pastel', legend=False)
plt.title('Skip Class Distribution')
plt.xlabel('Skipped')
plt.ylabel('Count')
plt.xticks([0, 1], ['Not Skipped (0)', 'Skipped (1)'])
plt.tight_layout()
plt.show()

# %%
# Popularity Distribution
plt.figure(figsize=(6, 4))
sns.histplot(df['popularity'], kde=True, bins=20, color='skyblue')
plt.title('Popularity Distribution')
plt.xlabel('Popularity')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# %%
top_genres = df["genre"].value_counts().nlargest(10)
plt.figure(figsize=(8, 4))
sns.barplot(x=top_genres.index, y=top_genres.values, hue=top_genres.index, palette='muted', legend=False)
plt.xticks(rotation=45)
plt.title("Top 10 Genres")
plt.ylabel("Count")
plt.xlabel("Genre")
plt.tight_layout()
plt.show()

# %%
# Boxplot: Loudness by Skip Class
plt.figure(figsize=(6, 4))
sns.boxplot(x='skipped', y='loudness', hue='skipped', data=df, palette='Set2', legend=False)
plt.title('Loudness vs Skipped')
plt.xlabel('Skipped')
plt.ylabel('Loudness (dB)')
plt.tight_layout()
plt.show()

# %%
# Boxplot: Acousticness by Skip Class
plt.figure(figsize=(6, 4))
sns.boxplot(x='skipped', y='acousticness', hue='skipped', data=df, palette='Set3',legend=False)
plt.title('Acousticness by Skip Class')
plt.xlabel('Skipped')
plt.tight_layout()
plt.show()

# %%
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# %%
# Select features
selected_features = [
    "danceability", "genre", "energy", "speechiness", "valence",
    "instrumentalness", "duration_ms", "popularity"
]
X = df[selected_features].copy()
y = df["skipped"]

# Preprocessing
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(include="object").columns.tolist()

imputer_num = SimpleImputer(strategy="mean")
X[num_cols] = imputer_num.fit_transform(X[num_cols])

imputer_cat = SimpleImputer(strategy="most_frequent")
X[cat_cols] = imputer_cat.fit_transform(X[cat_cols])

encoder = LabelEncoder()
for col in cat_cols:
    X[col] = encoder.fit_transform(X[col].astype(str))

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# %%
# Apply SMOTE to handle imbalance
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_scaled, y)


# %%
# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# %%
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    "n_estimators": [100, 150],
    "max_depth": [10, 15],
    "min_samples_split": [2, 5],
    "class_weight": ["balanced"]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)

random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=4,               # Try 4 random combinations
    cv=3,
    scoring="f1",
    verbose=1,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
rf_model = random_search.best_estimator_

# %%
rf_pred = rf_model.predict(X_test)
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))
print("Best Parameters:", random_search.best_params_)


# %%
print("Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)


# %%
print("Training XGBoost...")
xgb_model = XGBClassifier(
    scale_pos_weight=(len(y_train) / sum(y_train)),
    eval_metric='logloss',
    n_estimators=100,
    max_depth=3,
    random_state=42
)

xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

# %%
models = {
    "Logistic Regression": (lr_model, lr_pred),
    "XGBoost": (xgb_model, xgb_pred),
    "Random Forest": (rf_model, rf_pred)
}

model_scores = {}

for name, (model, pred) in models.items():
    accuracy = accuracy_score(y_test, pred)
    model_scores[name] = accuracy
    print(f"\n{name} Accuracy: {round(accuracy, 4)}")
    print("Precision:", round(precision_score(y_test, pred), 4))
    print("Recall:", round(recall_score(y_test, pred), 4))
    print("F1 Score:", round(f1_score(y_test, pred), 4))

# %%
import pickle
from sklearn.model_selection import cross_val_score

# %%
best_model_name = max(model_scores, key=model_scores.get)
best_model = models[best_model_name][0]

print(f"\n Best Model: {best_model_name} with Accuracy: {model_scores[best_model_name]:.4f}")

# Retrieve final model from best_models dict
final_model = models[best_model_name][0]

#  5-fold cross-validation
cv_scores = cross_val_score(final_model, X_scaled, y, cv=5, scoring='accuracy',n_jobs=-1)

print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))
print("Standard Deviation:", np.std(cv_scores))

print("Train Accuracy:", final_model.score(X_train, y_train))
print("Test Accuracy:", final_model.score(X_test, y_test))

#  Save final model
with open("best_model.pkl", "wb") as f:
    pickle.dump(final_model, f)

#  Save encoder (if used)
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

#  Save feature column names
with open("model_features.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)    

print(" Final model, encoder, and feature names saved successfully.")

# %% [markdown]
# ##### 


