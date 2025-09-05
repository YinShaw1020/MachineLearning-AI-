import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import time
import joblib
import os

# Models / preprocessing (kept to preserve model behavior)
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier

# =========================
# Manual metric/stat helpers
# =========================

def cm2x2(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    # Classes: 0 (No Disease), 1 (Disease)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return np.array([[tn, fp],
                     [fn, tp]]), (tn, fp, fn, tp)

def acc(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())

def prec(y_true, y_pred, positive=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == positive) & (y_pred == positive))
    fp = np.sum((y_true != positive) & (y_pred == positive))
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

def rec(y_true, y_pred, positive=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == positive) & (y_pred == positive))
    fn = np.sum((y_true == positive) & (y_pred != positive))
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

def f1(y_true, y_pred, positive=1):
    p = prec(y_true, y_pred, positive)
    r = rec(y_true, y_pred, positive)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

def classification_report_text(y_true, y_pred):
    # Two classes: 0 and 1
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    lines = []
    header = "              precision    recall  f1-score   support"
    lines.append(header)
    for cls in [0, 1]:
        p = prec(y_true, y_pred, positive=cls)
        r = rec(y_true, y_pred, positive=cls)
        f = f1(y_true, y_pred, positive=cls)
        s = int(np.sum(y_true == cls))
        lines.append(f"{cls:>14} {p:10.2f} {r:8.2f} {f:9.2f} {s:10d}")
    overall_acc = acc(y_true, y_pred)
    lines.append("")
    lines.append(f"accuracy{overall_acc:>24.2f} {len(y_true):10d}")
    # Macro avg
    pm = 0.5 * (prec(y_true, y_pred, 0) + prec(y_true, y_pred, 1))
    rm = 0.5 * (rec(y_true, y_pred, 0) + rec(y_true, y_pred, 1))
    fm = 0.5 * (f1(y_true, y_pred, 0) + f1(y_true, y_pred, 1))
    sup = len(y_true)
    lines.append(f"{'macro avg':>14} {pm:10.2f} {rm:8.2f} {fm:9.2f} {sup:10d}")
    # Weighted avg
    w0 = np.sum(y_true == 0) / sup if sup else 0
    w1 = np.sum(y_true == 1) / sup if sup else 0
    pw = w0 * prec(y_true, y_pred, 0) + w1 * prec(y_true, y_pred, 1)
    rw = w0 * rec(y_true, y_pred, 0) + w1 * rec(y_true, y_pred, 1)
    fw = w0 * f1(y_true, y_pred, 0) + w1 * f1(y_true, y_pred, 1)
    lines.append(f"{'weighted avg':>14} {pw:10.2f} {rw:8.2f} {fw:9.2f} {sup:10d}")
    return "\n".join(lines)

def roc_curve_manual(y_true, y_score):
    # y_true in {0,1}; y_score = probability or decision score for class 1
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    # Sort by descending score
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    # Unique thresholds include +/- inf endpoints to match sklearn behavior
    thresholds = np.r_[np.inf, np.unique(y_score_sorted)[::-1], -np.inf]

    P = np.sum(y_true_sorted == 1)
    N = np.sum(y_true_sorted == 0)
    tpr_list = []
    fpr_list = []

    # Vectorized cum-sums
    tp_cum = np.cumsum(y_true_sorted == 1)
    fp_cum = np.cumsum(y_true_sorted == 0)

    # For each threshold, find index where score < threshold
    for thr in thresholds:
        idx = np.searchsorted(-y_score_sorted, -thr, side='right')
        tp = tp_cum[idx - 1] if idx > 0 else 0
        fp = fp_cum[idx - 1] if idx > 0 else 0
        tpr_list.append(tp / P if P > 0 else 0.0)
        fpr_list.append(fp / N if N > 0 else 0.0)

    return np.array(fpr_list), np.array(tpr_list), thresholds

def auc_trapz(x, y):
    # Standard trapezoidal rule, identical to numpy.trapz
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.trapz(y, x))

def plot_confusion_matrix(ax, cm, title):
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title)
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(["No Disease", "Disease"])
    ax.set_yticklabels(["No Disease", "Disease"])
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    # Write counts
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

def vif_manual(X_scaled, feature_names):
    """
    Compute VIF for each column using OLS via numpy lstsq.
    VIF_i = 1 / (1 - R^2_i) where R^2_i is from regressing Xi on X_-i.
    """
    Xs = np.asarray(X_scaled, dtype=float)
    n, p = Xs.shape
    out_rows = []
    for i in range(p):
        y = Xs[:, i]
        X_others = np.delete(Xs, i, axis=1)
        # Add bias
        X_design = np.c_[np.ones((n, 1)), X_others]
        beta, residuals, rank, s = np.linalg.lstsq(X_design, y, rcond=None)
        y_pred = X_design @ beta
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        vif = 1.0 / (1.0 - r2) if (1.0 - r2) > 1e-12 else np.inf
        out_rows.append((feature_names[i], vif))
    return pd.DataFrame(out_rows, columns=["feature", "VIF"])


# =========================
# Data prep helper (original)
# =========================
@st.cache_data
def get_clean_scaled_data():
    df = pd.read_csv("heart.csv").drop_duplicates().dropna()

    def is_outlier_iqr(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        return (series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)

    continuous_cols = ['age', 'trestbps', 'thalach', 'oldpeak']
    for col in continuous_cols:
        df = df[~is_outlier_iqr(df[col])]

    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    return X_train_smote, X_test, y_train_smote, y_test, X.columns


st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

# ========== Sidebar ==========
st.sidebar.title("🫀 Heart Disease Comparison")
st.sidebar.markdown("Upload your dataset and explore model performance.📊")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("heart.csv")

st.sidebar.write(f"Dataset shape: {df.shape}")

# ========== Page Tabs ==========
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Preprocessing",
    "🪭 KNN", 
    "👙 Logistic Regression", 
    "👟 SVM", 
    "📊 Model Comparison"
])

with tab1:
    st.header("🔍 Preprocessing: Missing Values, Outlier, Overfitting")

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📈 Dataset Information")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)

    st.subheader("🧬 Data Types and Unique Values")
    feature_info = pd.DataFrame({
        "Data Type": df.dtypes,
        "Unique Values": df.nunique(),
        "Missing Values": df.isnull().sum()
    })
    st.dataframe(feature_info)

    st.subheader("🔢 Summary Statistics")
    st.dataframe(df.describe())

    st.subheader("🧩 Target Distribution (Pie Chart)")
    target_counts = df['target'].value_counts()
    labels = ['No Heart Disease' if i == 0 else 'Heart Disease' for i in target_counts.index]
    sizes = target_counts.values
    if len(sizes) > 0:
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
    else:
        st.warning("⚠️ No data available to display the pie chart.")

    # ==================== DATA CLEANING ====================
    st.subheader("Clean the Dataset")

    original_rows = df.shape[0]
    duplicate_count = df.duplicated().sum()
    df = df.drop_duplicates()

    missing_count = df.isnull().sum().sum()
    df = df.dropna()

    def is_outlier_iqr(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return (series < lower) | (series > upper)

    continuous_cols = ['age', 'trestbps', 'thalach', 'oldpeak']
    outlier_mask = pd.Series([False] * df.shape[0], index=df.index)
    for col in continuous_cols:
        outlier_mask = outlier_mask | is_outlier_iqr(df[col])

    outlier_count = outlier_mask.sum()
    df = df[~outlier_mask]
    final_rows = df.shape[0]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Original", original_rows)
    col2.metric("Duplicates", duplicate_count)
    col3.metric("Missing", missing_count)
    col4.metric("Outliers", outlier_count)
    col5.metric("Final Rows", final_rows)

    with st.expander("📄 View Cleaned Data"):
        st.dataframe(df)

    st.subheader("🔁 Multicollinearity Check (VIF)")

    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    vif_data = vif_manual(X_scaled, list(X.columns))
    st.dataframe(vif_data)

    # ==================== SMOTE + SCALING ====================
    st.subheader("Apply SMOTE and Standardization")

    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

    st.subheader("⚖️ Class Distribution Before & After SMOTE")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Before SMOTE")
        fig1, ax1 = plt.subplots()
        sns.countplot(x=y, ax=ax1, palette="pastel")
        st.pyplot(fig1)

    with col2:
        st.markdown("### After SMOTE")
        fig2, ax2 = plt.subplots()
        sns.countplot(x=y_balanced, ax=ax2, palette="muted")
        st.pyplot(fig2)

with tab2:
    st.header("🧠 KNN Pipeline")

    # Reuse cleaned/balanced from tab1 scope (safe to recompute quickly)
    df_knn = pd.read_csv("heart.csv").drop_duplicates().dropna()
    # remove outliers
    def is_outlier_iqr(series):
        Q1 = series.quantile(0.25); Q3 = series.quantile(0.75); IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR; upper = Q3 + 1.5 * IQR
        return (series < lower) | (series > upper)
    for c in ['age', 'trestbps', 'thalach', 'oldpeak']:
        df_knn = df_knn[~is_outlier_iqr(df_knn[c])]

    X = df_knn.drop("target", axis=1); y = df_knn["target"]
    X_scaled = StandardScaler().fit_transform(X)
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

    # ==================== BEST K SEARCH ====================
    st.subheader("Find Best K for KNN")

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    k_range = range(1, 21)
    cv_scores = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_balanced, y_balanced, cv=cv, scoring='accuracy')
        cv_scores.append(scores.mean())

    best_k = k_range[int(np.argmax(cv_scores))]
    best_score = float(np.max(cv_scores))

    fig, ax = plt.subplots()
    ax.plot(list(k_range), cv_scores, marker='o')
    ax.set_xlabel("Number of Neighbors (K)")
    ax.set_ylabel("Cross-Validated Accuracy")
    ax.set_title("KNN Accuracy vs. K")
    ax.grid(True)
    st.pyplot(fig)

    st.success(f"🏆 Best K = **{best_k}** with Accuracy = **{best_score:.4f}**")

    # ==================== MODEL TRAINING + TEST SPLIT ====================
    st.subheader("")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test)

    # ==================== EVALUATION ====================
    st.subheader("Evaluation Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc(y_test, y_pred):.2f}")
    c2.metric("Precision", f"{prec(y_test, y_pred):.2f}")
    c3.metric("Recall", f"{rec(y_test, y_pred):.2f}")
    c4.metric("F1 Score", f"{f1(y_test, y_pred):.2f}")

    with st.expander("📄 Classification Report"):
        st.code(classification_report_text(y_test, y_pred), language="text")

    cm, _ = cm2x2(y_test, y_pred)
    fig, ax = plt.subplots()
    plot_confusion_matrix(ax, cm, "Confusion Matrix - Logistic Regression")
    st.pyplot(fig)

with tab3:
    st.header("📈 Logistic Regression Analysis")

    # Load Data
    df_lr = pd.read_csv("heart.csv")

    # --- 1. Correlation Matrix ---
    st.subheader("🔍 Correlation Matrix")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_lr.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)

    # --- 2. Feature Scaling Visualization ---
    st.subheader("📊 Feature Scaling (Before vs After)")
    features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    original_data = df_lr[features]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(original_data)
    scaled_df = pd.DataFrame(scaled_data, columns=features)

    fig_scale, axs = plt.subplots(2, 3, figsize=(16, 8))
    axs = axs.flatten()
    for i, col in enumerate(features):
        sns.kdeplot(original_data[col], label='Before Scaling', ax=axs[i])
        sns.kdeplot(scaled_df[col], label='After Scaling', ax=axs[i])
        axs[i].set_title(col)
        axs[i].legend()
    axs[-1].axis('off')  # hide extra subplot
    st.pyplot(fig_scale)

    # --- 3. Data Cleaning ---
    st.subheader("")
    original_rows = df_lr.shape[0]

    # Remove duplicates
    duplicate_count = df_lr.duplicated().sum()
    df_lr = df_lr.drop_duplicates()

    # Remove missing values
    missing_count = df_lr.isnull().sum().sum()
    df_lr = df_lr.dropna()

    # Remove outliers using IQR (excluding 'chol')
    def is_outlier_iqr(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return (series < lower) | (series > upper)

    continuous_cols = ['age', 'trestbps', 'thalach', 'oldpeak']
    outlier_mask = pd.Series([False] * df_lr.shape[0], index=df_lr.index)
    for col in continuous_cols:
        outlier_mask |= is_outlier_iqr(df_lr[col])
    df_lr = df_lr[~outlier_mask]

    cleaned_rows = df_lr.shape[0]

    # --- 4. Model Training ---
    st.subheader("")
    X = df_lr.drop("target", axis=1)
    y = df_lr["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_smote, y_train_smote)

    y_pred = model.predict(X_test)

    # --- 5. Evaluation ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc(y_test, y_pred):.2f}")
    c2.metric("Precision", f"{prec(y_test, y_pred):.2f}")
    c3.metric("Recall", f"{rec(y_test, y_pred):.2f}")
    c4.metric("F1 Score", f"{f1(y_test, y_pred):.2f}")

    with st.expander("📄 Classification Report"):
        st.code(classification_report_text(y_test, y_pred), language="text")

    st.header("Logistic Regression Evaluation")
    cm, _ = cm2x2(y_test, y_pred)
    fig, ax = plt.subplots()
    plot_confusion_matrix(ax, cm, "Confusion Matrix - Logistic Regression")
    st.pyplot(fig)

with tab4:
    st.header("🧠 Support Vector Machine (SVM) Classification")

    X_train_smote, X_test, y_train_smote, y_test, feature_names = get_clean_scaled_data()

    svm_model = SVC(C=0.1, kernel='rbf', random_state=42)
    svm_model.fit(X_train_smote, y_train_smote)
    y_pred = svm_model.predict(X_test)

    st.subheader("📊 Evaluation Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc(y_test, y_pred):.2f}")
    col2.metric("Precision", f"{prec(y_test, y_pred):.2f}")
    col3.metric("Recall", f"{rec(y_test, y_pred):.2f}")
    col4.metric("F1 Score", f"{f1(y_test, y_pred):.2f}")

    st.text("📄 Classification Report")
    st.code(classification_report_text(y_test, y_pred), language='text')

    #  Confusion Matrix
    st.subheader("📉 Confusion Matrix")
    cm, _ = cm2x2(y_test, y_pred)
    fig, ax = plt.subplots()
    plot_confusion_matrix(ax, cm, "Confusion Matrix - SVM")
    st.pyplot(fig)

    st.subheader("🎯 SVM Decision Boundary (PCA Projection)")
    X_pca = PCA(n_components=2).fit_transform(X_train_smote)
    y_train_vis = y_train_smote

    clf_vis = SVC(kernel='rbf', C=1.0, random_state=42)
    clf_vis.fit(X_pca, y_train_vis)

    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    Z = clf_vis.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train_vis, cmap=plt.cm.coolwarm, edgecolors='k')
    legend_labels = ['No Disease', 'Heart Disease']
    ax.legend(handles=scatter.legend_elements()[0], labels=legend_labels)
    ax.set_title("SVM Decision Boundary (Training Data in PCA Space)")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.grid(True)
    st.pyplot(fig)

with tab5:
    st.header("📊 Model Comparison")

    # Load and clean dataset
    df_cmp = pd.read_csv("heart.csv")
    df_cmp = df_cmp.drop_duplicates().dropna()

    # Remove outliers (except 'chol')
    def is_outlier_iqr(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return (series < lower) | (series > upper)

    continuous_cols = ['age', 'trestbps', 'thalach', 'oldpeak']
    outlier_mask = pd.Series([False] * df_cmp.shape[0], index=df_cmp.index)
    for col in continuous_cols:
        outlier_mask = outlier_mask | is_outlier_iqr(df_cmp[col])
    df_cmp = df_cmp[~outlier_mask]

    # Split
    X = df_cmp.drop("target", axis=1)
    y = df_cmp["target"]
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Models
    models = {
        "KNN (k=13)": KNeighborsClassifier(n_neighbors=13),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM (C=0.1)": SVC(kernel='rbf', C=0.1, random_state=42, probability=True)
    }

    results = []
    fitted = {}
    for name, model in models.items():
        model.fit(X_train_smote, y_train_smote)
        fitted[name] = model
        y_pred = model.predict(X_test)
        results.append({
            "Model": name,
            "Accuracy": acc(y_test, y_pred),
            "Precision": prec(y_test, y_pred),
            "Recall": rec(y_test, y_pred),
            "F1 Score": f1(y_test, y_pred)
        })

    df_results = pd.DataFrame(results)

    # Display table
    st.dataframe(df_results.style.format({
        "Accuracy": "{:.3f}",
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1 Score": "{:.3f}"
    }))

    # Plot
    st.subheader("🔍 Metric Comparison")
    df_melted = df_results.melt(id_vars="Model", var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette="Set2")
    plt.title("Model Performance Comparison")
    plt.ylim(0.75, 1.0)
    plt.xticks(rotation=15)
    plt.tight_layout()
    st.pyplot(fig)

    # Plot ROC curves (manual)
    st.subheader("📈 ROC Curves")
    plt.figure(figsize=(8, 6))
    for name, model in fitted.items():
        # Use predict_proba if available; otherwise decision_function
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)
            # scale to [0,1] for nicer display (optional; rankings preserved)
            y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-12)
        fpr, tpr, _ = roc_curve_manual(y_test, y_score)
        roc_auc = auc_trapz(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.grid(True)
    st.pyplot(plt.gcf())

    st.subheader("🧮 Confusion Matrices")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, model) in zip(axes, fitted.items()):
        y_pred = model.predict(X_test)
        cm, _ = cm2x2(y_test, y_pred)
        plot_confusion_matrix(ax, cm, name)
    plt.tight_layout()
    st.pyplot(fig)

    # Show last cm counts (to match your original display)
    # (Using the last model's cm)
    y_pred = list(fitted.values())[-1].predict(X_test)
    cm, (tn, fp, fn, tp) = cm2x2(y_test, y_pred)
    st.write(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

    st.subheader("🥇 Best Performing Model")
    best_row = df_results.loc[df_results["F1 Score"].idxmax()]
    st.success(f"**{best_row['Model']}** performed best with F1 Score: **{best_row['F1 Score']:.3f}**")

    # Efficiency meta
    model_sizes = {}
    inference_times = {}
    for name, model in fitted.items():
        filename = f"{name.replace(' ', '_')}.joblib"
        joblib.dump(model, filename)
        model_sizes[name] = os.path.getsize(filename) / 1024  # KB
        start = time.time()
        model.predict(X_test)
        inference_times[name] = (time.time() - start) * 1000  # ms

    df_meta = pd.DataFrame({
        "Model": list(fitted.keys()),
        "Model Size (KB)": [model_sizes[m] for m in fitted.keys()],
        "Inference Time (ms)": [inference_times[m] for m in fitted.keys()]
    })

    st.subheader("⚙️ Model Efficiency")
    st.dataframe(df_meta.style.format({
        "Model Size (KB)": "{:.2f}",
        "Inference Time (ms)": "{:.2f}"
    }))

    st.subheader("⬇️Download")
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    csv = convert_df(df_results)
    st.download_button("📥 Download Model Metrics", data=csv, file_name='model_comparison.csv', mime='text/csv')
