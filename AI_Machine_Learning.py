import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import time
import joblib
import os
from sklearn.ensemble import VotingClassifier


from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

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
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
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

   
    # ==================== BEST K SEARCH ====================
    st.subheader("Find Best K for KNN")

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    k_range = range(1, 21)
    cv_scores = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_balanced, y_balanced, cv=cv, scoring='accuracy')
        cv_scores.append(scores.mean())

    best_k = k_range[cv_scores.index(max(cv_scores))]
    best_score = max(cv_scores)

    fig, ax = plt.subplots()
    ax.plot(k_range, cv_scores, marker='o', color='green')
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

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred):.2f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
    col4.metric("F1 Score", f"{f1_score(y_test, y_pred):.2f}")

    with st.expander("📄 Classification Report"):
        st.code(classification_report(y_test, y_pred), language="text")

    
    

    # (Assuming y_test and y_pred_logreg already defined earlier)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Disease"]).plot(ax=ax)
    ax.set_title("Confusion Matrix - Logistic Regression")
    st.pyplot(fig)


with tab3:
    st.header("📈 Logistic Regression Analysis")

    # Load Data
    df = pd.read_csv("heart.csv")

    # --- 1. Correlation Matrix ---
    st.subheader("🔍 Correlation Matrix")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)

    # --- 2. Feature Scaling Visualization ---
    st.subheader("📊 Feature Scaling (Before vs After)")
    features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    original_data = df[features]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(original_data)
    scaled_df = pd.DataFrame(scaled_data, columns=features)

    fig_scale, axs = plt.subplots(2, 3, figsize=(16, 8))
    axs = axs.flatten()
    for i, col in enumerate(features):
        sns.kdeplot(original_data[col], label='Before Scaling', ax=axs[i], color='blue')
        sns.kdeplot(scaled_df[col], label='After Scaling', ax=axs[i], color='red')
        axs[i].set_title(col)
        axs[i].legend()
    axs[-1].axis('off')  # hide extra subplot
    st.pyplot(fig_scale)

    # --- 3. Data Cleaning ---
    st.subheader("")
    original_rows = df.shape[0]

    # Remove duplicates
    duplicate_count = df.duplicated().sum()
    df = df.drop_duplicates()

    # Remove missing values
    missing_count = df.isnull().sum().sum()
    df = df.dropna()

    # Remove outliers using IQR (excluding 'chol')
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
        outlier_mask |= is_outlier_iqr(df[col])
    df = df[~outlier_mask]

    cleaned_rows = df.shape[0]
    

    # --- 4. Model Training ---
    st.subheader("")
    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_smote, y_train_smote)

    y_pred = model.predict(X_test)

    # --- 5. Evaluation ---

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred):.2f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
    col4.metric("F1 Score", f"{f1_score(y_test, y_pred):.2f}")

    with st.expander("📄 Classification Report"):
        st.code(classification_report(y_test, y_pred), language="text")

    st.header("Logistic Regression Evaluation")

    # (Assuming y_test and y_pred_logreg already defined earlier)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Disease"]).plot(ax=ax)
    ax.set_title("Confusion Matrix - Logistic Regression")
    st.pyplot(fig)

with tab4:
    st.header("🧠 Support Vector Machine (SVM) Classification")

    X_train_smote, X_test, y_train_smote, y_test, feature_names = get_clean_scaled_data()

    
    svm_model = SVC(C=0.1, kernel='rbf', random_state=42)
    svm_model.fit(X_train_smote, y_train_smote)
    y_pred = svm_model.predict(X_test)

    
    st.subheader("📊 Evaluation Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred):.2f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
    col4.metric("F1 Score", f"{f1_score(y_test, y_pred):.2f}")

    st.text("📄 Classification Report")
    st.code(classification_report(y_test, y_pred), language='text')

    #  Confusion Matrix
    st.subheader("📉 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Disease"]).plot(ax=ax)
    ax.set_title("Confusion Matrix - SVM")
    st.pyplot(fig)

    
    st.subheader("🎯 SVM Decision Boundary (PCA Projection)")
    from sklearn.decomposition import PCA

    X_pca = PCA(n_components=2).fit_transform(X_train_smote)
    y_train_vis = y_train_smote

    clf_vis = SVC(kernel='rbf', C=1.0)
    clf_vis.fit(X_pca, y_train_vis)

    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    Z = clf_vis.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

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

    st.header("📊 Model Comparison (with Combined/Voting Models)")

    # Load and clean dataset
    df = pd.read_csv("heart.csv")
    df = df.drop_duplicates().dropna()

    # Remove outliers (except 'chol')
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
    df = df[~outlier_mask]

    # Split
    X = df.drop("target", axis=1)
    y = df["target"]
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # === Base models ===
    knn13 = KNeighborsClassifier(n_neighbors=13)
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    # IMPORTANT: probability=True so we can do soft voting + ROC
    svm_rbf = SVC(kernel='rbf', C=0.1, probability=True, random_state=42)

    # === Combined models ===
    voting_hard = VotingClassifier(
        estimators=[("knn", knn13), ("lr", logreg), ("svm", svm_rbf)],
        voting="hard"
    )
    voting_soft = VotingClassifier(
        estimators=[("knn", knn13), ("lr", logreg), ("svm", svm_rbf)],
        voting="soft"  # averages predicted probabilities
        # Optionally: weights=[1,1,1]  # or tune weights
    )

    models = {
        "KNN (k=13)": knn13,
        "Logistic Regression": logreg,
        "SVM (C=0.1)": svm_rbf,
        "Combined (Hard Vote)": voting_hard,
        "Combined (Soft Vote)": voting_soft
    }

    # Train & evaluate
    results = []
    trained = {}
    for name, model in models.items():
        model.fit(X_train_smote, y_train_smote)
        trained[name] = model
        y_pred = model.predict(X_test)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred)
        })

    df_results = pd.DataFrame(results)

    # Display table
    st.dataframe(df_results.style.format({
        "Accuracy": "{:.3f}",
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1 Score": "{:.3f}"
    }))

    # Plot metrics (bar)
    st.subheader("🔍 Metric Comparison")
    df_melted = df_results.melt(id_vars="Model", var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette="Set2")
    plt.title("Model Performance Comparison")
    plt.ylim(0.75, 1.0)
    plt.xticks(rotation=15)
    plt.tight_layout()
    st.pyplot(fig)

    # ROC curves (works for all with predict_proba; for hard vote we skip)
    st.subheader("📈 ROC Curves")
    plt.figure(figsize=(8, 6))
    for name, model in trained.items():
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            # Hard Vote has no score/proba -> skip from ROC
            continue
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.grid(True)
    st.pyplot(plt)

    # Confusion Matrices
    st.subheader("🧮 Confusion Matrices")
    n_models_for_cm = len(models)
    cols = 3
    rows = int(np.ceil(n_models_for_cm / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = np.array(axes).reshape(rows, cols)

    idx = 0
    last_cm = None
    for name, model in trained.items():
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        y_pred = model.predict(X_test)
        last_cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(name)
        idx += 1

    # Hide any empty subplots
    while idx < rows * cols:
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")
        idx += 1

    plt.tight_layout()
    st.pyplot(fig)

    if last_cm is not None:
        tn, fp, fn, tp = last_cm.ravel()
        st.write(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

    # Best model by F1
    st.subheader("🥇 Best Performing Model")
    best_model = df_results.loc[df_results["F1 Score"].idxmax()]
    st.success(f"**{best_model['Model']}** performed best with F1 Score: **{best_model['F1 Score']:.3f}**")

    # Save sizes + inference times
    model_sizes = {}
    inference_times = {}
    for name, model in trained.items():
        filename = f"{name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib"
        joblib.dump(model, filename)
        model_sizes[name] = os.path.getsize(filename) / 1024  # KB

        start = time.time()
        model.predict(X_test)
        inference_times[name] = (time.time() - start) * 1000  # ms

    df_meta = pd.DataFrame({
        "Model": list(trained.keys()),
        "Model Size (KB)": [model_sizes[m] for m in trained.keys()],
        "Inference Time (ms)": [inference_times[m] for m in trained.keys()]
    })

    st.subheader("⚙️ Model Efficiency")
    st.dataframe(df_meta.style.format({
        "Model Size (KB)": "{:.2f}",
        "Inference Time (ms)": "{:.2f}"
    }))

    # Download
    st.subheader("⬇️Download")
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(df_results)
    st.download_button("📥 Download Model Metrics", data=csv, file_name='model_comparison.csv', mime='text/csv')

