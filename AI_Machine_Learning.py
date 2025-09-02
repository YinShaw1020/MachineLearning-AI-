import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import time
import os

# =========================
# From-scratch utilities
# =========================

def standardize_fit(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    return mean, std

def standardize_transform(X, mean, std):
    return (X - mean) / std

def iqr_mask(series):
    Q1 = np.percentile(series, 25)
    Q3 = np.percentile(series, 75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (series < lower) | (series > upper)

def stratified_train_test_split(X, y, test_size=0.2, random_state=42):
    rng = np.random.default_rng(random_state)
    X = np.asarray(X)
    y = np.asarray(y)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)
    n_pos_test = int(len(idx_pos) * test_size)
    n_neg_test = int(len(idx_neg) * test_size)
    test_idx = np.concatenate([idx_pos[:n_pos_test], idx_neg[:n_neg_test]])
    train_idx = np.concatenate([idx_pos[n_pos_test:], idx_neg[n_neg_test:]])
    rng.shuffle(test_idx)
    rng.shuffle(train_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def confusion_matrix_np(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[tn, fp], [fn, tp]])

def accuracy_score_np(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)

def precision_score_np(y_true, y_pred):
    cm = confusion_matrix_np(y_true, y_pred)
    tp = cm[1,1]; fp = cm[0,1]
    return tp / (tp + fp + 1e-12)

def recall_score_np(y_true, y_pred):
    cm = confusion_matrix_np(y_true, y_pred)
    tp = cm[1,1]; fn = cm[1,0]
    return tp / (tp + fn + 1e-12)

def f1_score_np(y_true, y_pred):
    p = precision_score_np(y_true, y_pred)
    r = recall_score_np(y_true, y_pred)
    return 2*p*r / (p + r + 1e-12)

def classification_report_np(y_true, y_pred):
    cm = confusion_matrix_np(y_true, y_pred)
    p = precision_score_np(y_true, y_pred)
    r = recall_score_np(y_true, y_pred)
    f1 = f1_score_np(y_true, y_pred)
    acc = accuracy_score_np(y_true, y_pred)
    return (
        f"Accuracy: {acc:.4f}\n"
        f"Precision: {p:.4f}\n"
        f"Recall: {r:.4f}\n"
        f"F1-score: {f1:.4f}\n"
        f"Confusion Matrix:\n{cm}"
    )

def roc_curve_np(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    order = np.argsort(-scores)
    y = y_true[order]
    P = np.sum(y == 1)
    N = np.sum(y == 0)
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    tpr = tps / (P + 1e-12)
    fpr = fps / (N + 1e-12)
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    return fpr, tpr

def auc_np(fpr, tpr):
    return np.trapz(tpr, fpr)

# ---------- PCA via SVD ----------
def pca_fit_transform(X, n_components=2):
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:n_components]
    X_pca = Xc @ components.T
    return X_pca, components, Xc.mean(axis=0)

# ---------- VIF (no statsmodels) ----------
def vif_table(X, feature_names):
    X = np.asarray(X)
    n, p = X.shape
    out = []
    for j in range(p):
        y = X[:, j]
        X_others = np.delete(X, j, axis=1)
        Xo = np.hstack([np.ones((n,1)), X_others])
        beta = np.linalg.pinv(Xo) @ y
        yhat = Xo @ beta
        ss_res = np.sum((y - yhat)**2)
        ss_tot = np.sum((y - y.mean())**2) + 1e-12
        r2 = 1 - ss_res/ss_tot
        vif = 1.0 / (1 - r2 + 1e-12)
        out.append(vif)
    return pd.DataFrame({"feature": feature_names, "VIF": out})

# ---------- Simple SMOTE (k=5) ----------
def simple_smote(X, y, k=5, random_state=42):
    rng = np.random.default_rng(random_state)
    X = np.asarray(X); y = np.asarray(y)
    X_pos = X[y==1]
    X_neg = X[y==0]
    n_pos, n_neg = len(X_pos), len(X_neg)
    if n_pos == n_neg:
        return X.copy(), y.copy()
    if n_pos < n_neg:
        minority = X_pos; minority_label = 1
        need = n_neg - n_pos
    else:
        minority = X_neg; minority_label = 0
        need = n_pos - n_neg
    if len(minority) <= 1:
        idx = rng.integers(0, len(minority), size=need)
        synth = minority[idx]
    else:
        from numpy.linalg import norm
        neighbors = []
        for i in range(len(minority)):
            d = norm(minority - minority[i], axis=1)
            nn_idx = np.argsort(d)[1:k+1] if len(minority) > k else np.argsort(d)[1:]
            neighbors.append(nn_idx)
        synth = []
        for _ in range(need):
            i = rng.integers(0, len(minority))
            nbrs = neighbors[i]
            j = nbrs[rng.integers(0, len(nbrs))]
            lam = rng.random()
            synth_vec = minority[i] + lam * (minority[j] - minority[i])
            synth.append(synth_vec)
        synth = np.vstack(synth)
    X_new = np.vstack([X, synth])
    y_new = np.concatenate([y, np.full(len(synth), minority_label)])
    return X_new, y_new

# ---------- Stratified K-Fold ----------
def stratified_kfold_indices(y, n_splits=10, shuffle=True, random_state=42):
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    idx_pos = np.where(y==1)[0].tolist()
    idx_neg = np.where(y==0)[0].tolist()
    if shuffle:
        rng.shuffle(idx_pos); rng.shuffle(idx_neg)
    folds = [[] for _ in range(n_splits)]
    for i, idx in enumerate(idx_pos):
        folds[i % n_splits].append(idx)
    for i, idx in enumerate(idx_neg):
        folds[i % n_splits].append(idx)
    return [np.array(sorted(f)) for f in folds]

# ---------- KNN (from scratch) ----------
class KNNClassifier:
    def __init__(self, n_neighbors=5):
        self.k = n_neighbors
        self.X = None
        self.y = None
    def fit(self, X, y):
        self.X = np.asarray(X); self.y = np.asarray(y)
        return self
    def predict(self, X):
        X = np.asarray(X)
        from numpy.linalg import norm
        preds = []
        for x in X:
            d = norm(self.X - x, axis=1)
            nn = np.argsort(d)[:self.k]
            vote = np.mean(self.y[nn])
            preds.append(1 if vote >= 0.5 else 0)
        return np.array(preds)
    def predict_proba(self, X):
        X = np.asarray(X)
        from numpy.linalg import norm
        probs = []
        for x in X:
            d = norm(self.X - x, axis=1)
            nn = np.argsort(d)[:self.k]
            p = np.mean(self.y[nn])
            probs.append([1-p, p])
        return np.array(probs)

# ---------- Logistic Regression (GD) ----------
class LogisticRegressionScratch:
    def __init__(self, lr=0.1, epochs=2000, l2=0.0, random_state=42):
        self.lr = lr; self.epochs = epochs; self.l2 = l2
        self.w = None; self.b = 0.0
        self.rng = np.random.default_rng(random_state)
    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -30, 30)
        return 1.0/(1.0 + np.exp(-z))
    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y)
        n, d = X.shape
        self.w = self.rng.normal(scale=0.01, size=d)
        self.b = 0.0
        for _ in range(self.epochs):
            z = X @ self.w + self.b
            p = self._sigmoid(z)
            grad_w = (X.T @ (p - y))/n + self.l2*self.w
            grad_b = np.mean(p - y)
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self
    def predict_proba(self, X):
        X = np.asarray(X)
        z = X @ self.w + self.b
        p = self._sigmoid(z)
        return np.vstack([1-p, p]).T
    def predict(self, X):
        p = self.predict_proba(X)[:,1]
        return (p >= 0.5).astype(int)

# ---------- Kernel SVM (RBF) via simplified SMO ----------
class KernelSVMScratch:
    def __init__(self, C=0.1, gamma="auto", tol=1e-3, max_passes=10, max_iter=1000, random_state=42):
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.max_passes = max_passes
        self.max_iter = max_iter
        self.rng = np.random.default_rng(random_state)
        self.alphas = None
        self.b = 0.0
        self.X = None
        self.y = None
        self.K = None  # Gram matrix

    def _rbf(self, X1, X2, gamma_val):
        X1_sq = np.sum(X1**2, axis=1)[:, None]
        X2_sq = np.sum(X2**2, axis=1)[None, :]
        dist2 = X1_sq + X2_sq - 2 * (X1 @ X2.T)
        return np.exp(-gamma_val * np.clip(dist2, 0, None))

    def _compute_gamma(self, X):
        if isinstance(self.gamma, str) and self.gamma == "auto":
            return 1.0 / X.shape[1]
        return float(self.gamma)

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, int)
        y2 = np.where(y == 1, 1.0, -1.0)

        n = X.shape[0]
        self.alphas = np.zeros(n)
        self.b = 0.0
        self.X = X
        self.y = y2

        gamma_val = self._compute_gamma(X)
        self.K = self._rbf(X, X, gamma_val)

        passes = 0
        iters = 0
        while passes < self.max_passes and iters < self.max_iter:
            num_changed = 0
            for i in range(n):
                Ei = self._f_i(i) - self.y[i]
                if (self.y[i]*Ei < -self.tol and self.alphas[i] < self.C) or (self.y[i]*Ei > self.tol and self.alphas[i] > 0):
                    j = i
                    while j == i:
                        j = self.rng.integers(0, n)
                    Ej = self._f_i(j) - self.y[j]

                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]

                    if self.y[i] != self.y[j]:
                        L = max(0.0, alpha_j_old - alpha_i_old)
                        H = min(self.C, self.C + alpha_j_old - alpha_i_old)
                    else:
                        L = max(0.0, alpha_i_old + alpha_j_old - self.C)
                        H = min(self.C, alpha_i_old + alpha_j_old)
                    if L == H:
                        continue

                    eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                    if eta >= 0:
                        continue

                    self.alphas[j] -= self.y[j] * (Ei - Ej) / eta
                    if self.alphas[j] > H: self.alphas[j] = H
                    elif self.alphas[j] < L: self.alphas[j] = L

                    if abs(self.alphas[j] - alpha_j_old) < 1e-6:
                        continue

                    self.alphas[i] += self.y[i] * self.y[j] * (alpha_j_old - self.alphas[j])

                    b1 = (self.b - Ei
                          - self.y[i]*(self.alphas[i]-alpha_i_old)*self.K[i,i]
                          - self.y[j]*(self.alphas[j]-alpha_j_old)*self.K[i,j])
                    b2 = (self.b - Ej
                          - self.y[i]*(self.alphas[i]-alpha_i_old)*self.K[i,j]
                          - self.y[j]*(self.alphas[j]-alpha_j_old)*self.K[j,j])

                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = 0.5 * (b1 + b2)

                    num_changed += 1

            passes = passes + 1 if num_changed == 0 else 0
            iters += 1
        return self

    def _f_i(self, i):
        return np.sum(self.alphas * self.y * self.K[:, i]) + self.b

    def decision_function(self, X):
        X = np.asarray(X, float)
        gamma_val = self._compute_gamma(self.X)
        Kx = self._rbf(X, self.X, gamma_val)
        return Kx @ (self.alphas * self.y) + self.b

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)

# =========================
# Data pipeline (cached)
# =========================
@st.cache_data
def get_clean_scaled_data():
    df = pd.read_csv("heart.csv").drop_duplicates().dropna()
    cont_cols = ['age', 'trestbps', 'thalach', 'oldpeak']
    mask = np.zeros(len(df), dtype=bool)
    for c in cont_cols:
        mask |= iqr_mask(df[c].values)
    df = df.loc[~mask].copy()

    X = df.drop("target", axis=1).values.astype(float)
    y = df["target"].values.astype(int)

    mean, std = standardize_fit(X)
    X_scaled = standardize_transform(X, mean, std)

    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    X_train_smote, y_train_smote = simple_smote(X_train, y_train, k=5, random_state=42)

    return X_train_smote, X_test, y_train_smote, y_test, list(df.drop("target", axis=1).columns), mean, std

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

    continuous_cols = ['age', 'trestbps', 'thalach', 'oldpeak']
    outlier_mask = pd.Series([False] * df.shape[0], index=df.index)
    for col in continuous_cols:
        outlier_mask = outlier_mask | pd.Series(iqr_mask(df[col].values), index=df.index)
    outlier_count = int(outlier_mask.sum())
    df = df.loc[~outlier_mask].copy()
    final_rows = df.shape[0]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Original", original_rows)
    col2.metric("Duplicates", int(duplicate_count))
    col3.metric("Missing", int(missing_count))
    col4.metric("Outliers", outlier_count)
    col5.metric("Final Rows", final_rows)

    with st.expander("📄 View Cleaned Data"):
        st.dataframe(df)

    st.subheader("🔁 Multicollinearity Check (VIF)")
    X_vif = df.drop("target", axis=1).values.astype(float)
    mean_v, std_v = standardize_fit(X_vif)
    X_vif_scaled = standardize_transform(X_vif, mean_v, std_v)
    vif_df = vif_table(X_vif_scaled, df.drop("target", axis=1).columns.tolist())
    st.dataframe(vif_df)

    # ==================== SMOTE + SCALING ====================
    st.subheader("Apply SMOTE and Standardization")
    X_all = df.drop("target", axis=1).values.astype(float)
    y_all = df["target"].values.astype(int)
    mean_all, std_all = standardize_fit(X_all)
    X_scaled_all = standardize_transform(X_all, mean_all, std_all)
    X_balanced, y_balanced = simple_smote(X_scaled_all, y_all, k=5, random_state=42)

    st.subheader("⚖️ Class Distribution Before & After SMOTE")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Before SMOTE")
        fig1, ax1 = plt.subplots()
        sns.countplot(x=pd.Series(y_all, name="target"), ax=ax1)
        st.pyplot(fig1)
    with col2:
        st.markdown("### After SMOTE")
        fig2, ax2 = plt.subplots()
        sns.countplot(x=pd.Series(y_balanced, name="target"), ax=ax2)
        st.pyplot(fig2)

with tab2:
    st.header("🧠 KNN Pipeline")

    df_knn = (pd.read_csv(uploaded_file) if uploaded_file else pd.read_csv("heart.csv")).drop_duplicates().dropna()
    cont_cols = ['age', 'trestbps', 'thalach', 'oldpeak']
    mask = np.zeros(len(df_knn), dtype=bool)
    for c in cont_cols:
        mask |= iqr_mask(df_knn[c].values)
    df_knn = df_knn.loc[~mask].copy()

    X_all = df_knn.drop("target", axis=1).values.astype(float)
    y_all = df_knn["target"].values.astype(int)
    mean_all, std_all = standardize_fit(X_all)
    X_scaled_all = standardize_transform(X_all, mean_all, std_all)
    X_balanced, y_balanced = simple_smote(X_scaled_all, y_all, k=5, random_state=42)

    # ==================== BEST K SEARCH ====================
    st.subheader("Find Best K for KNN")
    k_range = range(1, 21)
    cv_scores = []
    folds = stratified_kfold_indices(y_balanced, n_splits=10, shuffle=True, random_state=42)
    for k in k_range:
        acc_fold = []
        for i in range(10):
            val_idx = folds[i]
            train_idx = np.setdiff1d(np.arange(len(y_balanced)), val_idx)
            knn = KNNClassifier(n_neighbors=k).fit(X_balanced[train_idx], y_balanced[train_idx])
            y_pred_val = knn.predict(X_balanced[val_idx])
            acc_fold.append(accuracy_score_np(y_balanced[val_idx], y_pred_val))
        cv_scores.append(np.mean(acc_fold))
    best_k = list(k_range)[int(np.argmax(cv_scores))]
    best_score = np.max(cv_scores)

    fig, ax = plt.subplots()
    ax.plot(list(k_range), cv_scores, marker='o')
    ax.set_xlabel("Number of Neighbors (K)")
    ax.set_ylabel("Cross-Validated Accuracy")
    ax.set_title("KNN Accuracy vs. K")
    ax.grid(True)
    st.pyplot(fig)

    st.success(f"🏆 Best K = **{best_k}** with Accuracy = **{best_score:.4f}**")

    # ==================== TRAIN/TEST ====================
    mean_t, std_t = standardize_fit(X_all)
    X_scaled = standardize_transform(X_all, mean_t, std_t)
    X_train, X_test, y_train, y_test = stratified_train_test_split(X_scaled, y_all, test_size=0.2, random_state=42)
    X_train_smote, y_train_smote = simple_smote(X_train, y_train, k=5, random_state=42)

    model = KNNClassifier(n_neighbors=best_k).fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test)

    # ==================== EVALUATION ====================
    st.subheader("Evaluation Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{accuracy_score_np(y_test, y_pred):.2f}")
    c2.metric("Precision", f"{precision_score_np(y_test, y_pred):.2f}")
    c3.metric("Recall", f"{recall_score_np(y_test, y_pred):.2f}")
    c4.metric("F1 Score", f"{f1_score_np(y_test, y_pred):.2f}")

    with st.expander("📄 Classification Report"):
        st.code(classification_report_np(y_test, y_pred), language="text")

    cm = confusion_matrix_np(y_test, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix - KNN")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["No Disease", "Disease"])
    ax.set_yticklabels(["No Disease", "Disease"])
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha='center', va='center')
    plt.colorbar(im, ax=ax)
    st.pyplot(fig)

with tab3:
    st.header("📈 Logistic Regression Analysis")

    df_lr = (pd.read_csv(uploaded_file) if uploaded_file else pd.read_csv("heart.csv")).drop_duplicates().dropna()

    # --- 1. Correlation Matrix ---
    st.subheader("🔍 Correlation Matrix")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_lr.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)

    # --- 2. Feature Scaling Visualization ---
    st.subheader("📊 Feature Scaling (Before vs After)")
    features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    original_data = df_lr[features].values.astype(float)
    m0, s0 = standardize_fit(original_data)
    scaled_df = standardize_transform(original_data, m0, s0)
    fig_scale, axs = plt.subplots(2, 3, figsize=(16, 8))
    axs = axs.flatten()
    for i, col in enumerate(features):
        sns.kdeplot(original_data[:, i], label='Before Scaling', ax=axs[i])
        sns.kdeplot(scaled_df[:, i], label='After Scaling', ax=axs[i])
        axs[i].set_title(col)
        axs[i].legend()
    axs[-1].axis('off')
    st.pyplot(fig_scale)

    # --- 3. Data Cleaning ---
    st.subheader("")
    original_rows = df_lr.shape[0]
    duplicate_count = df_lr.duplicated().sum()
    df_lr = df_lr.drop_duplicates()
    missing_count = df_lr.isnull().sum().sum()
    df_lr = df_lr.dropna()

    cont_cols = ['age', 'trestbps', 'thalach', 'oldpeak']
    mask = np.zeros(len(df_lr), dtype=bool)
    for c in cont_cols:
        mask |= iqr_mask(df_lr[c].values)
    df_lr = df_lr.loc[~mask].copy()
    cleaned_rows = df_lr.shape[0]

    # --- 4. Model Training ---
    st.subheader("")
    X = df_lr.drop("target", axis=1).values.astype(float)
    y = df_lr["target"].values.astype(int)
    mean_s, std_s = standardize_fit(X)
    X_scaled = standardize_transform(X, mean_s, std_s)

    X_train, X_test, y_train, y_test = stratified_train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train_smote, y_train_smote = simple_smote(X_train, y_train, k=5, random_state=42)

    model = LogisticRegressionScratch(lr=0.1, epochs=2000, l2=0.001, random_state=42).fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test)

    # --- 5. Evaluation ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{accuracy_score_np(y_test, y_pred):.2f}")
    c2.metric("Precision", f"{precision_score_np(y_test, y_pred):.2f}")
    c3.metric("Recall", f"{recall_score_np(y_test, y_pred):.2f}")
    c4.metric("F1 Score", f"{f1_score_np(y_test, y_pred):.2f}")

    with st.expander("📄 Classification Report"):
        st.code(classification_report_np(y_test, y_pred), language="text")

    cm = confusion_matrix_np(y_test, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix - Logistic Regression")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["No Disease", "Disease"])
    ax.set_yticklabels(["No Disease", "Disease"])
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha='center', va='center')
    plt.colorbar(im, ax=ax)
    st.pyplot(fig)

with tab4:
    st.header("🧠 Support Vector Machine (SVM) Classification")
    X_train_smote, X_test, y_train_smote, y_test, feature_names, mean_glob, std_glob = get_clean_scaled_data()

    # RBF-kernel SVM with C=0.1
    svm_model = KernelSVMScratch(C=0.1, gamma="auto", tol=1e-3, max_passes=10, max_iter=1000, random_state=42)
    svm_model.fit(X_train_smote, y_train_smote)
    y_pred = svm_model.predict(X_test)

    st.subheader("📊 Evaluation Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{accuracy_score_np(y_test, y_pred):.2f}")
    c2.metric("Precision", f"{precision_score_np(y_test, y_pred):.2f}")
    c3.metric("Recall", f"{recall_score_np(y_test, y_pred):.2f}")
    c4.metric("F1 Score", f"{f1_score_np(y_test, y_pred):.2f}")

    st.text("📄 Classification Report")
    st.code(classification_report_np(y_test, y_pred), language='text')

    st.subheader("📉 Confusion Matrix")
    cm = confusion_matrix_np(y_test, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix - SVM (RBF, C=0.1)")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["No Disease", "Disease"])
    ax.set_yticklabels(["No Disease", "Disease"])
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha='center', va='center')
    plt.colorbar(im, ax=ax)
    st.pyplot(fig)

    st.subheader("🎯 SVM Decision Boundary (PCA Projection)")
    X_pca, comps, _ = pca_fit_transform(X_train_smote, n_components=2)
    y_train_vis = y_train_smote
    svm_vis = KernelSVMScratch(C=0.1, gamma="auto", max_passes=10, max_iter=1000, random_state=1).fit(X_pca, y_train_vis)

    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = svm_vis.predict(grid).reshape(xx.shape)

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

    df_cmp = (pd.read_csv(uploaded_file) if uploaded_file else pd.read_csv("heart.csv")).drop_duplicates().dropna()

    cont_cols = ['age', 'trestbps', 'thalach', 'oldpeak']
    mask = np.zeros(len(df_cmp), dtype=bool)
    for c in cont_cols:
        mask |= iqr_mask(df_cmp[c].values)
    df_cmp = df_cmp.loc[~mask].copy()

    X = df_cmp.drop("target", axis=1).values.astype(float)
    y = df_cmp["target"].values.astype(int)
    mean_c, std_c = standardize_fit(X)
    X_scaled = standardize_transform(X, mean_c, std_c)
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    X_train_smote, y_train_smote = simple_smote(X_train, y_train, k=5, random_state=42)

    # === Base models only (no voting) ===
    knn13 = KNNClassifier(n_neighbors=13)
    logreg = LogisticRegressionScratch(lr=0.1, epochs=2000, l2=0.001, random_state=42)
    svm_rbf = KernelSVMScratch(C=0.1, gamma="auto", tol=1e-3, max_passes=10, max_iter=1000, random_state=42)

    models = {
        "KNN (k=13)": knn13,
        "Logistic Regression": logreg,
        "SVM (RBF, C=0.1)": svm_rbf
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
            "Accuracy": accuracy_score_np(y_test, y_pred),
            "Precision": precision_score_np(y_test, y_pred),
            "Recall": recall_score_np(y_test, y_pred),
            "F1 Score": f1_score_np(y_test, y_pred)
        })

    df_results = pd.DataFrame(results)

    st.dataframe(df_results.style.format({
        "Accuracy": "{:.3f}",
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1 Score": "{:.3f}"
    }))

    st.subheader("🔍 Metric Comparison")
    df_melted = df_results.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric")
    plt.title("Model Performance Comparison")
    plt.ylim(0.75, 1.0)
    plt.xticks(rotation=15)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("📈 ROC Curves")
    plt.figure(figsize=(8, 6))
    for name, model in trained.items():
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            y_score = model.predict(X_test).astype(float)
        fpr, tpr = roc_curve_np(y_test, y_score)
        roc_auc = auc_np(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.grid(True)
    st.pyplot(plt)

    st.subheader("🧮 Confusion Matrices")
    n_models_for_cm = len(models)
    cols = 3
    rows = int(np.ceil(n_models_for_cm / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = np.array(axes).reshape(rows, cols)

    idx = 0
    for name, model in trained.items():
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        y_pred = model.predict(X_test)
        cm = confusion_matrix_np(y_test, y_pred)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["No", "Yes"])
        ax.set_yticklabels(["No", "Yes"])
        for (i,j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha='center', va='center', fontsize=10)
        idx += 1
    while idx < rows * cols:
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")
        idx += 1
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("🥇 Best Performing Model")
    best_model_row = df_results.loc[df_results["F1 Score"].idxmax()]
    st.success(f"**{best_model_row['Model']}** performed best with F1 Score: **{best_model_row['F1 Score']:.3f}**")

    # Efficiency (no pickling): parameter count + inference time
    def parameter_count(model):
        if isinstance(model, KNNClassifier):
            # stores full train set
            return int(getattr(model, "X", np.empty((0,0))).size + getattr(model, "y", np.empty((0,))).size)
        if isinstance(model, LogisticRegressionScratch):
            return int((0 if model.w is None else model.w.size) + 1)  # + bias
        if isinstance(model, KernelSVMScratch):
            # alphas + bias (support vectors are in X)
            return int((0 if model.alphas is None else model.alphas.size) + 1)
        return 0

    inference_times = {}
    params = {}
    for name, model in trained.items():
        start = time.time()
        _ = model.predict(X_test)
        inference_times[name] = (time.time() - start) * 1000.0
        params[name] = parameter_count(model)

    df_meta = pd.DataFrame({
        "Model": list(trained.keys()),
        "Parameter Count": [params[m] for m in trained.keys()],
        "Inference Time (ms)": [inference_times[m] for m in trained.keys()]
    })

    st.subheader("⚙️ Model Efficiency")
    st.dataframe(df_meta.style.format({
        "Inference Time (ms)": "{:.2f}"
    }))

    st.subheader("⬇️Download")
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    csv = convert_df(df_results)
    st.download_button("📥 Download Model Metrics", data=csv, file_name='model_comparison.csv', mime='text/csv')
