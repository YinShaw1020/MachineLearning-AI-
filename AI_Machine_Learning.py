import streamlit as st
import csv
import math
import random
import time
from collections import Counter, defaultdict
from io import StringIO

# -----------------------------
# Utilities (no third-party libs)
# -----------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def read_csv_to_rows(file_like) -> list:
    # file_like: file buffer or path string
    if isinstance(file_like, str):
        f = open(file_like, "r", newline="", encoding="utf-8")
        close_after = True
    else:
        f = StringIO(file_like.getvalue().decode("utf-8"))
        close_after = True
    rows = []
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)
    if close_after:
        f.close()
    return rows

def to_float(val):
    try:
        return float(val)
    except:
        return None

def drop_duplicates(rows: list) -> list:
    seen = set()
    out = []
    for r in rows:
        key = tuple((k, r[k]) for k in sorted(r.keys()))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out

def drop_missing(rows: list, required_cols: list) -> list:
    out = []
    for r in rows:
        ok = True
        for c in required_cols:
            if r.get(c) is None or r.get(c) == "" or to_float(r.get(c)) is None:
                ok = False
                break
        if ok:
            out.append(r)
    return out

def iqr_bounds(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return None, None
    def percentile(p):
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return s[int(k)]
        return s[f] + (s[c] - s[f]) * (k - f)
    Q1 = percentile(0.25)
    Q3 = percentile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return lower, upper

def filter_outliers(rows, colnames):
    # remove rows that are outliers for ANY of the given columns
    # compute bounds per column
    cols_vals = {c: [] for c in colnames}
    for r in rows:
        for c in colnames:
            v = to_float(r[c])
            if v is not None:
                cols_vals[c].append(v)
    bounds = {}
    for c in colnames:
        lo, hi = iqr_bounds(cols_vals[c])
        bounds[c] = (lo, hi)
    out = []
    for r in rows:
        keep = True
        for c in colnames:
            v = to_float(r[c])
            if v is None:
                keep = False
                break
            lo, hi = bounds[c]
            if v < lo or v > hi:
                keep = False
                break
        if keep:
            out.append(r)
    return out

def columns_and_types(rows):
    # infer numeric vs non-numeric
    if not rows: return [], {}
    cols = list(rows[0].keys())
    types = {}
    for c in cols:
        numeric = True
        for r in rows:
            if to_float(r[c]) is None:
                numeric = False
                break
        types[c] = "numeric" if numeric else "string"
    return cols, types

def as_Xy(rows, target_col):
    X_cols = [c for c in rows[0].keys() if c != target_col]
    X = []
    y = []
    for r in rows:
        feats = [to_float(r[c]) for c in X_cols]
        X.append(feats)
        y.append(int(float(r[target_col])))
    return X_cols, X, y

def standardize_fit(X):
    # X: list[list[float]]
    n = len(X)
    if n == 0: return [], []
    d = len(X[0])
    means = [0.0]*d
    stds  = [0.0]*d
    for j in range(d):
        s = sum(X[i][j] for i in range(n))
        m = s / n
        means[j] = m
        var = sum((X[i][j] - m)**2 for i in range(n)) / n
        stds[j] = math.sqrt(var) if var > 0 else 1.0
    return means, stds

def standardize_apply(X, means, stds):
    n = len(X)
    d = len(X[0]) if n else 0
    Z = [[0.0]*d for _ in range(n)]
    for i in range(n):
        for j in range(d):
            Z[i][j] = (X[i][j] - means[j]) / stds[j]
    return Z

def stratified_train_test_split(X, y, test_size=0.2, seed=RANDOM_SEED):
    random.seed(seed)
    idxs = list(range(len(y)))
    # group by class
    by_c = defaultdict(list)
    for i in idxs:
        by_c[y[i]].append(i)
    train_idx, test_idx = [], []
    for c, lst in by_c.items():
        random.shuffle(lst)
        n_test = max(1, int(len(lst)*test_size))
        test_idx += lst[:n_test]
        train_idx += lst[n_test:]
    def take(idxs_list):
        Xo, yo = [], []
        for i in idxs_list:
            Xo.append(X[i])
            yo.append(y[i])
        return Xo, yo
    return take(train_idx) + take(test_idx)

def oversample_simple(X, y, seed=RANDOM_SEED):
    # Duplicate minority class samples until balanced (simple ROS; not SMOTE)
    random.seed(seed)
    counts = Counter(y)
    classes = list(counts.keys())
    max_n = max(counts.values()) if counts else 0
    X_new, y_new = [], []
    buckets = defaultdict(list)
    for xi, yi in zip(X, y):
        buckets[yi].append(xi)
    for c in classes:
        lst = buckets[c]
        if not lst: continue
        k = len(lst)
        # copy originals
        for xi in lst:
            X_new.append(xi[:])
            y_new.append(c)
        # duplicate randomly to reach max_n
        while k < max_n:
            xi = random.choice(lst)
            X_new.append(xi[:])
            y_new.append(c)
            k += 1
    return X_new, y_new

# ---------- Metrics ----------
def confusion_matrix_counts(y_true, y_pred):
    tp=fp=tn=fn=0
    for t,p in zip(y_true, y_pred):
        if t==1 and p==1: tp+=1
        elif t==0 and p==1: fp+=1
        elif t==0 and p==0: tn+=1
        elif t==1 and p==0: fn+=1
    return tn, fp, fn, tp

def accuracy(y_true, y_pred):
    tn,fp,fn,tp = confusion_matrix_counts(y_true, y_pred)
    total = tn+fp+fn+tp
    return (tn+tp)/total if total>0 else 0.0

def precision(y_true, y_pred):
    tn,fp,fn,tp = confusion_matrix_counts(y_true, y_pred)
    den = (tp+fp)
    return tp/den if den>0 else 0.0

def recall(y_true, y_pred):
    tn,fp,fn,tp = confusion_matrix_counts(y_true, y_pred)
    den = (tp+fn)
    return tp/den if den>0 else 0.0

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2*p*r/(p+r) if p+r>0 else 0.0

def roc_curve_points(y_true, scores):
    # scores: probability/score for class=1
    # thresholds from unique scores
    pairs = sorted(zip(scores, y_true), key=lambda x: -x[0])
    thresholds = sorted(set(scores), reverse=True)
    points = []
    P = sum(1 for t in y_true if t==1)
    N = sum(1 for t in y_true if t==0)
    for thr in thresholds + [-1e9]:
        tp=fp=0
        for s,t in pairs:
            pred = 1 if s>=thr else 0
            if pred==1 and t==1: tp+=1
            if pred==1 and t==0: fp+=1
        tpr = tp/P if P>0 else 0.0
        fpr = fp/N if N>0 else 0.0
        points.append((fpr, tpr))
    # ensure (0,0) and (1,1) endpoints
    points = sorted(set(points))
    return points

def auc_trapezoid(points):
    # points sorted by FPR
    pts = sorted(points)
    area=0.0
    for i in range(1, len(pts)):
        x0,y0 = pts[i-1]
        x1,y1 = pts[i]
        area += (x1 - x0) * (y0 + y1) / 2.0
    return area

# ---------- Models (from scratch) ----------
# KNN
class KNN:
    def __init__(self, k=13):
        self.k = k
        self.X = []
        self.y = []

    def fit(self, X, y):
        self.X = X
        self.y = y

    @staticmethod
    def _dist(a, b):
        return math.sqrt(sum((ai - bi)**2 for ai,bi in zip(a,b)))

    def predict(self, X):
        preds = []
        for xi in X:
            dists = [(self._dist(xi, xj), yj) for xj, yj in zip(self.X, self.y)]
            dists.sort(key=lambda t: t[0])
            top = dists[:self.k]
            votes = sum(1 for _, lab in top if lab==1)
            zeros = self.k - votes
            preds.append(1 if votes >= zeros else 0)
        return preds

    def predict_proba(self, X):
        # prob for class=1 = proportion of neighbors labeled 1
        probs = []
        for xi in X:
            dists = [(self._dist(xi, xj), yj) for xj, yj in zip(self.X, self.y)]
            dists.sort(key=lambda t: t[0])
            top = dists[:self.k]
            votes = sum(1 for _, lab in top if lab==1)
            probs.append(votes / max(1, len(top)))
        return probs

# Logistic Regression (binary, L2, GD)
class LogisticRegressionGD:
    def __init__(self, lr=0.1, epochs=1000, reg=0.0):
        self.lr = lr
        self.epochs = epochs
        self.reg = reg
        self.w = None
        self.b = 0.0

    @staticmethod
    def _sigmoid(z):
        if z>=0:
            ez = math.exp(-z)
            return 1.0/(1.0+ez)
        else:
            ez = math.exp(z)
            return ez/(1.0+ez)

    def fit(self, X, y):
        n = len(X)
        d = len(X[0]) if n else 0
        self.w = [0.0]*d
        self.b = 0.0
        for _ in range(self.epochs):
            grad_w = [0.0]*d
            grad_b = 0.0
            for i in range(n):
                z = sum(self.w[j]*X[i][j] for j in range(d)) + self.b
                p = self._sigmoid(z)
                err = p - y[i]
                for j in range(d):
                    grad_w[j] += err * X[i][j]
                grad_b += err
            # L2 regularization
            for j in range(d):
                grad_w[j] = grad_w[j]/n + self.reg*self.w[j]
            grad_b = grad_b / n
            for j in range(d):
                self.w[j] -= self.lr * grad_w[j]
            self.b -= self.lr * grad_b

    def predict(self, X):
        probs = self.predict_proba(X)
        return [1 if p>=0.5 else 0 for p in probs]

    def predict_proba(self, X):
        probs = []
        for xi in X:
            z = sum(self.w[j]*xi[j] for j in range(len(xi))) + self.b
            probs.append(self._sigmoid(z))
        return probs

# Linear "SVM" via Pegasos (hinge-loss SGD)
class LinearSVM:
    def __init__(self, C=0.1, epochs=10, lr0=0.1):
        self.C = C
        self.epochs = epochs
        self.lr0 = lr0
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        # y in {0,1} -> convert to {-1,+1}
        y2 = [1 if t==1 else -1 for t in y]
        n = len(X)
        d = len(X[0]) if n else 0
        self.w = [0.0]*d
        self.b = 0.0
        t = 0
        for _ in range(self.epochs):
            idxs = list(range(n))
            random.shuffle(idxs)
            for i in idxs:
                t += 1
                lr = self.lr0 / (1 + self.lr0 * t * self.C)
                margin = y2[i] * (sum(self.w[j]*X[i][j] for j in range(d)) + self.b)
                if margin < 1:
                    # gradient step
                    for j in range(d):
                        self.w[j] = (1 - lr) * self.w[j] + lr * self.C * y2[i] * X[i][j]
                    self.b = self.b + lr * self.C * y2[i]
                else:
                    for j in range(d):
                        self.w[j] = (1 - lr) * self.w[j]
        # done

    def decision_function(self, X):
        return [sum(self.w[j]*xi[j] for j in range(len(xi))) + self.b for xi in X]

    def predict(self, X):
        return [1 if z>=0 else 0 for z in self.decision_function(X)]

# Voting
class VotingClassifierScratch:
    def __init__(self, estimators: dict, voting="soft"):
        self.ests = estimators  # dict name->model
        self.voting = voting

    def fit(self, X, y):
        for m in self.ests.values():
            if hasattr(m, "fit"):
                m.fit(X, y)

    def predict(self, X):
        if self.voting == "hard":
            preds = []
            for xi in X:
                votes = []
                for m in self.ests.values():
                    p = m.predict([xi])[0]
                    votes.append(p)
                c = Counter(votes).most_common(1)[0][0]
                preds.append(c)
            return preds
        else:
            # soft: need probabilities/scores
            preds=[]
            for xi in X:
                scores=[]
                for m in self.ests.values():
                    if hasattr(m, "predict_proba"):
                        s = m.predict_proba([xi])[0]
                    elif hasattr(m, "decision_function"):
                        # map decision score to (0..1) via logistic
                        z = m.decision_function([xi])[0]
                        s = 1.0/(1.0+math.exp(-z))
                    else:
                        # fallback to class prediction as score 0/1
                        s = m.predict([xi])[0]
                    scores.append(s)
                avg = sum(scores)/len(scores) if scores else 0.5
                preds.append(1 if avg>=0.5 else 0)
            return preds

    def predict_proba_soft(self, X):
        # for ROC in soft mode
        probs=[]
        for xi in X:
            scores=[]
            for m in self.ests.values():
                if hasattr(m, "predict_proba"):
                    s = m.predict_proba([xi])[0]
                elif hasattr(m, "decision_function"):
                    z = m.decision_function([xi])[0]
                    s = 1.0/(1.0+math.exp(-z))
                else:
                    s = m.predict([xi])[0]
                scores.append(s)
            avg = sum(scores)/len(scores) if scores else 0.5
            probs.append(avg)
        return probs

# ------------------------------------
# Streamlit App (layout mirrors yours)
# ------------------------------------
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

# ===== Sidebar =====
st.sidebar.title("🫀 Heart Disease Comparison")
st.sidebar.markdown("Upload your dataset and explore model performance.📊")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# Load CSV
if uploaded_file:
    rows = read_csv_to_rows(uploaded_file)
else:
    rows = read_csv_to_rows("heart.csv")

st.sidebar.write(f"Dataset shape: ({len(rows)}, {len(rows[0]) if rows else 0})")

# ===== Tabs =====
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
    st.dataframe(rows[:10])

    st.subheader("📈 Dataset Information")
    cols, types = columns_and_types(rows)
    info_buf = []
    info_buf.append(f"Columns: {len(cols)}")
    for c in cols:
        uniques = len(set(r[c] for r in rows))
        missing = sum(1 for r in rows if r[c] is None or r[c]=="" or to_float(r[c]) is None and types[c]=="numeric")
        info_buf.append(f"{c}: type={types[c]}, unique={uniques}, missing~{missing}")
    st.text("\n".join(info_buf))

    st.subheader("🧬 Data Types and Unique Values")
    st.dataframe(
        [{"Column": c,
          "Data Type": types[c],
          "Unique Values": len(set(r[c] for r in rows)),
          "Missing Values": sum(1 for r in rows if r[c] is None or r[c]=="" or (types[c]=="numeric" and to_float(r[c]) is None))}
         for c in cols]
    )

    st.subheader("🔢 Summary Statistics")
    # simple describe for numeric columns
    desc = []
    for c in cols:
        if types[c] == "numeric":
            vals = [to_float(r[c]) for r in rows if to_float(r[c]) is not None]
            if not vals: continue
            n = len(vals)
            mean = sum(vals)/n
            sd = math.sqrt(sum((v-mean)**2 for v in vals)/n)
            mn, mx = min(vals), max(vals)
            desc.append({"Column": c, "Count": n, "Mean": round(mean,3),
                         "Std": round(sd,3), "Min": mn, "Max": mx})
    st.dataframe(desc)

    st.subheader("🧩 Target Distribution (Pie Chart)")
    # emulate pie via bar (no external plotting libs)
    tgt_counts = Counter(int(float(r["target"])) for r in rows if r.get("target") is not None and r["target"]!="")
    st.bar_chart({"count": [tgt_counts.get(0,0), tgt_counts.get(1,0)]}, x=None)

    # ====== DATA CLEANING ======
    st.subheader("Clean the Dataset")
    original_rows = len(rows)
    duplicate_count = original_rows - len(drop_duplicates(rows))
    rows_nodup = drop_duplicates(rows)

    # drop missing for numeric columns
    numeric_cols = [c for c in cols if c!="target" and types[c]=="numeric"]
    rows_nomiss = drop_missing(rows_nodup, numeric_cols + ["target"])
    missing_count = len(rows_nodup) - len(rows_nomiss)

    # outliers on specific columns
    cont_cols = ['age', 'trestbps', 'thalach', 'oldpeak']
    cont_cols = [c for c in cont_cols if c in numeric_cols]
    rows_clean = filter_outliers(rows_nomiss, cont_cols)
    outlier_count = len(rows_nomiss) - len(rows_clean)
    final_rows = len(rows_clean)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Original", original_rows)
    col2.metric("Duplicates", duplicate_count)
    col3.metric("Missing", missing_count)
    col4.metric("Outliers", outlier_count)
    col5.metric("Final Rows", final_rows)

    with st.expander("📄 View Cleaned Data"):
        st.dataframe(rows_clean[:50])

    st.subheader("🔁 Multicollinearity Check (VIF)")
    # simple pairwise correlation magnitude as proxy (no true VIF)
    # Show average absolute correlation for each feature (approx only)
    # (True VIF needs regressions; we avoid external libs.)
    _, X_all, y_all = as_Xy(rows_clean, "target")
    means, stds = standardize_fit(X_all)
    X_scaled = standardize_apply(X_all, means, stds)
    def corr_col(j, k):
        n = len(X_scaled)
        s = sum(X_scaled[i][j]*X_scaled[i][k] for i in range(n)) / (n if n>0 else 1)
        return s
    approx_vif_table = []
    for j, name in enumerate([c for c in cols if c!="target"]):
        s = 0.0
        cnt = 0
        for k in range(len(cols)-1):
            if j==k: continue
            s += abs(corr_col(j,k))
            cnt += 1
        approx = s / cnt if cnt>0 else 0.0
        approx_vif_table.append({"feature": name, "avg |corr| (proxy)": round(approx,3)})
    st.dataframe(approx_vif_table)

    st.subheader("Apply SMOTE and Standardization")
    # From-scratch standardization + simple ROS to mimic class balance
    X_cols, X_all, y_all = as_Xy(rows_clean, "target")
    means, stds = standardize_fit(X_all)
    X_scaled = standardize_apply(X_all, means, stds)
    X_bal, y_bal = oversample_simple(X_scaled, y_all)

    st.subheader("⚖️ Class Distribution Before & After Oversampling")
    colb1, colb2 = st.columns(2)
    with colb1:
        st.markdown("### Before")
        st.bar_chart({"count":[y_all.count(0), y_all.count(1)]})
    with colb2:
        st.markdown("### After")
        st.bar_chart({"count":[y_bal.count(0), y_bal.count(1)]})

with tab2:
    st.header("🧠 KNN Pipeline")

    # Find best K (1..20) via 10-fold CV (stratified)
    # We'll do a quick manual CV on the balanced standardized data
    X_cols, X_all, y_all = as_Xy(filter_outliers(drop_missing(drop_duplicates(rows), [c for c,_t in columns_and_types(rows)[1].items() if _t=="numeric"])+[], ['age','trestbps','thalach','oldpeak']), "target")
    means, stds = standardize_fit(X_all)
    X_scaled = standardize_apply(X_all, means, stds)
    X_bal, y_bal = oversample_simple(X_scaled, y_all)

    def kfold_indices(y, k=10, seed=RANDOM_SEED):
        random.seed(seed)
        idxs = list(range(len(y)))
        by_c = defaultdict(list)
        for i in idxs:
            by_c[y[i]].append(i)
        folds = [[] for _ in range(k)]
        for c, lst in by_c.items():
            random.shuffle(lst)
            for i, idx in enumerate(lst):
                folds[i % k].append(idx)
        return folds

    folds = kfold_indices(y_bal, k=10)
    k_range = list(range(1,21))
    k_scores = []
    for K in k_range:
        accs = []
        for i in range(10):
            test_idx = set(folds[i])
            train_idx = [j for j in range(len(y_bal)) if j not in test_idx]
            Xtr = [X_bal[j] for j in train_idx]
            ytr = [y_bal[j] for j in train_idx]
            Xte = [X_bal[j] for j in test_idx]
            yte = [y_bal[j] for j in test_idx]
            model = KNN(k=K)
            model.fit(Xtr, ytr)
            ypred = model.predict(Xte)
            accs.append(accuracy(yte, ypred))
        k_scores.append(sum(accs)/len(accs) if accs else 0.0)

    best_k = k_range[max(range(len(k_scores)), key=lambda i: k_scores[i])]
    best_score = k_scores[best_k-1]

    st.line_chart({"Accuracy": k_scores})

    st.success(f"🏆 Best K = **{best_k}** with Accuracy = **{best_score:.4f}**")

    # Train/Test split and evaluation
    X_train, y_train, X_test, y_test = stratified_train_test_split(X_scaled, y_all, test_size=0.2)
    X_train_bal, y_train_bal = oversample_simple(X_train, y_train)
    knn_model = KNN(k=best_k)
    knn_model.fit(X_train_bal, y_train_bal)
    y_pred = knn_model.predict(X_test)

    st.subheader("Evaluation Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy(y_test, y_pred):.2f}")
    col2.metric("Precision", f"{precision(y_test, y_pred):.2f}")
    col3.metric("Recall", f"{recall(y_test, y_pred):.2f}")
    col4.metric("F1 Score", f"{f1_score(y_test, y_pred):.2f}")

    with st.expander("📄 Classification Report"):
        p = precision(y_test, y_pred)
        r = recall(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        st.code(f"precision: {p:.4f}\nrecall: {r:.4f}\nf1-score: {f1:.4f}\naccuracy: {accuracy(y_test, y_pred):.4f}")

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix_counts(y_test, y_pred)
    st.subheader("Confusion Matrix - KNN")
    st.table([{"": "", "Pred 0": "", "Pred 1": ""},
              {"True 0": tn, "Pred 0": tn, "Pred 1": fp},
              {"True 1": fn, "Pred 0": fn, "Pred 1": tp}])

with tab3:
    st.header("📈 Logistic Regression Analysis")

    # Correlation matrix proxy not shown as heatmap (no external plotting)
    st.subheader("🔍 Correlation Matrix (proxy)")
    st.info("Showing simple pairwise correlation magnitudes (approx). See Preprocessing → VIF proxy table.")

    st.subheader("📊 Feature Scaling (Before vs After)")
    # Show distributions proxy: mean/std before vs after
    cols, types = columns_and_types(rows)
    num_cols = [c for c in cols if c!="target" and types[c]=="numeric"]
    numeric_vals = {c:[to_float(r[c]) for r in rows if to_float(r[c]) is not None] for c in num_cols}
    before_stats = {c: (sum(v)/len(v), math.sqrt(sum((x - (sum(v)/len(v)))**2 for x in v)/len(v)) if v else 0.0) for c,v in numeric_vals.items()}
    # After scaling on cleaned data:
    rows_clean = filter_outliers(drop_missing(drop_duplicates(rows), num_cols+["target"]), ['age','trestbps','thalach','oldpeak'])
    X_cols, X_all, y_all = as_Xy(rows_clean, "target")
    m, s = standardize_fit(X_all)
    after_stats = {X_cols[j]: (0.0, 1.0) for j in range(len(X_cols))}
    st.write("**Before (mean, std)**")
    st.table([{ "feature": c, "mean": round(before_stats[c][0],3), "std": round(before_stats[c][1],3)} for c in num_cols])
    st.write("**After (mean, std)**")
    st.table([{ "feature": c, "mean": 0.0, "std": 1.0} for c in X_cols])

    st.subheader("")
    # Model training
    X_scaled = standardize_apply(X_all, m, s)
    X_train, y_train, X_test, y_test = stratified_train_test_split(X_scaled, y_all, test_size=0.2)
    X_train_bal, y_train_bal = oversample_simple(X_train, y_train)
    logreg = LogisticRegressionGD(lr=0.1, epochs=800, reg=0.01)
    logreg.fit(X_train_bal, y_train_bal)
    y_pred = logreg.predict(X_test)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy(y_test, y_pred):.2f}")
    col2.metric("Precision", f"{precision(y_test, y_pred):.2f}")
    col3.metric("Recall", f"{recall(y_test, y_pred):.2f}")
    col4.metric("F1 Score", f"{f1_score(y_test, y_pred):.2f}")

    with st.expander("📄 Classification Report"):
        p = precision(y_test, y_pred)
        r = recall(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        st.code(f"precision: {p:.4f}\nrecall: {r:.4f}\nf1-score: {f1:.4f}\naccuracy: {accuracy(y_test, y_pred):.4f}")

    st.header("Logistic Regression Evaluation")
    tn, fp, fn, tp = confusion_matrix_counts(y_test, y_pred)
    st.subheader("Confusion Matrix - Logistic Regression")
    st.table([{"": "", "Pred 0": "", "Pred 1": ""},
              {"True 0": tn, "Pred 0": tn, "Pred 1": fp},
              {"True 1": fn, "Pred 0": fn, "Pred 1": tp}])

with tab4:
    st.header("🧠 Support Vector Machine (SVM) Classification")

    # Clean + split
    rows_clean = filter_outliers(drop_missing(drop_duplicates(rows), [c for c,t in columns_and_types(rows)[1].items() if t=="numeric"]+["target"]), ['age','trestbps','thalach','oldpeak'])
    X_cols, X_all, y_all = as_Xy(rows_clean, "target")
    m, s = standardize_fit(X_all)
    X_scaled = standardize_apply(X_all, m, s)
    X_train, y_train, X_test, y_test = stratified_train_test_split(X_scaled, y_all, test_size=0.2)
    X_train_bal, y_train_bal = oversample_simple(X_train, y_train)

    svm_model = LinearSVM(C=0.1, epochs=15, lr0=0.5)
    svm_model.fit(X_train_bal, y_train_bal)
    y_pred = svm_model.predict(X_test)

    st.subheader("📊 Evaluation Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy(y_test, y_pred):.2f}")
    col2.metric("Precision", f"{precision(y_test, y_pred):.2f}")
    col3.metric("Recall", f"{recall(y_test, y_pred):.2f}")
    col4.metric("F1 Score", f"{f1_score(y_test, y_pred):.2f}")

    st.text("📄 Classification Report")
    p = precision(y_test, y_pred)
    r = recall(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    st.code(f"precision: {p:.4f}\nrecall: {r:.4f}\nf1-score: {f1:.4f}\naccuracy: {accuracy(y_test, y_pred):.4f}")

    st.subheader("📉 Confusion Matrix")
    tn, fp, fn, tp = confusion_matrix_counts(y_test, y_pred)
    st.table([{"": "", "Pred 0": "", "Pred 1": ""},
              {"True 0": tn, "Pred 0": tn, "Pred 1": fp},
              {"True 1": fn, "Pred 0": fn, "Pred 1": tp}])

    st.subheader("🎯 SVM Decision Boundary (PCA Projection)")
    st.info("PCA scatter/contour omitted to avoid external libraries; showing linear decision scores histogram instead.")
    scores = svm_model.decision_function(X_train_bal)
    # simple histogram via bar chart of binned scores
    bins = [-3,-2,-1,-0.5,0,0.5,1,2,3]
    hist = [0]*(len(bins)+1)
    for sc in scores:
        placed=False
        for i,b in enumerate(bins):
            if sc<b:
                hist[i]+=1
                placed=True
                break
        if not placed:
            hist[-1]+=1
    st.bar_chart({"count": hist})

with tab5:
    st.header("📊 Model Comparison (with Combined/Voting Models)")

    rows_clean = filter_outliers(drop_missing(drop_duplicates(rows), [c for c,t in columns_and_types(rows)[1].items() if t=="numeric"]+["target"]), ['age','trestbps','thalach','oldpeak'])
    X_cols, X_all, y_all = as_Xy(rows_clean, "target")
    m, s = standardize_fit(X_all)
    X_scaled = standardize_apply(X_all, m, s)
    X_train, y_train, X_test, y_test = stratified_train_test_split(X_scaled, y_all, test_size=0.2)
    X_train_bal, y_train_bal = oversample_simple(X_train, y_train)

    # Base models
    knn13 = KNN(k=13)
    logreg = LogisticRegressionGD(lr=0.1, epochs=800, reg=0.01)
    svm = LinearSVM(C=0.1, epochs=15, lr0=0.5)

    # Fit
    for m_ in (knn13, logreg, svm):
        m_.fit(X_train_bal, y_train_bal)

    # Combined
    voting_hard = VotingClassifierScratch({"knn": knn13, "lr": logreg, "svm": svm}, voting="hard")
    voting_soft = VotingClassifierScratch({"knn": knn13, "lr": logreg, "svm": svm}, voting="soft")

    models = {
        "KNN (k=13)": knn13,
        "Logistic Regression": logreg,
        "SVM (C=0.1)": svm,
        "Combined (Hard Vote)": voting_hard,
        "Combined (Soft Vote)": voting_soft
    }

    # Evaluate
    rows_table = []
    trained = {}
    for name, model in models.items():
        if isinstance(model, VotingClassifierScratch):
            model.fit(X_train_bal, y_train_bal)
        trained[name] = model
        y_pred = model.predict(X_test)
        rows_table.append({
            "Model": name,
            "Accuracy": round(accuracy(y_test, y_pred), 3),
            "Precision": round(precision(y_test, y_pred), 3),
            "Recall": round(recall(y_test, y_pred), 3),
            "F1 Score": round(f1_score(y_test, y_pred), 3)
        })

    st.dataframe(rows_table)

    st.subheader("🔍 Metric Comparison")
    # build melted-like tall format for bar chart
    # We'll show F1 only to keep chart readable without seaborn
    model_names = [r["Model"] for r in rows_table]
    f1_vals = [r["F1 Score"] for r in rows_table]
    st.bar_chart({"F1 Score": f1_vals})

    st.subheader("📈 ROC Curves")
    # Plot TPR over FPR as a line chart per model (build series)
    roc_series = {}
    for name, model in trained.items():
        # get scores
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test)  # KNN & LR use list[float]
            if isinstance(scores[0], (list, tuple)):
                scores = [s[1] for s in scores]
        elif isinstance(model, VotingClassifierScratch) and model.voting=="soft":
            scores = model.predict_proba_soft(X_test)
        elif hasattr(model, "decision_function"):
            raw = model.decision_function(X_test)
            scores = [1.0/(1.0+math.exp(-z)) for z in raw]
        else:
            # fallback to 0/1
            scores = model.predict(X_test)

        pts = roc_curve_points(y_test, scores)
        roc_series[name] = pts
    # Show AUCs
    auc_rows = []
    for name, pts in roc_series.items():
        auc_rows.append({"Model": name, "AUC": round(auc_trapezoid(pts), 3)})
    st.table(auc_rows)

    st.subheader("🧮 Confusion Matrices")
    # Show last model's CM counts inline & loop others as text
    for name, model in trained.items():
        y_pred = model.predict(X_test)
        tn, fp, fn, tp = confusion_matrix_counts(y_test, y_pred)
        st.markdown(f"**{name}**")
        st.table([{"": "", "Pred 0": "", "Pred 1": ""},
                  {"True 0": tn, "Pred 0": tn, "Pred 1": fp},
                  {"True 1": fn, "Pred 0": fn, "Pred 1": tp}])

    st.subheader("🥇 Best Performing Model")
    best = max(rows_table, key=lambda r: r["F1 Score"])
    st.success(f"**{best['Model']}** performed best with F1 Score: **{best['F1 Score']:.3f}**")

    st.subheader("⚙️ Model Efficiency")
    # proxy sizes/timings without joblib
    sizes = []
    times = []
    for name, model in trained.items():
        start = time.time()
        _ = model.predict(X_test)
        inf_ms = (time.time() - start)*1000.0
        # "size" proxy = number of parameters or stored samples
        if isinstance(model, KNN):
            size_kb = len(model.X) * (len(model.X[0]) if model.X else 0) * 8 / 1024.0
        elif isinstance(model, LogisticRegressionGD):
            size_kb = (len(model.w)+1) * 8 / 1024.0
        elif isinstance(model, LinearSVM):
            size_kb = (len(model.w)+1) * 8 / 1024.0
        else:
            size_kb = 1.0
        sizes.append(size_kb)
        times.append(inf_ms)

    meta_rows = []
    for (name,_), sz, tms in zip(trained.items(), sizes, times):
        meta_rows.append({"Model": name, "Model Size (KB)": round(sz,2), "Inference Time (ms)": round(tms,2)})
    st.dataframe(meta_rows)

    st.subheader("⬇️Download")
    st.info("CSV download of metrics omitted (no external encoder); copy from the table above if needed.")
