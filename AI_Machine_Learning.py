import streamlit as st

# ================================
# Pure-Python helpers (no imports)
# ================================
def _abs(x): return x if x >= 0 else -x
def _exp(x):
    e = 2.718281828459045
    return e ** x
def _sqrt(x):
    if x <= 0: return 0.0
    g = x
    for _ in range(30):
        g = 0.5 * (g + x / g)
    return g
def _sigmoid(z):
    if z >= 0:
        t = _exp(-z)
        return 1.0 / (1.0 + t)
    else:
        t = _exp(z)
        return t / (1.0 + t)
def dot(a, b):
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s
def format_float(x, n=3):
    s = str(round(x + 10**(-n-2), n))
    if "." not in s:
        s += "." + ("0"*n)
    else:
        after = len(s) - s.index(".") - 1
        if after < n:
            s += "0" * (n - after)
    return s

# -----------------------
# CSV parsing (file/str)
# -----------------------
def parse_csv_text_to_numeric(text):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() != ""]
    if not lines:
        return [], []
    header = [h.strip() for h in lines[0].split(",")]
    rows = []
    for ln in lines[1:]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != len(header): 
            continue
        if any((p == "" or p.lower() == "nan") for p in parts):
            continue
        rec = []
        bad = False
        for p in parts:
            # naive float-check
            s = p
            if s.startswith("-"): s = s[1:]
            if s.count(".") > 1 or any((ch != "." and (ch < "0" or ch > "9")) for ch in s):
                bad = True; break
            try:
                val = float(p)
            except:
                bad = True; break
            rec.append(val)
        if not bad:
            rows.append(rec)
    return header, rows

def get_column_index(header, name):
    for i, h in enumerate(header):
        if h == name:
            return i
    raise ValueError("Column not found: " + name)

def remove_duplicates(rows):
    seen = set(); out = []
    for r in rows:
        t = tuple(r)
        if t not in seen:
            seen.add(t); out.append(r)
    return out

def percentile(sorted_vals, q):
    n = len(sorted_vals)
    if n == 0: return 0.0
    idx = q * (n - 1)
    lo = int(idx); hi = lo + 1
    if hi >= n: return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo]*(1-frac) + sorted_vals[hi]*frac

def iqr_mask(rows, col_idx):
    vals = [r[col_idx] for r in rows]
    sv = sorted(vals)
    q1 = percentile(sv, 0.25)
    q3 = percentile(sv, 0.75)
    IQR = q3 - q1
    low = q1 - 1.5 * IQR
    high = q3 + 1.5 * IQR
    return [(v < low or v > high) for v in vals]

def filter_rows(rows, mask):
    return [r for i, r in enumerate(rows) if not mask[i]]

def col_means_stds(X):
    if not X: return [], []
    R, C = len(X), len(X[0])
    means = [0.0]*C
    for j in range(C):
        s = 0.0
        for i in range(R): s += X[i][j]
        means[j] = s / R
    stds = [1.0]*C
    for j in range(C):
        s2 = 0.0; m = means[j]
        for i in range(R):
            d = X[i][j] - m
            s2 += d*d
        v = s2 / R
        stds[j] = _sqrt(v) if v > 0 else 1.0
    return means, stds

def standardize(X, means, stds):
    R = len(X); C = len(X[0]) if R else 0
    Z = [[0.0]*C for _ in range(R)]
    for i in range(R):
        for j in range(C):
            Z[i][j] = (X[i][j] - means[j]) / stds[j]
    return Z

def train_test_split_ordered(X, y, test_ratio=0.2):
    n = len(X)
    t = int(n * test_ratio)
    return X[:n-t], X[n-t:], y[:n-t], y[n-t:]

# -------------
# Metrics
# -------------
def confusion_matrix_binary(y_true, y_pred):
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1: tp += 1
        elif t == 0 and p == 1: fp += 1
        elif t == 0 and p == 0: tn += 1
        elif t == 1 and p == 0: fn += 1
    return tn, fp, fn, tp

def accuracy(y_true, y_pred):
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / (len(y_true) or 1)

def precision_recall_f1(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix_binary(y_true, y_pred)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = (2*prec*rec)/(prec+rec) if (prec+rec) else 0.0
    return prec, rec, f1, (tn, fp, fn, tp)

# -------------
# KNN (scratch)
# -------------
def _dist2(a, b):
    s = 0.0
    for i in range(len(a)):
        d = a[i] - b[i]
        s += d*d
    return s

def knn_predict_row(Xtr, ytr, x, k):
    best = []
    for i, r in enumerate(Xtr):
        d2 = _dist2(r, x)
        inserted = False
        for j in range(len(best)):
            if d2 < best[j][0]:
                best.insert(j, (d2, ytr[i])); inserted = True; break
        if not inserted: best.append((d2, ytr[i]))
        if len(best) > k: best.pop()
    v1 = sum(1 for _, lab in best[:k] if lab == 1)
    v0 = k - v1
    return 1 if v1 >= v0 else 0

def knn_predict(Xtr, ytr, X, k):
    return [knn_predict_row(Xtr, ytr, x, k) for x in X]

# ----------------------------
# Logistic Regression (from 0)
# ----------------------------
def logistic_train(X, y, lr=0.05, iters=1200, l2=0.0):
    R = len(X); C = len(X[0]) if R else 0
    w = [0.0]*C; b = 0.0
    for _ in range(iters):
        gw = [0.0]*C; gb = 0.0
        for i in range(R):
            z = dot(w, X[i]) + b
            p = _sigmoid(z)
            err = p - y[i]
            for j in range(C):
                gw[j] += err * X[i][j]
            gb += err
        for j in range(C):
            gw[j] = gw[j]/R + l2*w[j]
            w[j] -= lr * gw[j]
        gb = gb / R
        b -= lr * gb
    return w, b

def logistic_predict(X, w, b, thr=0.5):
    out = []
    for i in range(len(X)):
        p = _sigmoid(dot(w, X[i]) + b)
        out.append(1 if p >= thr else 0)
    return out

# ------------------------
# Toy oversampler (dup)
# ------------------------
def toy_oversample(X, y):
    n1 = sum(1 for t in y if t == 1)
    n0 = len(y) - n1
    if n1 == n0: return X[:], y[:]
    Xn, yn = X[:], y[:]
    if n1 < n0:
        need = n0 - n1
        idxs = [i for i, t in enumerate(y) if t == 1]
        k = 0
        while need > 0 and idxs:
            Xn.append(X[idxs[k % len(idxs)]][:]); yn.append(1)
            k += 1; need -= 1
    else:
        need = n1 - n0
        idxs = [i for i, t in enumerate(y) if t == 0]
        k = 0
        while need > 0 and idxs:
            Xn.append(X[idxs[k % len(idxs)]][:]); yn.append(0)
            k += 1; need -= 1
    return Xn, yn

# ===============
# Streamlit App
# ===============
st.set_page_config(page_title="Heart Disease — Pure Python Models", layout="wide")
st.title("🫀 Heart Disease Predictor (No external ML libs)")
st.caption("Only Streamlit is imported; KNN and Logistic Regression are implemented from scratch.")

with st.sidebar:
    st.header("Data")
    up = st.file_uploader("Upload heart.csv (or leave empty to use local file)", type=["csv"])
    target_name = st.text_input("Target column name", value="target")
    cont_cols = st.text_input("Continuous columns for IQR outlier removal (comma-separated)", 
                              value="age,trestbps,thalach,oldpeak")
    test_ratio = st.slider("Test ratio", 0.1, 0.5, 0.2, 0.05)

    st.header("Models")
    k_val = st.number_input("K for KNN (odd recommended)", min_value=1, value=13, step=2)
    lr = st.number_input("LR learning rate", min_value=0.001, value=0.07, step=0.001, format="%.3f")
    iters = st.number_input("LR iterations", min_value=200, value=1400, step=100)
    l2 = st.number_input("LR L2 (ridge)", min_value=0.0, value=0.0005, step=0.0005, format="%.4f")
    thr = st.slider("LR decision threshold", 0.1, 0.9, 0.5, 0.05)

# Load CSV
csv_text = ""
if up is not None:
    csv_text = up.read().decode("utf-8", errors="ignore")
else:
    # try local file
    try:
        with open("heart.csv", "r", encoding="utf-8") as f:
            csv_text = f.read()
    except:
        pass

if not csv_text:
    st.warning("Please upload `heart.csv` or place one next to this script.")
    st.stop()

# Parse & clean
header, rows = parse_csv_text_to_numeric(csv_text)
if not header or not rows:
    st.error("Failed to parse CSV or CSV is empty.")
    st.stop()

try:
    tgt_idx = get_column_index(header, target_name)
except Exception as e:
    st.error(str(e)); st.stop()

# remove duplicates (missing already skipped in parser)
original = len(rows)
dup_removed = remove_duplicates(rows)
duplicates_count = original - len(dup_removed)
rows = dup_removed

# IQR outlier removal on selected columns
cont_names = [c.strip() for c in cont_cols.split(",") if c.strip()]
valid_idx = []
for name in cont_names:
    try:
        valid_idx.append(get_column_index(header, name))
    except:
        pass

if rows and valid_idx:
    combined = [False]*len(rows)
    for ci in valid_idx:
        m = iqr_mask(rows, ci)
        combined = [ (a or b) for a, b in zip(combined, m) ]
    outlier_count = sum(1 for x in combined if x)
    rows = filter_rows(rows, combined)
else:
    outlier_count = 0

final_rows = len(rows)

# Split X/y
X, y = [], []
for r in rows:
    feats = [r[j] for j in range(len(r)) if j != tgt_idx]
    X.append(feats)
    y.append(int(r[tgt_idx]))

if not X:
    st.error("No data after cleaning."); st.stop()

# Standardize
means, stds = col_means_stds(X)
Xz = standardize(X, means, stds)

# Train/test
Xtr, Xte, ytr, yte = train_test_split_ordered(Xz, y, test_ratio=test_ratio)

# Balance
Xb, yb = toy_oversample(Xtr, ytr)

# Auto-fix K if too large
if k_val > len(Xb):
    k_val = max(1, (len(Xb)//2)*2 + 1)

# Train models
w, b = logistic_train(Xb, yb, lr=lr, iters=iters, l2=l2)

# Predict
yhat_knn = knn_predict(Xb, yb, Xte, k_val)
yhat_lr  = logistic_predict(Xte, w, b, thr=thr)
yhat_ens = [1 if (a + c) >= 1 else 0 for a, c in zip(yhat_knn, yhat_lr)]

# Evaluate
def eval_block(name, yhat):
    prec, rec, f1, cm = precision_recall_f1(yte, yhat)
    acc = accuracy(yte, yhat)
    tn, fp, fn, tp = cm
    st.subheader(name)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", format_float(acc))
    c2.metric("Precision", format_float(prec))
    c3.metric("Recall", format_float(rec))
    c4.metric("F1", format_float(f1))

    st.markdown("**Confusion Matrix (TN FP / FN TP):**")
    st.table([
        ["TN", tn, "FP", fp],
        ["FN", fn, "TP", tp],
    ])
    return {"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}

st.header("🧹 Preprocessing Summary")
m1, m2, m3 = st.columns(3)
m1.metric("Original Rows", original)
m2.metric("Duplicates Removed", duplicates_count)
m3.metric("Outliers Removed", outlier_count)
st.caption(f"Final rows used: **{final_rows}**. Features: **{len(header)-1}** (excluding target).")

st.header("📊 Results")
res = []
res.append(eval_block(f"KNN (k={k_val})", yhat_knn))
res.append(eval_block("Logistic Regression (GD)", yhat_lr))
res.append(eval_block("Combined Hard Vote", yhat_ens))

# Results table + download
st.subheader("📥 Download metrics")
csv_lines = ["Model,Accuracy,Precision,Recall,F1"]
for r in res:
    csv_lines.append(
        f'{r["Model"]},{format_float(r["Accuracy"])},{format_float(r["Precision"])},{format_float(r["Recall"])},{format_float(r["F1"])}'
    )
csv_blob = "\n".join(csv_lines).encode("utf-8")
st.download_button("Download model_comparison.csv", data=csv_blob, file_name="model_comparison.csv", mime="text/csv")

with st.expander("🔎 Data preview (first 5 rows after cleaning)"):
    # Show as simple table (no pandas)
    head_rows = rows[:5]
    st.table([header] + head_rows)

st.caption("Note: No NumPy/Pandas/Scikit used — just Streamlit + pure Python implementations.")
