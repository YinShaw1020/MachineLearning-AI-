# app.py
# Streamlit-only app: no external libs (numpy/pandas/sklearn/matplotlib).
# Implements: data loading, cleaning, scaling, stratified split, simple SMOTE-like balancing,
# KNN, Logistic Regression (GD), Linear SVM-like classifier (hinge-loss SGD),
# metrics, confusion matrix, ROC, and basic SVG charts — all from scratch.

import streamlit as st
import csv, math, random, statistics, time, io

# --------------------------- Utilities (no 3rd-party) ---------------------------

def read_csv_as_rows(file_like):
    text = file_like.read().decode("utf-8") if hasattr(file_like, "read") else open(file_like, "r", encoding="utf-8").read()
    f = io.StringIO(text)
    rdr = csv.DictReader(f)
    rows = [dict(r) for r in rdr]
    return rows

def coerce_numeric(rows):
    # Try cast to float for every field except those clearly non-numeric
    for r in rows:
        for k, v in list(r.items()):
            if v is None or v == "":
                r[k] = None
                continue
            try:
                r[k] = float(v)
            except:
                # keep as string
                pass
    return rows

def dropna_rows(rows):
    return [r for r in rows if all(v is not None for v in r.values())]

def drop_duplicates(rows):
    seen = set()
    out = []
    for r in rows:
        key = tuple(sorted(r.items()))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out

def column_stats(rows, cols):
    stats = {}
    for c in cols:
        vals = [r[c] for r in rows if isinstance(r[c], (int, float))]
        if len(vals) == 0:
            stats[c] = {"count":0, "mean":None, "std":None, "min":None, "max":None}
        else:
            mean = sum(vals)/len(vals)
            var = sum((x-mean)**2 for x in vals) / max(1, (len(vals)-1))
            sd = math.sqrt(var)
            stats[c] = {"count":len(vals), "mean":mean, "std":sd, "min":min(vals), "max":max(vals)}
    return stats

def iqr_bounds(values):
    s = sorted(values)
    if not s:
        return None, None
    def percentile(p):
        k = p*(len(s)-1)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c: return s[int(k)]
        return s[f] + (s[c]-s[f])*(k-f)
    Q1 = percentile(0.25)
    Q3 = percentile(0.75)
    IQR = Q3 - Q1
    return Q1 - 1.5*IQR, Q3 + 1.5*IQR

def remove_outliers_iqr(rows, cont_cols):
    keep = []
    bounds = {}
    for c in cont_cols:
        vals = [r[c] for r in rows if isinstance(r[c], (int, float))]
        lo, hi = iqr_bounds(vals)
        bounds[c] = (lo, hi)
    for r in rows:
        bad = False
        for c in cont_cols:
            v = r.get(c, None)
            if isinstance(v, (int, float)):
                lo, hi = bounds[c]
                if lo is not None and (v < lo or v > hi):
                    bad = True; break
        if not bad:
            keep.append(r)
    removed = len(rows) - len(keep)
    return keep, removed

def split_Xy(rows, target="target"):
    X_cols = [c for c in rows[0].keys() if c != target]
    X = [[r[c] if isinstance(r[c], (int,float)) else 0.0 for c in X_cols] for r in rows]
    y = [int(r[target]) for r in rows]
    return X, y, X_cols

def zscore_fit(X):
    # returns (means, stds) per feature
    n = len(X); d = len(X[0]) if n>0 else 0
    means = []
    stds = []
    for j in range(d):
        col = [X[i][j] for i in range(n)]
        m = sum(col)/n
        v = sum((x-m)**2 for x in col)/max(1,n-1)
        s = math.sqrt(v) if v>0 else 1.0
        means.append(m); stds.append(s)
    return means, stds

def zscore_transform(X, means, stds):
    Xs = []
    for row in X:
        Xs.append([(row[j]-means[j])/stds[j] for j in range(len(row))])
    return Xs

def stratified_train_test_split(X, y, test_size=0.2, seed=42):
    random.seed(seed)
    idx0 = [i for i,t in enumerate(y) if t==0]
    idx1 = [i for i,t in enumerate(y) if t==1]
    random.shuffle(idx0); random.shuffle(idx1)
    n0 = len(idx0); n1 = len(idx1)
    t0 = int(round(test_size*n0)); t1 = int(round(test_size*n1))
    test_idx = set(idx0[:t0] + idx1[:t1])
    X_train, y_train, X_test, y_test = [],[],[],[]
    for i,(xi,yi) in enumerate(zip(X,y)):
        if i in test_idx: X_test.append(xi); y_test.append(yi)
        else: X_train.append(xi); y_train.append(yi)
    return X_train, X_test, y_train, y_test

def simple_balance_duplicate(X, y, seed=42):
    # naive oversampling to balance classes by random duplication
    random.seed(seed)
    idx0 = [i for i,t in enumerate(y) if t==0]
    idx1 = [i for i,t in enumerate(y) if t==1]
    if len(idx0)==0 or len(idx1)==0:
        return X[:], y[:]
    if len(idx0)==len(idx1):
        return X[:], y[:]
    maj = idx0 if len(idx0)>len(idx1) else idx1
    minr = idx1 if len(idx1)<len(idx0) else idx0
    need = len(maj) - len(minr)
    Xb = X[:]; yb = y[:]
    for _ in range(need):
        j = random.choice(minr)
        Xb.append(X[j][:]); yb.append(y[j])
    return Xb, yb

def dot(a,b): return sum(x*y for x,y in zip(a,b))
def add(a,b): return [x+y for x,y in zip(a,b)]
def sub(a,b): return [x-y for x,y in zip(a,b)]
def mul_scalar(a,k): return [x*k for x in a]
def l2_norm(a): return math.sqrt(sum(x*x for x in a))

# --------------------------- Models (from scratch) ---------------------------

class KNN:
    def __init__(self, k=5):
        self.k = k
        self.X = None
        self.y = None
    def fit(self, X, y):
        self.X = X; self.y = y
    def predict_one(self, x):
        dists = []
        for i,xi in enumerate(self.X):
            # Euclidean distance
            d = math.sqrt(sum((a-b)**2 for a,b in zip(x,xi)))
            dists.append((d, self.y[i]))
        dists.sort(key=lambda t:t[0])
        top = dists[:self.k]
        vote = sum(1 if cls==1 else 0 for _,cls in top)
        # tie-break to 1 if equal? choose >= (k/2)
        return 1 if vote*2 >= self.k else 0
    def predict(self, X):
        return [self.predict_one(x) for x in X]
    def predict_proba(self, X):
        # fraction of 1s in neighbors
        probs = []
        for x in X:
            dists = []
            for i,xi in enumerate(self.X):
                d = math.sqrt(sum((a-b)**2 for a,b in zip(x,xi)))
                dists.append((d, self.y[i]))
            dists.sort(key=lambda t:t[0])
            top = dists[:self.k]
            p1 = sum(1 if cls==1 else 0 for _,cls in top)/self.k
            probs.append([1-p1, p1])
        return probs

class LogisticRegressionGD:
    def __init__(self, lr=0.05, epochs=400, l2=0.001):
        self.lr=lr; self.epochs=epochs; self.l2=l2
        self.w=None; self.b=0.0
    def _sigmoid(self, z): return 1.0/(1.0+math.exp(-z))
    def fit(self, X, y):
        d = len(X[0])
        self.w = [0.0]*d; self.b = 0.0
        for _ in range(self.epochs):
            grad_w = [0.0]*d; grad_b=0.0
            for xi,yi in zip(X,y):
                z = dot(self.w, xi) + self.b
                p = self._sigmoid(z)
                err = p - yi
                for j in range(d):
                    grad_w[j] += err*xi[j]
                grad_b += err
            # add L2
            for j in range(d): grad_w[j] += self.l2*self.w[j]
            # step
            for j in range(d): self.w[j] -= self.lr*grad_w[j]/len(X)
            self.b -= self.lr*grad_b/len(X)
    def predict_proba(self, X):
        probs=[]
        for xi in X:
            p = self._sigmoid(dot(self.w, xi)+self.b)
            probs.append([1-p, p])
        return probs
    def predict(self, X):
        return [1 if p[1]>=0.5 else 0 for p in self.predict_proba(X)]

class LinearSVM_SGD:
    # Simple hinge-loss SGD with L2 regularization
    def __init__(self, lr=0.05, epochs=400, C=1.0):
        self.lr=lr; self.epochs=epochs; self.C=C
        self.w=None; self.b=0.0
    def fit(self, X, y):
        d = len(X[0])
        self.w = [0.0]*d; self.b = 0.0
        # convert y in { -1, +1 }
        yy = [1 if t==1 else -1 for t in y]
        n = len(X)
        for _ in range(self.epochs):
            # SGD: shuffle
            order = list(range(n))
            random.shuffle(order)
            for i in order:
                xi = X[i]; yi = yy[i]
                margin = yi*(dot(self.w, xi) + self.b)
                # gradient of (1/2)||w||^2 + C*max(0,1-margin)
                # If margin >=1: grad_w = w, grad_b = 0
                # Else: grad_w = w - C*yi*xi, grad_b = -C*yi
                if margin >= 1:
                    grad_w = self.w[:]
                    grad_b = 0.0
                else:
                    grad_w = [self.w[j] - self.C*yi*xi[j] for j in range(d)]
                    grad_b = -self.C*yi
                # step
                for j in range(d):
                    self.w[j] -= self.lr*grad_w[j]
                self.b -= self.lr*grad_b
    def decision_function(self, X):
        return [dot(self.w, xi)+self.b for xi in X]
    def predict(self, X):
        return [1 if s>=0 else 0 for s in self.decision_function(X)]
    def predict_proba(self, X):
        # Not calibrated; squash with logistic for visualization
        scores = self.decision_function(X)
        probs=[]
        for s in scores:
            p = 1/(1+math.exp(-s))
            probs.append([1-p, p])
        return probs

# --------------------------- Metrics & Curves ---------------------------

def confusion_matrix(y_true, y_pred):
    tp=fp=tn=fn=0
    for yt, yp in zip(y_true, y_pred):
        if yt==1 and yp==1: tp+=1
        elif yt==0 and yp==1: fp+=1
        elif yt==0 and yp==0: tn+=1
        else: fn+=1
    return tn, fp, fn, tp

def precision_recall_f1(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    acc  = (tp+tn)/max(1,len(y_true))
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    return acc, prec, rec, f1, (tn,fp,fn,tp)

def roc_curve_points(y_true, y_scores):
    # Threshold sweep over sorted unique scores
    pairs = sorted([(s, yt) for s,yt in zip(y_scores, y_true)], key=lambda t:t[0])
    uniq = sorted(set(y_scores))
    if not uniq: return [0.0],[0.0],[]
    fprs=[]; tprs=[]; ths=[]
    P = sum(1 for v in y_true if v==1)
    N = sum(1 for v in y_true if v==0)
    for thr in uniq:
        tp = sum(1 for s,y in pairs if s>=thr and y==1)
        fp = sum(1 for s,y in pairs if s>=thr and y==0)
        fn = P - tp
        tn = N - fp
        fpr = fp/max(1,N); tpr = tp/max(1,P)
        fprs.append(fpr); tprs.append(tpr); ths.append(thr)
    # prepend (0,0) and append (1,1) for nicer shape
    if (0.0,0.0) not in zip(fprs,tprs):
        fprs=[0.0]+fprs; tprs=[0.0]+tprs
    if (1.0,1.0) not in zip(fprs,tprs):
        fprs=fprs+[1.0]; tprs=tprs+[1.0]
    return fprs, tprs, ths

def auc_trapezoid(xs, ys):
    # assumes xs sorted increasing
    area=0.0
    for i in range(1,len(xs)):
        area += (xs[i]-xs[i-1])*(ys[i]+ys[i-1])/2
    return area

# --------------------------- Small SVG helpers ---------------------------

def svg_bar_chart(data_pairs, title="", width=700, height=300, max_val=1.0):
    # data_pairs: [(label, value), ...]
    pad=40
    bar_w = (width-2*pad)/max(1,len(data_pairs))
    svg = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
    svg.append(f'<text x="{width/2}" y="20" font-size="16" text-anchor="middle">{title}</text>')
    # axes
    svg.append(f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="black"/>')
    svg.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="black"/>')
    # bars
    for i,(lab,val) in enumerate(data_pairs):
        h = (height-2*pad) * (max(0.0, min(val, max_val)) / max_val)
        x = pad + i*bar_w + 5
        y = height - pad - h
        svg.append(f'<rect x="{x}" y="{y}" width="{bar_w-10}" height="{h}" fill="steelblue" />')
        svg.append(f'<text x="{x+bar_w/2-5}" y="{height-pad+15}" font-size="10" text-anchor="middle" transform="rotate(15 {x+bar_w/2-5},{height-pad+15})">{lab}</text>')
        svg.append(f'<text x="{x+bar_w/2-5}" y="{y-5}" font-size="10" text-anchor="middle">{val:.3f}</text>')
    svg.append('</svg>')
    return "\n".join(svg)

def svg_confusion_matrix(tn, fp, fn, tp, title="Confusion Matrix", width=360, height=260):
    pad=30; cell=100
    svg=[f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
    svg.append(f'<text x="{width/2}" y="20" font-size="16" text-anchor="middle">{title}</text>')
    # grid
    x0=pad; y0=50
    labels=[("TN",tn),("FP",fp),("FN",fn),("TP",tp)]
    for i in range(2):
        for j in range(2):
            idx = i*2+j
            x = x0 + j*cell
            y = y0 + i*cell
            svg.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="#e8eef9" stroke="#1f4aa8"/>')
            svg.append(f'<text x="{x+cell/2}" y="{y+40}" font-size="14" text-anchor="middle">{labels[idx][0]}</text>')
            svg.append(f'<text x="{x+cell/2}" y="{y+70}" font-size="14" text-anchor="middle">{labels[idx][1]}</text>')
    svg.append(f'<text x="{x0+cell}" y="{y0+2*cell+30}" font-size="12" text-anchor="middle">Predicted 0    1</text>')
    svg.append(f'<text x="{x0-10}" y="{y0+cell}" font-size="12" text-anchor="end" transform="rotate(-90 {x0-10},{y0+cell})">Actual 0    1</text>')
    svg.append('</svg>')
    return "\n".join(svg)

def svg_pie_chart(values, labels, title="", width=360, height=260):
    total = sum(values) if values else 1
    cx, cy, r = width//2, height//2+10, min(width,height)//3
    colors = ["#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f","#edc949"]
    svg=[f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
    svg.append(f'<text x="{width/2}" y="20" font-size="16" text-anchor="middle">{title}</text>')
    angle=0.0
    for i,v in enumerate(values):
        frac = v/total
        sweep = frac*2*math.pi
        x1 = cx + r*math.cos(angle)
        y1 = cy + r*math.sin(angle)
        angle2 = angle + sweep
        x2 = cx + r*math.cos(angle2)
        y2 = cy + r*math.sin(angle2)
        large = 1 if sweep>math.pi else 0
        path = f"M {cx},{cy} L {x1},{y1} A {r},{r} 0 {large},1 {x2},{y2} z"
        svg.append(f'<path d="{path}" fill="{colors[i%len(colors)]}"></path>')
        # label
        mid = angle + sweep/2
        lx = cx + (r+18)*math.cos(mid)
        ly = cy + (r+18)*math.sin(mid)
        svg.append(f'<text x="{lx}" y="{ly}" font-size="12" text-anchor="middle">{labels[i]} ({v})</text>')
        angle = angle2
    svg.append('</svg>')
    return "\n".join(svg)

def svg_roc(fprs, tprs, auc_val, width=360, height=260, title="ROC Curve"):
    pad=35
    svg=[f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
    svg.append(f'<text x="{width/2}" y="18" font-size="16" text-anchor="middle">{title} (AUC={auc_val:.2f})</text>')
    # axes
    x0=pad; y0=height-pad; x1=width-pad; y1=pad
    svg.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="black"/>')
    svg.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="black"/>')
    # diagonal
    svg.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="#999" stroke-dasharray="4,4"/>')
    # polyline
    pts=[]
    for f,t in zip(fprs,tprs):
        xp = x0 + (x1-x0)*f
        yp = y0 - (y0-y1)*t
        pts.append(f"{xp},{yp}")
    svg.append(f'<polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{" ".join(pts)}"/>')
    svg.append(f'<text x="{(x0+x1)/2}" y="{y0+25}" font-size="12" text-anchor="middle">False Positive Rate</text>')
    svg.append(f'<text x="{x0-25}" y="{(y0+y1)/2}" font-size="12" text-anchor="middle" transform="rotate(-90 {x0-25},{(y0+y1)/2})">True Positive Rate</text>')
    svg.append('</svg>')
    return "\n".join(svg)

# --------------------------- App ---------------------------

st.set_page_config(page_title="Heart Disease Predictor (No External Libs)", layout="wide")
st.sidebar.title("🫀 Heart Disease Comparison (No-Dependency Edition)")
st.sidebar.markdown("Upload CSV (with a 'target' column). All processing & ML are from scratch.")

uploaded = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
try:
    rows = read_csv_as_rows(uploaded) if uploaded else read_csv_as_rows("heart.csv")
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

rows = coerce_numeric(rows)
orig_shape = (len(rows), len(rows[0]) if rows else 0)

st.sidebar.write(f"Dataset shape: {orig_shape}")

tabs = st.tabs(["🔍 Preprocessing", "🪭 KNN", "👙 Logistic Regression", "👟 SVM", "📊 Model Comparison"])

# ---------- Tab 1: Preprocessing ----------
with tabs[0]:
    st.header("🔍 Preprocessing: Missing Values, Outlier, Overfitting")

    # Preview (first 5)
    st.subheader("📊 Dataset Preview (first 5 rows)")
    if rows:
        header = list(rows[0].keys())
        preview = [rows[i] for i in range(min(5, len(rows)))]
        st.table(preview)
    else:
        st.warning("Empty dataset.")

    # Info-like
    st.subheader("📈 Dataset Information")
    dtype_info = {}
    for k in (rows[0].keys() if rows else []):
        kinds = set(type(r[k]).__name__ for r in rows)
        dtype_info[k] = ", ".join(sorted(kinds))
    info_text = io.StringIO()
    info_text.write("Columns and detected Python types:\n")
    for k, t in dtype_info.items():
        info_text.write(f"- {k}: {t}\n")
    st.text(info_text.getvalue())

    # Data Types, Unique, Missing
    st.subheader("🧬 Data Types and Unique/Missing")
    stats_rows = []
    for k in (rows[0].keys() if rows else []):
        vals = [r[k] for r in rows]
        uniq = len(set(vals))
        miss = sum(1 for v in vals if v is None)
        kinds = ", ".join(sorted(set(type(v).__name__ for v in vals)))
        stats_rows.append({"Column":k, "Types":kinds, "Unique":uniq, "Missing":miss})
    st.table(stats_rows)

    # Summary statistics
    st.subheader("🔢 Summary Statistics (numeric)")
    num_cols = [k for k in (rows[0].keys() if rows else []) if all((isinstance(r[k], (int,float)) or r[k] is None) for r in rows) and k!="target"]
    ss = column_stats(rows, num_cols)
    sumtab = [{"Feature":k, **ss[k]} for k in num_cols]
    st.table(sumtab)

    # Target pie
    st.subheader("🧩 Target Distribution (Pie Chart)")
    target_vals = [int(r["target"]) for r in rows if r.get("target") is not None]
    zeros = sum(1 for v in target_vals if v==0)
    ones  = sum(1 for v in target_vals if v==1)
    pie_svg = svg_pie_chart([zeros, ones], ["No Heart Disease","Heart Disease"], title="Target")
    st.markdown(pie_svg, unsafe_allow_html=True)

    # Cleaning
    st.subheader("Clean the Dataset")
    original_rows = len(rows)
    rows_nodup = drop_duplicates(rows)
    duplicate_count = original_rows - len(rows_nodup)

    rows_nomiss = dropna_rows(rows_nodup)
    missing_count = len(rows_nodup) - len(rows_nomiss)

    cont_cols_for_outliers = [c for c in ["age","trestbps","thalach","oldpeak"] if c in (rows_nomiss[0].keys() if rows_nomiss else [])]
    rows_clean, outlier_count = remove_outliers_iqr(rows_nomiss, cont_cols_for_outliers)
    final_rows = len(rows_clean)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Original", original_rows)
    c2.metric("Duplicates", duplicate_count)
    c3.metric("Missing", missing_count)
    c4.metric("Outliers", outlier_count)
    c5.metric("Final Rows", final_rows)

    with st.expander("📄 View Cleaned Data (first 10)"):
        st.table(rows_clean[:10])

# Shared preprocessed set for training tabs
# (Re-run simple pipeline here to keep statelessness tidy)
rows_proc = dropna_rows(drop_duplicates(rows))
rows_proc, _ = remove_outliers_iqr(rows_proc, [c for c in ["age","trestbps","thalach","oldpeak"] if rows_proc and c in rows_proc[0]])
if not rows_proc:
    st.stop()
X, y, feature_names = split_Xy(rows_proc, target="target")
mu, sd = zscore_fit(X)
X_scaled = zscore_transform(X, mu, sd)
X_train, X_test, y_train, y_test = stratified_train_test_split(X_scaled, y, test_size=0.2, seed=42)
X_train_bal, y_train_bal = simple_balance_duplicate(X_train, y_train, seed=42)

# ---------- Tab 2: KNN ----------
with tabs[1]:
    st.header("🧠 KNN Pipeline")

    st.subheader("Find Best K for KNN")
    k_range = list(range(1, 21))
    # 5-fold CV (stratified)
    def kfold_indices(y, k=5, seed=42):
        random.seed(seed)
        idx0 = [i for i,t in enumerate(y) if t==0]
        idx1 = [i for i,t in enumerate(y) if t==1]
        random.shuffle(idx0); random.shuffle(idx1)
        folds0 = [idx0[i::k] for i in range(k)]
        folds1 = [idx1[i::k] for i in range(k)]
        folds = [sorted(folds0[i]+folds1[i]) for i in range(k)]
        return folds

    folds = kfold_indices(y_train_bal, k=5, seed=42)
    cv_scores=[]
    for K in k_range:
        accs=[]
        for i in range(5):
            val_idx = set(folds[i])
            Xtr = [X_train_bal[j] for j in range(len(X_train_bal)) if j not in val_idx]
            ytr = [y_train_bal[j] for j in range(len(y_train_bal)) if j not in val_idx]
            Xva = [X_train_bal[j] for j in range(len(X_train_bal)) if j in val_idx]
            yva = [y_train_bal[j] for j in range(len(y_train_bal)) if j in val_idx]
            clf = KNN(k=K); clf.fit(Xtr,ytr)
            yp = clf.predict(Xva)
            acc,_,_,_,_ = precision_recall_f1(yva, yp)
            accs.append(acc)
        cv_scores.append(sum(accs)/len(accs) if accs else 0.0)
    best_idx = max(range(len(cv_scores)), key=lambda i: cv_scores[i])
    best_k = k_range[best_idx]
    best_score = cv_scores[best_idx]

    # Show CV scores bar
    st.markdown(svg_bar_chart([(str(k), s) for k,s in zip(k_range, cv_scores)],
                              title="KNN Accuracy vs K (CV, 5-fold)"), unsafe_allow_html=True)
    st.success(f"🏆 Best K = **{best_k}** with Accuracy = **{best_score:.4f}**")

    # Train best K and evaluate on test
    knn = KNN(k=best_k)
    knn.fit(X_train_bal, y_train_bal)
    y_pred = knn.predict(X_test)

    st.subheader("Evaluation Metrics")
    acc, prec, rec, f1, (tn,fp,fn,tp) = precision_recall_f1(y_test, y_pred)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.2f}")
    c2.metric("Precision", f"{prec:.2f}")
    c3.metric("Recall", f"{rec:.2f}")
    c4.metric("F1 Score", f"{f1:.2f}")

    st.markdown(svg_confusion_matrix(tn,fp,fn,tp, title="Confusion Matrix - KNN"), unsafe_allow_html=True)

# ---------- Tab 3: Logistic Regression ----------
with tabs[2]:
    st.header("📈 Logistic Regression Analysis")

    # Simple "before vs after scaling" visualization (std dev text)
    st.subheader("📊 Feature Scaling (Summary)")
    before_stats = column_stats(rows_proc, feature_names)
    after_stats = column_stats(
        [{feature_names[j]: X_scaled[i][j] for j in range(len(feature_names))} for i in range(len(X_scaled))],
        feature_names
    )
    scale_table = []
    for f in feature_names:
        bs = before_stats[f]; as_ = after_stats[f]
        scale_table.append({
            "Feature": f,
            "Mean (before)": f"{(bs['mean'] or 0):.3f}",
            "Std (before)": f"{(bs['std'] or 0):.3f}",
            "Mean (after)": f"{(as_['mean'] or 0):.3f}",
            "Std (after)": f"{(as_['std'] or 0):.3f}",
        })
    st.table(scale_table[:10])

    # Train
    logreg = LogisticRegressionGD(lr=0.05, epochs=500, l2=0.001)
    logreg.fit(X_train_bal, y_train_bal)
    y_pred = logreg.predict(X_test)

    acc, prec, rec, f1, (tn,fp,fn,tp) = precision_recall_f1(y_test, y_pred)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.2f}")
    c2.metric("Precision", f"{prec:.2f}")
    c3.metric("Recall", f"{rec:.2f}")
    c4.metric("F1 Score", f"{f1:.2f}")

    # "Classification report" text
    report = f"""precision    recall  f1-score  support
0       {(tn/(tn+fp) if (tn+fp)>0 else 0):.2f}     {(tn/(tn+fn) if (tn+fn)>0 else 0):.2f}    {0:.2f}      {tn+fn}
1       {prec:.2f}     {rec:.2f}    {f1:.2f}      {tp+fp}
accuracy                        {acc:.2f}      {len(y_test)}
"""
    with st.expander("📄 Classification Report"):
        st.code(report, language="text")

    st.markdown(svg_confusion_matrix(tn,fp,fn,tp, title="Confusion Matrix - Logistic Regression"), unsafe_allow_html=True)

# ---------- Tab 4: SVM ----------
with tabs[3]:
    st.header("🧠 Support Vector Machine (Linear, Hinge-Loss SGD)")

    svm = LinearSVM_SGD(lr=0.01, epochs=8, C=1.0)  # small epochs to keep snappy
    svm.fit(X_train_bal, y_train_bal)
    y_pred = svm.predict(X_test)

    acc, prec, rec, f1, (tn,fp,fn,tp) = precision_recall_f1(y_test, y_pred)
    st.subheader("📊 Evaluation Metrics")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.2f}")
    c2.metric("Precision", f"{prec:.2f}")
    c3.metric("Recall", f"{rec:.2f}")
    c4.metric("F1 Score", f"{f1:.2f}")

    # "Classification report"
    report = f"""precision    recall  f1-score  support
0       {(tn/(tn+fp) if (tn+fp)>0 else 0):.2f}     {(tn/(tn+fn) if (tn+fn)>0 else 0):.2f}    {0:.2f}      {tn+fn}
1       {prec:.2f}     {rec:.2f}    {f1:.2f}      {tp+fp}
accuracy                        {acc:.2f}      {len(y_test)}
"""
    st.text("📄 Classification Report")
    st.code(report, language="text")

    st.subheader("📉 Confusion Matrix")
    st.markdown(svg_confusion_matrix(tn,fp,fn,tp, title="Confusion Matrix - SVM"), unsafe_allow_html=True)

    # Simple ROC using raw decision scores
    st.subheader("🎯 ROC (uncalibrated scores)")
    scores = [s for s in svm.decision_function(X_test)]
    fprs, tprs, _ = roc_curve_points(y_test, scores)
    aucv = auc_trapezoid(sorted(fprs), [y for _,y in sorted(zip(fprs,tprs))])
    st.markdown(svg_roc(fprs, tprs, aucv, title="ROC Curve - SVM"), unsafe_allow_html=True)

# ---------- Tab 5: Model Comparison ----------
with tabs[4]:
    st.header("📊 Model Comparison (with Simple Voting)")

    # Train all
    knn = KNN(k=5); knn.fit(X_train_bal, y_train_bal)
    logreg = LogisticRegressionGD(lr=0.05, epochs=500, l2=0.001); logreg.fit(X_train_bal, y_train_bal)
    svm = LinearSVM_SGD(lr=0.01, epochs=8, C=1.0); svm.fit(X_train_bal, y_train_bal)

    models = {
        "KNN (k=5)": knn,
        "Logistic Regression": logreg,
        "SVM (Linear)": svm
    }

    results=[]
    for name, m in models.items():
        yp = m.predict(X_test)
        acc,prec,rec,f1,(tn,fp,fn,tp) = precision_recall_f1(y_test, yp)
        results.append({"Model":name, "Accuracy":acc, "Precision":prec, "Recall":rec, "F1 Score":f1, "CM":(tn,fp,fn,tp)})

    # Voting (hard + soft using available proba/score)
    # Hard vote
    yp_hard=[]
    for i in range(len(X_test)):
        votes = [models["KNN (k=5)"].predict([X_test[i]])[0],
                 models["Logistic Regression"].predict([X_test[i]])[0],
                 models["SVM (Linear)"].predict([X_test[i]])[0]]
        yp_hard.append(1 if sum(votes)>=2 else 0)
    acc,prec,rec,f1,(tn,fp,fn,tp) = precision_recall_f1(y_test, yp_hard)
    results.append({"Model":"Combined (Hard Vote)","Accuracy":acc,"Precision":prec,"Recall":rec,"F1 Score":f1,"CM":(tn,fp,fn,tp)})

    # Soft vote: average of probabilities (SVM uses logistic of score)
    yp_soft_probs=[]
    for i in range(len(X_test)):
        ps = []
        for name in models:
            p1 = models[name].predict_proba([X_test[i]])[0][1]
            ps.append(p1)
        p = sum(ps)/len(ps)
        yp_soft_probs.append(p)
    yp_soft = [1 if p>=0.5 else 0 for p in yp_soft_probs]
    acc,prec,rec,f1,(tn,fp,fn,tp) = precision_recall_f1(y_test, yp_soft)
    results.append({"Model":"Combined (Soft Vote)","Accuracy":acc,"Precision":prec,"Recall":rec,"F1 Score":f1,"CM":(tn,fp,fn,tp)})

    # Show table
    show_simple = [{"Model":r["Model"],
                    "Accuracy":f'{r["Accuracy"]:.3f}',
                    "Precision":f'{r["Precision"]:.3f}',
                    "Recall":f'{r["Recall"]:.3f}',
                    "F1 Score":f'{r["F1 Score"]:.3f}'} for r in results]
    st.table(show_simple)

    # Bar of metrics (F1 only for compactness)
    st.subheader("🔍 Metric Comparison (F1 Score)")
    st.markdown(svg_bar_chart([(r["Model"], r["F1 Score"]) for r in results],
                              title="F1 Score by Model"), unsafe_allow_html=True)

    # ROC curves (where we have a score/prob)
    st.subheader("📈 ROC Curves")
    roc_svgs=[]
    for name, m in models.items():
        if hasattr(m, "predict_proba"):
            scores = [m.predict_proba([x])[0][1] for x in X_test]
        elif hasattr(m, "decision_function"):
            scores = m.decision_function(X_test)
        else:
            continue
        fprs, tprs, _ = roc_curve_points(y_test, scores)
        aucv = auc_trapezoid(sorted(fprs), [y for _,y in sorted(zip(fprs,tprs))])
        roc_svgs.append((name, svg_roc(fprs,tprs,aucv, title=f"ROC - {name}")))
    for name, svg in roc_svgs:
        st.markdown(svg, unsafe_allow_html=True)

    # Confusion Matrices
    st.subheader("🧮 Confusion Matrices")
    for r in results:
        tn,fp,fn,tp = r["CM"]
        st.markdown(svg_confusion_matrix(tn,fp,fn,tp, title=f"Confusion Matrix - {r['Model']}"),
                    unsafe_allow_html=True)

    # Best model by F1
    best = max(results, key=lambda r: r["F1 Score"])
    st.subheader("🥇 Best Performing Model")
    st.success(f"**{best['Model']}** performed best with F1 Score: **{best['F1 Score']:.3f}**")

