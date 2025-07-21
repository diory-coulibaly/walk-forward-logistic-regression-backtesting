import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

st.title("📈 Walk-Forward Backtesting App - Logistic Regression")

# File upload
st.sidebar.header("Upload CSV Files")
goog_file = st.sidebar.file_uploader("Upload GOOG.csv", type=["csv"])
sp500_file = st.sidebar.file_uploader("Upload S&P500.csv", type=["csv"])

# Function to load and clean data from semicolon-separated files with space-as-decimal
@st.cache_data
def load_and_clean(file, colname):
    df = pd.read_csv(file, sep=';')
    df = df[['Date', 'Adj.Close']].copy()
    df.columns = ['Date', colname]

    # Replace comma with dot and convert to float
    df[colname] = df[colname].astype(str).str.replace(',', '.').astype(float)

    # Parse date
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df.dropna(inplace=True)
    return df


if goog_file and sp500_file:
    goog = load_and_clean(goog_file, 'Goog')
    sp500 = load_and_clean(sp500_file, 'SP500')

    # Merge on date
    df = pd.merge(goog, sp500, on='Date', how='inner')
    df = df.sort_values('Date')

    # Compute returns
    df['Goog'] = df['Goog'].pct_change()
    df['SP500'] = df['SP500'].pct_change()

    # Lag and target setup
    lag = st.sidebar.slider("Select Lag (days)", 1, 10, 2)
    window_size = st.sidebar.slider("Training Window (days)", 50, 300, 100)

    df[f'Goog_lag{lag}'] = df['Goog'].shift(lag)
    df[f'SP500_lag{lag}'] = df['SP500'].shift(lag)
    df['Target'] = (df['Goog'] > 0).astype(int)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    preds = []
    for start in range(0, len(df) - window_size - 1):
        end = start + window_size
        train = df.iloc[start:end]
        test = df.iloc[end:end + 1]

        X_train = train[[f'Goog_lag{lag}', f'SP500_lag{lag}']]
        y_train = train['Target']
        X_test = test[[f'Goog_lag{lag}', f'SP500_lag{lag}']]
        y_test = test['Target'].values[0]
        test_date = test['Date'].values[0]

        if X_train.nunique().min() <= 1:
            continue

        try:
            model = LogisticRegression(solver='liblinear')
            model.fit(X_train, y_train)
            prob = model.predict_proba(X_test)[0][1]
            pred_class = model.predict(X_test)[0]

            preds.append({
                'Date': test_date,
                'True': y_test,
                'Pred_Prob': prob,
                'Pred_Class': pred_class
            })
        except:
            continue

    results = pd.DataFrame(preds)
    if not results.empty:
        st.success(f"✅ Predictions complete. Total: {len(results)}")

        # Metrics
        acc = accuracy_score(results['True'], results['Pred_Class'])
        fpr, tpr, _ = roc_curve(results['True'], results['Pred_Prob'])
        roc_auc = auc(fpr, tpr)
        cm = confusion_matrix(results['True'], results['Pred_Class'])

        st.metric("Accuracy", f"{acc:.2f}")
        st.metric("AUC", f"{roc_auc:.2f}")

        st.subheader("Confusion Matrix")
        st.write(pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Pred 0', 'Pred 1']))

        st.subheader("ROC Curve")
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        st.pyplot(fig)

        st.download_button("📥 Download Predictions CSV", data=results.to_csv(index=False), file_name="predictions.csv")
    else:
        st.warning("Not enough valid data to generate predictions.")
else:
    st.info("Please upload both GOOG.csv and S&P500.csv to begin.")
