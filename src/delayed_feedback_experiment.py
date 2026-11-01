import sys
from pathlib import Path
sys.path.insert(0, str(Path('src').resolve()))
import pandas as pd
import numpy as np
import config
from data_loader import load_base_dataset
from features import FeaturePreprocessor
from evaluation import threshold_for_fpr, compute_metrics
from delayed_feedback import get_training_slice_under_delay
import lightgbm as lgb

def run_delayed_benchmark(sample_frac=0.02, seed=42):
    df = load_base_dataset(sample_frac=sample_frac, random_state=seed)
    results = []
    for lag in [0, 1, 2]:
        train = get_training_slice_under_delay(df, current_month=6, delay_months=lag)
        test = df[df['month'] >= 6]
        if len(train) < 50:
            continue
        pre = FeaturePreprocessor().fit(train)
        X_tr = pre.transform(train)
        y_tr = train['fraud_bool'].values
        X_te = pre.transform(test)
        y_te = test['fraud_bool'].values
        spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
        model = lgb.LGBMClassifier(n_estimators=50, scale_pos_weight=spw, random_state=seed, verbosity=-1)
        model.fit(X_tr, y_tr)
        p_te = model.predict_proba(X_te)[:, 1]
        thr = threshold_for_fpr(y_tr, model.predict_proba(X_tr)[:, 1], 0.05)
        met = compute_metrics(y_te, p_te, thr)
        results.append({
            'delay_lag_months': lag,
            'n_training_verified': len(train),
            'ROC_AUC': met['ROC_AUC'],
            'PR_AUC': met['PR_AUC'],
            'TPR_at_5pct_FPR': met['TPR_at_5pct_FPR'],
        })
    rep = pd.DataFrame(results)
    rep.to_csv(config.OUTPUT_DIR / 'delayed_feedback_benchmark.csv', index=False)
    print('DELAYED_FEEDBACK_BENCHMARK_COMPLETED')
    print(rep)

if __name__ == '__main__':
    run_delayed_benchmark()
