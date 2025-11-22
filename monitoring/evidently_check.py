# monitoring/evidently_check.py
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping

def run_drift_check(reference_csv, current_csv, out_html='/tmp/drift_report.html'):
    ref = pd.read_csv(reference_csv)
    curr = pd.read_csv(current_csv)
    report = Report(metrics=[DataDriftPreset()])
    col_map = ColumnMapping()
    report.run(reference_data=ref, current_data=curr, column_mapping=col_map)
    report.save_html(out_html)
    return out_html

if __name__ == '__main__':
    print(run_drift_check('reference.csv', 'current.csv'))
