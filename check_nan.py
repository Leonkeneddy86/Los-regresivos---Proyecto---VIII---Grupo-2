import pandas as pd

cols = ['Gender_label','Customer_Type_label','Type_of_Travel_label','Class_label','Age','Flight Distance','Inflight wifi service',
        'Departure/Arrival time convenient','Ease of Online booking','Gate location','Food and drink','Online boarding','Seat comfort',
        'Inflight entertainment','On-board service','Leg room service','Baggage handling','Checkin service','Inflight service',
        'Cleanliness','Departure Delay in Minutes','Arrival Delay in Minutes']

try:
    df = pd.read_csv('data/clean_data.csv')
except Exception as e:
    print('ERROR reading CSV:', e)
    raise

missing = [c for c in cols if c not in df.columns]
if missing:
    print('MISSING_COLUMNS:', missing)

present = [c for c in cols if c in df.columns]
if present:
    nan_counts = df[present].isna().sum()
    print('\nNaN counts for listed columns:')
    for c, cnt in nan_counts.items():
        print(f"{c}: {cnt}")

    any_nan = nan_counts[nan_counts>0]
    if any_nan.empty:
        print('\nRESULT: No listed columns contain NaN.')
    else:
        print('\nRESULT: Columns with NaN:')
        for c, cnt in any_nan.items():
            print(f"  - {c}: {cnt} NaN values")
else:
    print('No listed columns are present in the CSV.')
