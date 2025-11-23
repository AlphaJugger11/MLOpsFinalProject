import pandas as pd, sys
paths = ["federated_clients/client_1.csv","federated_clients/client_2.csv","federated_clients/client_3.csv"]
for p in paths:
    try:
        df = pd.read_csv(p)
        cols = [c for c in df.columns if c not in ("subject","activity")]
        print(p, "rows:", len(df), "features(in_dim):", len(cols), "classes:", df["activity"].nunique() if "activity" in df.columns else "unknown")
    except Exception as e:
        print("ERROR reading", p, e)
        sys.exit(1)
