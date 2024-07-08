import pandas as pd



df = pd.read_excel("ML1/data.xlsx")
df.head()


def proba_to_predict(x, threshold=0.5):
    if x < threshold:
        return 0
    else:
        return 1

df["Tahmin"] = df["Model Olasılık Tahmini"].apply(lambda x: proba_to_predict(x, 0.5))



