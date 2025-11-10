import pandas as pd

scholar_data = pd.read_csv("./data/Scholar.csv")
DBLP_data = pd.read_csv("./data/DBLP1.csv", encoding="latin1")

DBLP_data.info()
DBLP_data.head()
DBLP_data.sample(5)

def clean_text(text):
    text = text.lower().strip()