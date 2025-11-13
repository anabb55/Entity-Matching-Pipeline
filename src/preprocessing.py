import pandas as pd
import html
import ftfy
import re
import unicodedata

scholar_data = pd.read_csv("./data/Scholar.csv")
DBLP_data = pd.read_csv("./data/DBLP1.csv", encoding="latin1")

DBLP_data.info()
DBLP_data.head()

## there are no exact duplicates
# duplicates = scholar_data[scholar_data.duplicated(keep=False)]
# print(duplicates)

replace_map = {
    "â??â??": " - ",
    "â??": "'",
    "Â?": "'",
    "â?¦": "...",

}

stopwords = {
    "and", "or", "in", "on", "at",
    "of", "for", "the", "a", "an",
    "to", "with", "by"
}

def clean_text(text:str) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = html.unescape(text)
    text = ftfy.fix_text(text)   
    for bad, good in replace_map.items():
        text = text.replace(bad, good)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def clean_authors(text: str) -> str:
    if pd.isna(text) or not str(text).strip():
        return ""
    s = str(text)
    s = html.unescape(s)
    s = ftfy.fix_text(s)
    for bad, good in replace_map.items():
        s = s.replace(bad, good)
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()

    return s
    


def make_clean_df(df, cols_keep, cols_to_clean):
    clean_df = df[cols_keep].copy()
    clean_df["year"] = (pd.to_numeric(clean_df["year"], errors="coerce").astype("Int64"))
    for col in cols_to_clean:
        if col == "authors":
            clean_df[col] = df[col].apply(clean_authors)
        else:
             clean_df[col] = df[col].apply(clean_text)
            
    
    return clean_df


def tokenize_title(title:str) -> set:
    cleaned = clean_text(title)
    if not cleaned:
        return set()
    tokens = cleaned.split()
    tokens = [t for t in tokens if len(t) > 1]
    tokens = [t for t in tokens if t not in stopwords]

    return set(tokens)


def extract_last_name(authors:str) -> set:
    if pd.isna(authors) or not str(authors).strip():
        return set()
    s = str(authors).strip()
    s = s.replace(" and ", " , ")
    s = s.replace(" & ", " , ")
    s = s.replace(";", " , ")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    last_names = set()

    for part in parts:
        part = re.sub(r"[^\w\s\-]", " ", part)
        tokens = [t for t in part.split() if t]
        tokens = [t for t in tokens if len(t) > 1]
        tokens = [t for t in tokens if not t.isdigit()]
        
        if not tokens:
            continue

        last_name = tokens[-1]
        last_names.add(last_name)

    return last_names



   
cols_to_clean = ["title", "venue", "authors"]
cols_keep = ["id", "year"]

scholar_clean = make_clean_df(scholar_data, cols_keep, cols_to_clean)
DBLP_clean = make_clean_df(DBLP_data, cols_keep, cols_to_clean)
scholar_clean.to_csv("data/Scholar_cleaned.csv", index=False)
DBLP_clean.to_csv("data/DBLP_cleaned.csv", index=False)

## test
last_n = extract_last_name(scholar_clean.iloc[62]["authors"])
print(last_n)








