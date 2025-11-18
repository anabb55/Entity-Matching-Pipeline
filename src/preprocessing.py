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
    "ââ": " ",
    "å": "a",

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
    s = re.sub(r"[^\w\s,\-]", " ", s)
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


def jaccard(set1, set2) -> float:
    if not set1 and not set2:
        return 0.0
    
    return len(set1 & set2) / len(set1 | set2)


def deduplicate_sorted(df):
    df = df.copy()
    df = df.sort_values("title").reset_index(drop=True)

    df["title_tokens"] = df["title"].apply(tokenize_title)
    df["author_lastnames"] = df["authors"].apply(extract_last_name)

    to_drop = set()
    groups =[]

    i = 0

    while i < len(df) - 1:
        j = i + 1

        if j in to_drop:
            i += 1
            continue

        tokens_i = df.loc[i, "title_tokens"]
        tokens_j = df.loc[j, "title_tokens"]
        authors_i = df.loc[i, "author_lastnames"]
        authors_j = df.loc[j, "author_lastnames"]

        if tokens_i and tokens_j:
            sim = jaccard(tokens_i, tokens_j)
        else:
            sim = 0.0


        same_all_authors = (authors_i == authors_j) and len(authors_i) > 0

        if (sim >= 0.9 and len(authors_i & authors_j) > 0):
            to_drop.add(j)
            groups.append([i, j])

        i += 1

    deduplicated = df.drop(index = list(to_drop)).reset_index(drop=True)
    deduplicated = deduplicated.drop(columns = ["title_tokens", "author_lastnames"], errors= "ignore")

    print("Input records:", len(df))
    print("Duplicate groups:", len(groups))
    print("Final unique records:", len(deduplicated))

    return deduplicated

   


def is_numeric_heavy(tokens: set):
    if not tokens:
        return True

    num_tokens = sum(t.isdigit() for t in tokens)
    return num_tokens / len(tokens) >= 0.5
   
              
def is_bad_title_shape(title: str):
    if not isinstance(title, str):
        return True

    s = title.strip()
    if not s:
        return True

    if len(s) < 5:
        return True
    
    alpha_count = sum(ch.isalpha() for ch in s)
    if alpha_count < 3:
        return True

    return False


def is_weird_character(title: str):
    if not isinstance(title, str):
        return True

    s = title.strip()
    if not s:
        return True

    total = len(s)
    non_ascii = sum(ord(ch) > 127 for ch in s) ## if ASCII code is above 127 -> weird characters
    ratio = non_ascii / total

    return ratio > 0.8


def mark_noisy_rows(df):
    df = df.copy()

    df["title_tokens"] = df["title"].apply(tokenize_title)

    df["is_noisy"] = (
         df["title"].apply(is_bad_title_shape) | df["title"].apply(is_weird_character))

    df = df.drop(columns = ["title_tokens"])
    return df


def filter_noisy_rows(df):
    df_marked = mark_noisy_rows(df)
    clean_df = df_marked[df_marked["is_noisy"] == False].copy()
    return clean_df.drop(columns = ["is_noisy"])


   
cols_to_clean = ["title", "venue", "authors"]
cols_keep = ["id", "year"]

scholar_clean = make_clean_df(scholar_data, cols_keep, cols_to_clean)
DBLP_clean = make_clean_df(DBLP_data, cols_keep, cols_to_clean)

scholar_clean = filter_noisy_rows(scholar_clean)
DBLP_clean = filter_noisy_rows(DBLP_clean)

scholar_clean.to_csv("data/Scholar_cleaned.csv", index=False)
DBLP_clean.to_csv("data/DBLP_cleaned.csv", index=False)


scholar_deduplicated= deduplicate_sorted(scholar_clean)
scholar_deduplicated.to_csv("data/Scholar_deduplicated.csv", index=False)
DBLP_deduplicated = deduplicate_sorted(DBLP_clean)
DBLP_deduplicated.to_csv("data/DBLP_deduplicated.csv", index=False)

## test
# last_n = extract_last_name(scholar_clean.iloc[62]["authors"])
# print(last_n)








