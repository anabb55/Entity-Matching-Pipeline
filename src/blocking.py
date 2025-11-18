import pandas as pd
from itertools import product
from preprocessing import extract_last_name, stopwords

scholar = pd.read_csv("data/Scholar_deduplicated.csv")
dblp = pd.read_csv("data/DBLP_deduplicated.csv")
perfect = pd.read_csv("data/DBLP-Scholar_perfectMapping.csv")

scholar_copy = scholar.copy()
dblp_copy = dblp.copy()

scholar_copy.rename(columns={"id" : "id_s"}, inplace=True)
dblp_copy.rename(columns = {"id" : "id_d"}, inplace=True)

def candidate_pairs_from_block(df_l, df_r, key_col_l, key_col_r, id_col_l, id_col_r):
    pairs = set()

    left_group = df_l.groupby(key_col_l)[id_col_l].apply(list)
    right_group = df_r.groupby(key_col_r)[id_col_r].apply(list)

    common_keys = left_group.index.intersection(right_group.index)  ## index is for example "comp" as part of the title; we should find rows in both datasets that have the same beginning of the title

    for key in common_keys:
        ids_l = left_group[key]
        ids_r = right_group[key]
        for l, r in product(ids_l, ids_r):
            pairs.add((l, r))

    return pairs


def separate_lastnames(df, id_col):
    tmp = df[[id_col, "authors"]].copy()
    tmp["lastnames_set"] = tmp["authors"].apply(extract_last_name)
    tmp = tmp.explode("lastnames_set")
    tmp = tmp[tmp["lastnames_set"].notna()]
    tmp = tmp[tmp["lastnames_set"] != ""]
    tmp.rename(columns={"lastnames_set": "lastname"}, inplace=True)

    return tmp[[id_col, "lastname"]]

def first_content_token(title: str):
    if not isinstance(title, str):
        return ""
    tokens = title.split()
    tokens =[t for t in tokens if t not in stopwords and len(t) > 1]
    return tokens[0] if tokens else ""


scholar_authors = separate_lastnames(scholar_copy, id_col="id_s")
dblp_authors = separate_lastnames(dblp_copy, id_col="id_d")

author_pairs = candidate_pairs_from_block(scholar_authors, dblp_authors, key_col_l="lastname", key_col_r="lastname", id_col_l="id_s", id_col_r="id_d")
print("Author-based candidate pairs:", len(author_pairs))

scholar_copy["title_first"] = scholar_copy["title"].apply(first_content_token)
dblp_copy["title_first"] = dblp_copy["title"].apply(first_content_token)

title_pairs = candidate_pairs_from_block(scholar_copy, dblp_copy, key_col_l="title_first", key_col_r="title_first", id_col_l="id_s", id_col_r="id_d")
print("Title-based candidate pairs", len(title_pairs)) 



core_pairs = author_pairs & title_pairs 
candidate_pairs = core_pairs | title_pairs
candidate_pairs_count = len(candidate_pairs)
print("Union:", candidate_pairs_count)

gold_set = set((row["idScholar"], row["idDBLP"]) for _, row in perfect.iterrows())
gold_total = len(gold_set)
gold_in_candidates = gold_set & candidate_pairs
gold_caught = len(gold_in_candidates)

gold_retained = 100 * gold_caught / gold_total if gold_total > 0 else 0.0

print("Gold total: ",gold_total)
print("Gold pairs in candidates: ", gold_caught)
print("Gold retained: ", gold_retained)



scholar_ids = set(scholar_copy["id_s"])
dblp_ids    = set(dblp_copy["id_d"])

gold_set = set(
    (row["idScholar"], row["idDBLP"])
    for _, row in perfect.iterrows()
)

gold_total = len(gold_set)
print("Gold total (original):", gold_total)

filtered_gold_set = set(
    (s, d)
    for (s, d) in gold_set
    if (s in scholar_ids) and (d in dblp_ids)
)

filtered_gold_total = len(filtered_gold_set)
print("Gold pairs that still exist after cleaning/dedup:", filtered_gold_total)
print("Gold pairs lost due to preprocessing:", gold_total - filtered_gold_total)