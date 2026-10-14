import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    quasis = [col for col in anon_df.columns if col in aux_df.columns and col not in ("anon_id", "name")]
    merged = anon_df[["anon_id"] + quasis].merge(aux_df[["name"] + quasis], on=quasis, how="inner")
    unique = merged[merged.groupby("anon_id")["name"].transform("size") == 1]  
    return unique[["anon_id", "name"]].rename(columns={"name": "matched_name"})

def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    if len(anon_df) == 0:
        return 0.0
    return len(matches_df) / len(anon_df)
