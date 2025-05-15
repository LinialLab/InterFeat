
#!/usr/bin/env python
# coding: utf-8
### refactored via AI version of code. Need to validate no other changes beyond the functionizing!

import re
import string
import os
import ssl
import pandas as pd
import sqlite3
import numpy as np
from Bio import Entrez
from scipy.stats import fisher_exact, chi2_contingency
from time import sleep
from tqdm import tqdm

rem = string.punctuation
punct_pattern = r"[{}]".format(rem)#  use to strip out whitespace/punctuation from texts
from Bio import Entrez
import os
## TODO - add in environ - api key vals for entrez
Entrez.email = os.environ.get('ENTREZ_EMAIL')
Entrez.api_key = os.environ.get('ENTREZ_API_KEY')

import ssl
ssl._create_default_https_context = ssl._create_stdlib_context
def init_db():
    """
    Initializes the SQLite database to cache PubMed search results.
    Creates a table 'results' if it does not exist with columns 'term' and 'count'.
    """
    conn = sqlite3.connect('pubmed_results.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS results
                 (term TEXT PRIMARY KEY, count INTEGER)''')
    conn.commit()
    conn.close()


def cache_search(term):
    """
    Checks if the search count for the given term is cached in the database.
    If cached, returns the count. Otherwise, performs a PubMed search for the term,
    caches the result, and returns the count.

    Parameters:
        term (str): The search term.

    Returns:
        int: The number of search results for the term.
    """
    conn = sqlite3.connect('pubmed_results.db')
    c = conn.cursor()
    c.execute("SELECT count FROM results WHERE term=?", (term,))
    result = c.fetchone()
    if result:
        conn.close()
        return result[0]
    else:
        count = search_pubmed(term)
        c.execute("INSERT INTO results (term, count) VALUES (?, ?)", (term, count))
        conn.commit()
        conn.close()
        return count


def search_pubmed(term):
    """
    Searches PubMed for the given term and returns the number of results.

    Parameters:
        term (str): The search term.

    Returns:
        int: The number of search results for the term.
    """
    # Entrez.email and Entrez.api_key should be set before calling this function.
    handle = Entrez.esearch(db="pubmed", term=term, retmax=0)
    record = Entrez.read(handle)
    handle.close()
    return int(record["Count"])


def collect_search_data(queries, targets, total=37e6, min_results_count=20,concat_target_terms=True):
    """
    Collects search data from PubMed for given queries and targets,
    and calculates statistics on their co-occurrence.

    Parameters:
        queries (set): A set of query terms.
        targets (list): A list of target terms.
        total (int): Total number of PubMed documents.
        min_results_count (int): Minimum number of results required for queries and targets.

    Returns:
        pd.DataFrame: A DataFrame containing the search results and statistical measures.
    """
    results = []
    total_operations = len(queries) * len(targets)
    progress_bar = tqdm(total=total_operations, desc="Collecting Data", leave=True)
    ## merge targets - with and - done by default and done in advance normally?
    if concat_target_terms:
        if len(targets) > 1:  # added, maybe not needed?
            targets = [" OR ".join("({})".format(x) for x in targets)]  ##  normally, enable this
        print("Targets:", targets)

    for query in queries:
        q_count = cache_search(query)
        if q_count > min_results_count:
            for target in targets:
                t_count = cache_search(target)
                if t_count > min_results_count:
                    combined_term = f"({query}) AND ({target})"
                    qt_count = cache_search(combined_term)

                    a = qt_count
                    b = max(t_count - qt_count, 0)
                    c = max(q_count - qt_count, 0)
                    d = max(total - (a + b + c), 0)
                    table = [
                        [a, b],
                        [c, d]]

                    try:
                        odds_ratio, p_value = fisher_exact(table, 'less')
                    except Exception as e:
                        print(f"Error performing Fisher Exact Test for query '{query}' and target '{target}': {e}")
                        odds_ratio = np.NaN
                        p_value = np.NaN

                    # Mutual Information calculation
                    epsilon = 1e-10
                    numerator = total * a
                    denominator = (q_count * t_count) + epsilon
                    if a > 0 and numerator > 0:
                        MI = np.log2(numerator / denominator)
                    else:
                        MI = 0  # or np.NaN

                    results.append({
                        'Query': query,
                        'Target': target,
                        'Query Count': q_count,
                        'Target Count': t_count,
                        'Co-occurrence Count': qt_count,
                        'Odds Ratio': odds_ratio,
                        'Cooccurrence P-Value': p_value,
                        "MI": MI,
                        })
                progress_bar.update(1)
    progress_bar.close()
    return pd.DataFrame(results).sort_values(['Cooccurrence P-Value', 'Query', 'Target', "Co-occurrence Count"], ascending=True).round(3)


def load_candidate_queries(query_candidates_filepath: str = None,df:pd.DataFrame = None, do_path_filter: bool = False):
    """
    Loads candidate queries from a CSV file and filters them based on given criteria.

    Parameters:
        query_candidates_filepath (str): Path to the CSV file containing candidate queries.
        do_path_filter (bool): Whether to apply path length filtering.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered candidate queries.
    """
    # df_query = pd.read_csv(query_candidates_filepath)
    if df is None:
        df_query = pd.read_csv(query_candidates_filepath)
    else:
        if df is None:
            print("no candidates file or df given")
            return None
        df_query = df  # Use the provided dataframe directly

    df_query = df_query.drop_duplicates(subset=[c for c in ["feature_name","cui_nomenclature"] if c in df_query.columns])
    if "KG_Hits" in df_query.columns: # KG based filtering, if that stage was run
        df_query = df_query.loc[df_query["KG_Hits"] <= 0]
        df_query = df_query.loc[df_query["feature_level_min_kg_hits"] <= 1].drop_duplicates().reset_index(drop=True)
    else:
        print("NOTE: Imputing KG_Hits col")
        df_query["KG_Hits"]=0
    if do_path_filter:
        print(df_query.shape[0], "pre feat imp filter")
        df_query = df_query.loc[(df_query["feature_importance"] > 0) | (df_query["MutualInfoTarget"] >= 0.001)]
    print(df_query.shape[0], "# df_query rows")
    if do_path_filter:
        df_query = df_query.query("(shortest_path_length>2) | (simple_path_length<20)")
        print("After path length combined filt:", df_query.shape[0])
    df_query = df_query.reset_index(drop=True).copy()
    print(df_query.nunique())
    return df_query


def add_fnames_to_queries(df_query: pd.DataFrame):
    """
    Adds feature names to the list of queries, with cleaning and deduplication.

    Parameters:
        df_query (pd.DataFrame): DataFrame containing candidate queries.

    Returns:
        pd.DataFrame: Updated DataFrame with feature names added to queries.
    """
    df = df_query.drop_duplicates("feature_name").copy()
    s1 = df.shape[0]
    if "cui_nomenclature" not in df_query.columns: # handle cases of running on partial data # added
        df_query["cui_nomenclature"] = df_query["feature_name"]#.str.lower()
    df = df.loc[df["feature_name"].str.lower() != df["cui_nomenclature"].str.lower()]
    print(df.shape[0] - s1, "# rows dropped, where feature name == cui")
    df["feature_name"] = df["feature_name"].str.replace(r"(qualifier value)", "", regex=False).str.replace(r"(qualifier)", "",
                                                                                                           regex=False).str.strip()
    df["feature_name"] = df["feature_name"].str.replace(punct_pattern, ' ', regex=True).str.replace("  ", ' ', regex=False).str.strip()
    df["cui_nomenclature"] = df["feature_name"]
    df["cui"] = ""
    print(df_query.shape[0], "# pre merge")
    df_query = pd.concat([df, df_query], ignore_index=True).drop_duplicates().copy()
    print(df_query.shape[0])
    return df_query


def get_promising_results(result_df: pd.DataFrame, SIGNIFICANT_PVAL=0.4, Drop_Query_Cui_Info=True, filter_joint_pathlength=False):
    """
    Filters and selects promising results based on statistical significance and feature importance.

    Parameters:
        result_df (pd.DataFrame): DataFrame containing the search results and related metrics.
        SIGNIFICANT_PVAL (float): Significance level for p-values in filtering.
        Drop_Query_Cui_Info (bool): Whether to drop query CUI information.
        filter_joint_pathlength (bool): Whether to filter based on joint path length.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered and sorted promising results.
    """
    pick_result_df = result_df.sort_values(["cui", "sim_score",
                                            "feature_level_sum_kg_hits", "Co-occurrence Count"], ascending=[True, False, True, True]).copy()
    pick_result_df = pick_result_df.loc[(pick_result_df["min_feat_kgHit"] <= 1) | (pick_result_df['Cooccurrence P-Value'] <= SIGNIFICANT_PVAL) \
                                        | (pick_result_df["Co-occurrence Count"] <= 4)]
    ## add:
    pick_result_df = pick_result_df.loc[(pick_result_df['Cooccurrence P-Value'] <= SIGNIFICANT_PVAL) \
                                    | (pick_result_df["Co-occurrence Count"] <= 10)]                                    
    print(pick_result_df.shape[0], "# rows after feature_level_avg_kg_hits filter")
    # pick_result_df = pick_result_df.query("sim_score>=0.07")
    print(pick_result_df.shape[0], "# rows after (local)sim_score filter")
    pick_result_df = pick_result_df.loc[(pick_result_df[['Query Count', "Target Count"]].min(axis=1) >= 20)]
    # pick_result_df = pick_result_df.loc[((pick_result_df["feature_importance"] >= 0.001) & (pick_result_df["Co-occurrence Count"] >= 4)) \
    #                                     | (pick_result_df["Co-occurrence Count"] <= 3)] # skip this
    print(pick_result_df.shape[0], "# rows after query-target count filter")
    pick_result_df = pick_result_df.loc[(pick_result_df["feature_importance"] > 0.001) | (pick_result_df["p_val"] < 0.2) | (
            pick_result_df["MutualInfoTarget"] > 0.001)]
    # pick_result_df = pick_result_df.loc[(pick_result_df["feature_importance"] >= 0.01) | (pick_result_df["p_val"] < 0.4) | (
    #         pick_result_df["MutualInfoTarget"] >= 0.01)]
    pick_result_df.sort_values(["feature_name", "Co-occurrence Count", "feature_importance", "MutualInfoTarget",
                                'KG_Hits'], ascending=[True, True, False, False, True], inplace=True)
    if Drop_Query_Cui_Info:
        pick_result_df = pick_result_df.drop_duplicates(subset=["feature_name",
                                                                'KG_Hits'
                                                                ])
    else:
        pick_result_df = pick_result_df.drop_duplicates(subset=["feature_name",
                                                                "Query",
                                                                'KG_Hits'
                                                                ])
    pick_result_df = pick_result_df.filter(['feature_name',
                                     "raw_name",
                                     'feature_importance', 'p_val', 'corr', 'MutualInfoTarget',
                                     'Co-occurrence Count',
                                     'F.Split-Lift (y==1)',
                                     'F.Split-Feature Split',
                                     'Query Count', 'Target',
                                     'sim_score',
                                     "shortest_path_length", "simple_path_length"
                                     ],axis=1).round(3)
    pick_result_df.drop(columns=['cui',
                                 "KG_Hits",
                                 ], errors="ignore", inplace=True)
    if Drop_Query_Cui_Info:
        pick_result_df = pick_result_df.drop(['cui_def', "Query"], errors="ignore").drop_duplicates()
    print(pick_result_df.shape[0])
    if filter_joint_pathlength:
        pick_result_df = pick_result_df.loc[(pick_result_df["shortest_path_length"] > 2) | (pick_result_df["simple_path_length"] < 20)]
        print(pick_result_df.shape[0], "After dropping close hop connections")
    pick_result_df.sort_values(["Co-occurrence Count", "feature_importance", "MutualInfoTarget", "feature_name"],
                               ascending=[True, False, False, False], inplace=True)
    if Drop_Query_Cui_Info:
        pick_result_df["clean_featName"] = pick_result_df["feature_name"].str.lower().str.replace("[^A-Za-z0-9]", '', regex=True).str.replace(
            "[-\(\)]", "").str.replace('\s+', ' ',
                                       case=False,
                                       regex=True).str.strip().str.replace("  ", " ").str.strip()
        pick_result_df = pick_result_df.drop_duplicates(subset=["clean_featName"])
        pick_result_df.drop(columns=["clean_featName"], inplace=True)
    else:
        pick_result_df = pick_result_df.drop_duplicates(subset=["feature_name",
                                                                "Query",
                                                                ])
    pick_result_df = pick_result_df.round(3)
    pick_result_df["Novel/Interesting?"] = None
    pick_result_df["Makes Sense?"] = None
    pick_result_df["COMMENTS"] = ""
    return pick_result_df

## models version
def filter_boring_feature_names_cooc(result_df, SIGNIFICANT_PVAL=0.6, cooc_count_filter_val=15, min_query_count=80, empty_concept_col="cui",
                                     name_col="Query"):
    """
    Identifies 'boring' feature names based on co-occurrence statistics,
    which are not novel or interesting for further analysis.

    Parameters:
        result_df (pd.DataFrame): DataFrame containing the search results.
        SIGNIFICANT_PVAL (float): Significance level for p-values in filtering.
        cooc_count_filter_val (int): Co-occurrence count threshold for filtering.
        min_query_count (int): Minimum query count required.
        empty_concept_col (str): Column name indicating empty concepts (e.g., 'cui').
        name_col (str): Column name for the feature name or query.

    Returns:
        list: List of 'boring' feature names to be excluded from further analysis.
    """
    df = result_df.loc[(result_df['Cooccurrence P-Value'] >= SIGNIFICANT_PVAL) & (result_df["Co-occurrence Count"] >= cooc_count_filter_val)]
    df = df.loc[df['Query Count'] >= min_query_count]
    boring_feature_names_list = list(df[name_col].unique())
    return boring_feature_names_list


def run_search_pubmed(config, df=None, do_path_filter=None, SAVE_OUTPUTS: bool = True):
    """
    Runs the PubMed search and analysis pipeline based on the given configuration.

    Parameters:
        config (dict): Configuration parameters for the pipeline.
        df (pd.DataFrame, optional): DataFrame containing query candidates. If not provided, the function will load the file from 'QUERY_CANDIDATES_FILE'.
        do_path_filter (bool, optional): Whether to apply path length filtering.
        SAVE_OUTPUTS (bool, optional): Whether to save the output results.

    Returns:
        pd.DataFrame: The result dataframe after processing the queries and PubMed search.
    """
    # Set Entrez API email and key
    Entrez.email = config.get('ENTREZ_EMAIL', os.environ.get('ENTREZ_EMAIL'))
    Entrez.api_key = config.get('ENTREZ_API_KEY', os.environ.get('ENTREZ_API_KEY'))
    init_db()

    if do_path_filter is None:
        do_path_filter=config.get('DO_MINI_COMBINED_PATH_FILT', False)

    # Process targets
    targets = config['targets']
    if len(targets) > 1:
        targets = [" OR ".join("({})".format(x) for x in targets)]
    print("Targets:", targets)


    # # Load + process candidate queries
    # df_query = load_candidate_queries(config['QUERY_CANDIDATES_FILE'], do_path_filter=do_path_filter)
    if df is None:
        df_query = load_candidate_queries(config['QUERY_CANDIDATES_FILE'], do_path_filter=do_path_filter)
    else:
        df_query = load_candidate_queries(df=df, do_path_filter=do_path_filter)

    # Add feature names to queries
    df_query = add_fnames_to_queries(df_query)
    print(f"{df_query.shape[0]} Rows, {df_query['feature_name'].nunique()} # unique Feature Names")

    # Get feature name queries
    fname_queries_subset = set(df_query["feature_name"].str.strip().unique())
    print(f"fname_queries_subset:\n{fname_queries_subset}")

    # Collect search data for feature names and targets
    fname_result_df = collect_search_data(fname_queries_subset, targets)

    # Get "boring" feature names, for filtering those out
    boring_feature_names = filter_boring_feature_names_cooc(fname_result_df, cooc_count_filter_val=config.get('cooc_count_filter_val', 10))
    print(f"{len(boring_feature_names)} boring_feature_names\n", boring_feature_names)
    # Exclude boring feature names from queries
    s1 = df_query['feature_name'].nunique()
    df_query = df_query.loc[~df_query["feature_name"].str.strip().isin(boring_feature_names)]
    print(f"{df_query.shape[0]} Rows, {df_query['feature_name'].nunique()} # unique Feature Names",f"{s1-df_query['feature_name'].nunique()} boring features dropped")

    # Prepare queries (could be CUIs or feature names)
    queries = set(df_query.cui_nomenclature.str.strip().unique())
    print(len(queries), "# candidate queries")

    # Collect search data for queries and targets
    result_df = collect_search_data(queries, targets)

    # Filter results with minimal counts (redundant)
    result_df = result_df.loc[result_df[['Query Count', "Target Count"]].min(axis=1) >= 20].reset_index(drop=True)

    # Merge results with original query dataframe
    result_df = df_query.set_index("cui_nomenclature").join(result_df.set_index("Query"), how="right").reset_index()
    result_df = result_df.rename(columns={"cui_nomenclature": "Query"}).drop(columns=[
        'Odds Ratio', 'TUI', "feature_level_min_kg_hits"], errors="ignore")
    result_df.drop_duplicates(subset=["Query", "feature_name", "KG_Hits", 'Co-occurrence Count'], inplace=True)

    # Process results
    punct_pattern = r"[{}]".format(string.punctuation)
    result_df["col1"] = result_df["Query"].str.lower().str.replace(punct_pattern, ' ').str.strip()
    result_df["col1"] = result_df["col1"].str.replace("[^A-Za-z0-9]", '', regex=True).str.replace("[-\(\)]", "").str.replace('\s+', ' ')
    result_df = result_df.sort_values(["col1", "MI"]).drop_duplicates(["col1", "feature_name", "KG_Hits", "Target"])
    result_df["min_feat_kgHit"] = result_df.groupby(["col1"])["KG_Hits"].transform("min")
    result_df["feature_level_sum_cooc_hits"] = result_df.groupby(["col1"])["Co-occurrence Count"].transform(lambda x: sum(x >= 2))
    result_df = result_df.drop(columns=["col1"]).round(3)
    print(result_df.select_dtypes("O").nunique())

    # Get promising results
    pick_result_df = get_promising_results(result_df, SIGNIFICANT_PVAL=config.get('SIGNIFICANT_PVAL'), Drop_Query_Cui_Info=True,
                                           filter_joint_pathlength=do_path_filter)

    print(result_df["feature_name"].nunique(), "# Overall candidate features")
    print(pick_result_df["feature_name"].nunique(), "# Top pick candidate features")

    # Save outputs if required
    # if config.get('SAVE_OUTPUTS', True):
    if SAVE_OUTPUTS:
        result_df.to_csv(f"{config.get('OUTPUT_RES_PREFIX', '')}{config.get('full_results_filename', 'candidates_search_results.csv')}", index=False)
        pick_result_df.to_csv(
            f"{config.get('OUTPUT_RES_PREFIX', '')}{config.get('filtered_results_filename', 'review_interesting_candidates_results.csv')}",
            index=False)
        print(pick_result_df.shape[0])

    # else:
    return result_df



if __name__ == "__main__":
    # run_search_pubmed(example_config)

    import os

    # Example configuration dictionary
    config = {
        # 'ENTREZ_EMAIL': 'your_email@example.com',  # Replace with your email
        # 'ENTREZ_API_KEY': 'your_entrez_api_key',  # Replace with your Entrez API key
        'targets': [
            "Cholelithiasis",
            "Gallstone",
            "Gallbladder disease",
            "cholecystitis",
            "Cholangitis"
            ],
        'QUERY_CANDIDATES_FILE': 'candidate_novel_cuis_chol.csv',
        'DO_MINI_COMBINED_PATH_FILT': True,
        'SAVE_OUTPUTS': False,
        'OUTPUT_RES_PREFIX': 'gallstone_',
        # 'SIGNIFICANT_PVAL': 0.3,
        'full_results_filename': 'candidates_search_results.csv',
        'filtered_results_filename': 'review_interesting_candidates_results.csv',
        # 'cooc_count_filter_val': 10
        }

    # # Ensure the Entrez email and API key are set in the environment or config
    # os.environ['ENTREZ_EMAIL'] = config['ENTREZ_EMAIL']
    # os.environ['ENTREZ_API_KEY'] = config['ENTREZ_API_KEY']

    # Run the search and analysis pipeline
    run_search_pubmed(config)
