#!/usr/bin/env python
# coding: utf-8

# * Let's start wth looking at features of sepsis , from physionet.
#     * https://physionet.org/content/challenge-2019/1.0.0/
#  
# * later, smarter links - https://chat.openai.com/c/c9e96408-87ab-4b0d-b4a0-8f01adccf794  (prompt call)  ;
# * https://chat.openai.com/c/1e374834-ed6b-4bcd-9dc9-fa9de5134450 - early onset...
# 
# * Pubmed evidence+ Relations extractor work (not using CUIs) - relevant esearch? (lacks paper) - https://ailabs.tw/healthcare/extracting-the-most-significant-and-relevant-relational-facts-from-large-scale-biomedical-literature/
#     * scispaCy ORE
#   * Doesnt seem HQ, but is relevant?
# 
# * https://oyewusiwuraola.medium.com/how-to-use-scispacy-entity-linkers-for-biomedical-named-entities-7cf13b29ef67
#     *  Also has examples of using the different scispacy entity type models
# *  https://github.com/WuraolaOyewusi/How-to-use-scispaCy-Entity-Linkers-for-Biomedical-Named-Entities/blob/master/scispacy_entities_extractions_and_linkers(uncleared_outputs)_.ipynb
# 
# *  * Also relevant - Hetionet (in PyKeen, DGL-KE) - finding important paths between 2 nodes. drug repurposing focused initially.
# 
#  
# Install note:
# * scispacy - you need python 3.10 (for nmslib to install ok)
# 
# 
# Alt tool: Metamap
# * https://gweissman.github.io/post/using-metamap-with-python-to-access-the-umls-metathesaurus-a-quick-start-guide/
# * https://github.com/AnthonyMRios/pymetamap
# 
# * Google cloud natural health NLP - supports medical NER etc', including UMLS/"Metatheasauurs" names
#     * https://cloud.google.com/healthcare-api/docs/concepts/nlp#supported_medical_vocabularies
# -------------------------------------------
# 
# UMLS Semantic types (TUI) and groups (could use to filter results):
# * https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/documentation/SemanticTypesAndGroups.html
# 
# 
# 
# **Idea**: Could filter not just by min/max feature level, also by aggregating by feature * __TUI__. ?
# 
# 
# 
# ### Betweenness -  as a continous measure of similarity/ # short paths (instead of binary/discrete cutoff)? 
# * https://docs.google.com/document/d/15jPnznpNyl9CubrmNOsCdABxj3DfKR8epvyzAXqJMbo/edit?_sm_vck=Qn6Rjr36TvSsrHSrFNQMHPtdVHRDnsBfRMBTN5MdJHTjrJnD5WVr#heading=h.bw9q0hr5kp2f
# * Betweenness centrality (networkX) - as a continuous score for similarity (between concepts/feature and target(s)), instead of binary/discrete ?
#     * Edge_betweenness_centrality ?
#     * Edge_betweenness_centrality_subset
# *  networkX - load from pandas ; https://developer.nvidia.com/blog/accelerating-networkx-on-nvidia-gpus-for-high-performance-graph-analytics/  ; nx.from_pandas_edgelist 
# *  Graphtool (faster); https://graph-tool.skewed.de/static/doc/autosummary/graph_tool.centrality.closeness.html 
# * Rustworkx.edge_betweenness_centrality
# * https://www.kaggle.com/code/rahulgoel1106/network-centrality-using-networkx 
# 
# 
# 
# ### Filter idea: 
# * drop duplicates by #KG_hits and sorted sim score +- add in NEL score = keep top hit and that if not in kg. 
# * Could check KG paths, after filtering for relevant predicates, e.g. is_a, precedes, etc' (and check sinple paths on that)

# 
# * How to use scispaCy Entity Linkers for Biomedical Named Entities*
#     * https://oyewusiwuraola.medium.com/how-to-use-scispacy-entity-linkers-for-biomedical-named-entities-7cf13b29ef67
#     * Very relevant! Also uses the other scispacy NER corpuses/models to extract first - e.g. craft, then links those candidates
# 
# `https://towardsdatascience.com/building-a-biomedical-entity-linker-with-llms-d385cb85c15a`
# * LLMs, mistral, compares to scispacy inc NEL
# 
# * https://www.kaggle.com/code/daking/extracting-entities-linked-to-umls-with-scispacy
#     * "One useful thing to play around with here is **filtering the linked entities based on your use case and the UMLS type tree, as types higher up on the tree indicate more general entities**"
# 
# * “Evaluating Explanations from AI Algorithms for Clinical Decision-Making: A Social Science-based Approach” *
# - Very relevant to what I want to do!
# - uses evidence from pubmed KG/semmed! 02.2024.
# - https://www.medrxiv.org/content/10.1101/2024.02.26.24303365v1.full.pdf 
# 

# * Could improve FS with boruta, other methods  - ARFS
# * https://github.com/ThomasBury/arfs/blob/main/docs/notebooks/basic_feature_selection.ipynb

# In[1]:

import os
# Set the environment variable
os.environ['NX_CUGRAPH_AUTOCONFIG'] = 'True'
## make networkx use gpu
## https://developer.nvidia.com/blog/networkx-introduces-zero-code-change-acceleration-using-nvidia-cugraph/

import pandas as pd
import numpy as np
from IPython import get_ipython
from IPython.core.display_functions import display
from tqdm import tqdm  # Import tqdm for the progress bar

## scispacy, medspacy, medcat, quickumls, semrep...
import networkx as nx
from itertools import combinations

import spacy
import scispacy
from itertools import compress
from scispacy.linking import EntityLinker
from scispacy.abbreviation import AbbreviationDetector ## https://github.com/allenai/scispacy?tab=readme-ov-file#example-usage
import sys, os ## append parent path to dir to allow import
parent_directory = os.path.abspath('..')
sys.path.append(parent_directory)


# from util import get_sentence_pairs_similarity,anti_join_df # disable - circular import
from util import *
# from util import get_sentence_pairs_similarity,anti_join_df
import string
rem = string.punctuation
punct_pattern = r"[{}]".format(rem)
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')



def get_kg_connections(df_hits, df_kg, list_target_cuis, get_simplePathLengths = True):
    # ## Graph connectivity metric
    # * Get shortest path (could also get # simplest paths) between source and target, from graph.
    # * We'll calculate that only on pairs lacking a direct link to save compute + other tricks.
    #
    # *    "shortest_path_length": Higher is novel
    # *    "simple_path_length": Lower is novel
    # *    "norm_path_length": Lower is novel\
    #
    #
    # * Additionally: We could try getting the  paths, calculated on a subset of the graph with the densest  = most generic nodes removed. i.e remove "asthma" and so on.
    #
    # * Undirected vs directed graph???
    # In[60]:
    df_kg = df_kg.copy()
    ## make string to suppress replace warning:
    df_kg["SUBJECT_CUI"] = df_kg["SUBJECT_CUI"].astype(str).replace(list_target_cuis, "y")
    df_kg["OBJECT_CUI"] = df_kg["OBJECT_CUI"].astype(str).replace(list_target_cuis, "y")
    # df_kg["SUBJECT_CUI"] = df_kg["SUBJECT_CUI"].replace(list_target_cuis, "y")
    # df_kg["OBJECT_CUI"] = df_kg["OBJECT_CUI"].replace(list_target_cuis, "y")
    G = df_kg[["SUBJECT_CUI", "OBJECT_CUI"]].drop_duplicates().copy()
    assert G["SUBJECT_CUI"].value_counts()["y"] > 5
    assert G["OBJECT_CUI"].value_counts()["y"] > 5
    G.columns = ['source', 'target']
    # Create an undirected graph from the DataFrame:
    G = nx.from_pandas_edgelist(G, 'source', 'target', create_using=nx.Graph())
    ## make directed graph
    # G = nx.from_pandas_edgelist(G, 'source', 'target', create_using=nx.DiGraph())  # what about to-from target vs from-to ??
    ## drop prefilt of distance 0 cases , because we also want their simple paths
    df_temp = df_hits.drop_duplicates(subset=["cui"]).copy()
    # # could skip and later readd all the cases with a known direct link to skip their compute time, if a bottleneck
    # df_known_cuis_list = df_temp.loc[df_temp["KG_Hits"]>0][["cui"]].reset_index(drop=True).copy()
    # df_known_cuis_list["short_path_length"] = 1 # shortest distance in nx - 1 dge
    # df_temp = df_temp.loc[df_temp["KG_Hits"]<1].reset_index(drop=True).copy()
    candidate_cuis_list = df_temp["cui"].unique()
    short_path_length_list = []
    simple_path_length_list = []  ## simple paths part is much slkower!!
    degrees_list = []  # store # out degree of nodes, (could do out degree if using directed graph ) for possible normalizing of simple paths?
    for i, c in tqdm(enumerate(candidate_cuis_list), total=len(candidate_cuis_list), desc="Calculating Paths"):
        if c in G:
            degrees_list.append(G.degree(c))
            try:
                # Calculate shortest path length
                path_length = nx.shortest_path_length(G, source=c, target="y")
                short_path_length_list.append(path_length)
            except nx.NetworkXNoPath:
                # Handle cases where no path exists
                short_path_length_list.append(999)  # Indicate no path (smaller is better)  - float('inf') , or a placeholder large int
            if get_simplePathLengths:
                try:
                    simple_path_length = len(list(nx.all_simple_edge_paths(G, source=c, target="y", cutoff=2)))  # can be slow!!
                    simple_path_length_list.append(simple_path_length)
                except:
                    simple_path_length_list.append(0)
        else:
            # # Handle cases where the candidate node is not in the graph
            # print(f"Node {c} not found in the graph.")
            short_path_length_list.append(99)  # Use  float('inf') None or another marker (0) to indicate missing node
            if get_simplePathLengths:simple_path_length_list.append(0)
            degrees_list.append(0)
    df_path_lengths = pd.DataFrame({"cui": candidate_cuis_list,
                                    "shortest_path_length": short_path_length_list,
                                    # "simple_path_length": simple_path_length_list,
                                    "node_degree": degrees_list
                                    }).sort_values(by=["shortest_path_length"], ascending=True)
    if get_simplePathLengths:
        df_path_lengths["simple_path_length"] = simple_path_length_list
        # lower is more novel
        ## Disable this and it's constituents above
        df_path_lengths["norm_path_length"] = (df_path_lengths["simple_path_length"].div(df_path_lengths["node_degree"]).fillna(0)).round(2)

    df_path_lengths = df_path_lengths.dropna(axis=1,how="all").drop_duplicates(subset=["cui"],keep="first")  # added drop dupes here and sorting

    return df_path_lengths

## TODO: Add 2d degree distance linkage? +- Sem dist min filter
def temporal_kg_concepts_eval(df_hits,df_kg_sep,KG_YEAR_CUTOFF = 2012):
    """
    Foward Temporal eval
    ## Check which links/features from our data appear in "future" data
    * Take KG, split to train/test by time (e.g. 2014 onwards), if a link appears in 2014 onwads and not in earlier , then it is a possible case of a novel detection from our model.
    * Need to consider how to filter cuis/features. e.g. overly common cui nomenclatures

    :param df_hits:
    :param df_kg_sep:
    :param KG_YEAR_CUTOFF:
    :return:
    """


    print(KG_YEAR_CUTOFF,"KG_YEAR_CUTOFF")


    df_hits["cui_count"] = df_hits.groupby(["cui"]).transform("size")


    print(df_kg_sep.nunique())


    # In[74]:


    df_kg_past = df_kg_sep.loc[df_kg_sep["first_year_pair"]<KG_YEAR_CUTOFF]
    print("past",df_kg_past.shape[0])
    df_kg_future = df_kg_sep.loc[df_kg_sep["first_year_pair"]>=KG_YEAR_CUTOFF]
    print("Future, before filter/antijoin",df_kg_future.shape[0])


    # In[75]:


    df_kg_future = anti_join_df(left=df_kg_future, right=df_kg_past, key=["SUBJECT_CUI","OBJECT_CUI"])

    # In[76]:


    past_cuis_list = list(set(df_kg_past["SUBJECT_CUI"].unique().tolist() + df_kg_past["OBJECT_CUI"].unique().tolist()))
    print("# past cuis",len(past_cuis_list))

    future_cuis_list = list(set(df_kg_future["SUBJECT_CUI"].unique().tolist() + df_kg_future["OBJECT_CUI"].unique().tolist()))
    print("# future cuis",len(future_cuis_list))
    future_cuis_list = [x for x in future_cuis_list if x not in past_cuis_list]
    print("# future cuis, not in past",len(future_cuis_list))


    # ## Keep only (model) features that appear in the future, not past , of the KG
    #
    # * Additional possible filter needed: If any cui (perfeature) is a match in PAST data?
    #     * Would need to filter out overly common cuis (e.g. "genetic risk") in such a case or multiples...
    # * Q: Important filter note: Might want to filter by first appearance/year of a specific cui? e.g. to avoid cases where cui was only added after cutoff. This is arguable, as a novelty might legitiamtely be added as a cui after cutoff..

    # In[77]:


    df_hits_future = df_hits.loc[df_hits["cui"].isin(future_cuis_list)].copy()
    print(df_hits_future.shape[0], "# Rows")
    print("nunique:")
    # print(df_hits_future.groupby(["SUBJECT_CUI","OBJECT_CUI"]).size(), "# Subj X Obj cui unique Pairs")
    print(df_hits_future.nunique())
    print("\nfeature #:")
    print(df_hits_future.feature_name.value_counts().head(11))
    print("\ncui #:")
    print(df_hits_future.cui_nomenclature.value_counts().head(11))


    # ### Get feature level (vs cui) level matched hits in past and then use that for filtering
    # * Will take higher confidence cuis/terms (to reduce noise)
    # * We remove very common cuis that are too broad (e.g. "gene risk"). Set at 97% quantile (5 in our case for gallstones)

    # In[78]:


    df_hits_past = df_hits.loc[df_hits["cui"].isin(past_cuis_list)].copy()
    print(df_hits_past.shape[0],"# rows\n")
    print(df_hits_past.select_dtypes(["O","category"]).nunique(),"\n")

    ## redo cui count to use past data, avoid future information leak. Note: distribution looks about the same.
    df_hits_past["cui_count"] = df_hits_past.groupby(["cui"]).transform("size")

    # print(df_hits_past["cui_count"].describe(percentiles=[.5,.9,.97]).round(1))
    display(df_hits_past.drop_duplicates(["cui"])["cui_count"].describe(percentiles=[.5,.9,.95,.97]).round(1)) # avoid double counting cuis
    past_cui_count_cutoff = max(df_hits_past.drop_duplicates(["cui"])["cui_count"].quantile(0.97),2)
    print("cui cutoff for past:",past_cui_count_cutoff)
    df_hits_past = df_hits_past.loc[df_hits_past["cui_count"]<past_cui_count_cutoff]
    print("\nAfter cui cutoff filter:\n")
    print(df_hits_past.shape[0],"# rows\n")
    print(df_hits_past.select_dtypes(["O","category"]).nunique(),"\n")

    # ### Get results - # features in future and not in past, out of all utility featues
    # * Additional filter - rmeoval of "common" cui terms from the future results also (not just past).

    # In[80]:

    print(f'Out of {df_hits.query("KG_Hits>0")["feature_name"].nunique()} Features with any hit in KG')
    # print(df_hits_future["feature_name"].nunique(),"# future feats pre past filt")
    df_hits_future = df_hits_future.loc[(~df_hits_future["cui"].isin(df_hits_past["cui"])) & (~df_hits_future["feature_name"].isin(df_hits_past["feature_name"]))]
    print(df_hits_future["feature_name"].nunique(),"# future feats after past filt")


    # * Filter out common cuis like with past
    # * Use overall count instead of future only count of those cuis
    # * This filtering may be excessive?

    # In[81]:

    df_hits_future = df_hits_future.loc[df_hits_future["cui_count"]<=past_cui_count_cutoff]
    print(df_hits_future.drop_duplicates("feature_name").shape[0])
    display(df_hits_future.drop_duplicates("feature_name"))


# In[69]:
if __name__ == "__main__":

    # ### EXAMPLE CONFIGS FOR link_kg_concepts():
    # # ## GALLSTONES
    # FEATURES_REPORT_PATH = "../gallstone_chol_ipw_feature_report.csv"
    # ## output path
    # CANDIDATE_NOVEL_CUIS_FILEPATH = "../candidate_novel_cuis_chol.csv"
    # TARGET_NAME = "GALLSTONES, Cholelithiasis"
    # additional_target_cui_terms_list = ["C0008325", "C0008311"]  # cholecystitis, Cholangitis - for gallstones

    # ####### gout:
    # ## input:
    # FEATURES_REPORT_PATH = "../gout_ipw_feature_report.csv"
    # CANDIDATE_NOVEL_CUIS_FILEPATH = "../candidate_novel_cuis_gout.csv"
    # TARGET_NAME = "Gout"
    # additional_target_cui_terms_list = []
    # # ###### Celiac:
    # ## input:
    # FEATURES_REPORT_PATH = "../celiac_feature_report.csv"
    # ## output path
    # CANDIDATE_NOVEL_CUIS_FILEPATH = "../candidate_novel_celiac.csv"
    # TARGET_NAME = "Coeliac disease" # / Celiac
    # additional_target_cui_terms_list = ["C5139492"] # gluten allergy
    # # ######### Multiple Sclerosis (MS)
    # FEATURES_REPORT_PATH = "../MS_ipw_feature_report.csv"
    # CANDIDATE_NOVEL_CUIS_FILEPATH = "../candidate_novel_MS.csv"
    # TARGET_NAME = "Multiple Sclerosis"
    # additional_target_cui_terms_list = []
    # # ######### Spine Degeneration
    # FEATURES_REPORT_PATH = "../spine_degen_feature_report.csv"
    # CANDIDATE_NOVEL_CUIS_FILEPATH = "../candidate_novel_spine.csv"
    # TARGET_NAME = "Spine degeneration"
    # additional_target_cui_terms_list = ["C0158266", "C0850918" , "C0021818", "C0038019", "C0158252","C0423673","C0024031"]
    # # ######### Oesophagus cancer
    # FEATURES_REPORT_PATH = "../oesophagus_feature_report.csv"
    # CANDIDATE_NOVEL_CUIS_FILEPATH = "../candidate_novel_oesophagus.csv"
    # TARGET_NAME = "Esophageal cancer"
    # additional_target_cui_terms_list = []

    # sample config for Gout. Could use DataClasses instead, but this is easier for new users..

    ### gout
    # config = {"FEATURES_REPORT_PATH":"../../gout_ipw_feature_report.csv",
    #           "CANDIDATE_NOVEL_CUIS_FILEPATH":"../../candidate_novel_cuis_gout.csv",
    #           "TARGET_NAME": "Gout"}

    ## heart attack
    # config = {"FEATURES_REPORT_PATH":"../../heart_feature_report.csv",
    #           "CANDIDATE_NOVEL_CUIS_FILEPATH":"../../candidate_novel_cuis_heart.csv",
    #           "TARGET_NAME": "Heart attack"}

    ## gallstones / Cholelithiasis
    config = {
      #   "FEATURES_REPORT_PATH":"../../gallstone_chol_ipw_feature_report.csv",
      # "CANDIDATE_NOVEL_CUIS_FILEPATH":"../../candidate_novel_cuis_chol.csv",
        "FEATURES_REPORT_PATH": "gallstone_chol_ipw_feature_report.csv",
        "CANDIDATE_NOVEL_CUIS_FILEPATH": "candidate_novel_cuis_chol.csv", # output path
            "TARGET_NAME" :"GALLSTONES, Cholelithiasis",
            "additional_target_cui_terms_list" : ["C0008325", "C0008311"] }


    # ## BROAD config, for chol
    #  sem_similarity_threshhold_score = 0.04, top_cutoff_simMin=0.65, MIN_EVIDENCE_FILTER=3,
    # config = {
    #     "FEATURES_REPORT_PATH":"../../gallstone_ipw_broad_feature_report.csv",
    #           "CANDIDATE_NOVEL_CUIS_FILEPATH":"../../broad_candidate_novel_cuis_chol.csv",
    #         "TARGET_NAME" :"GALLSTONES, Cholelithiasis",
    #         "additional_target_cui_terms_list" : ["C0008325", "C0008311"] }


    # #  Retinal Vein Occlusion. (Central retinal artery occlusion: H34.1 )
    ## Note: Central retinal artery occlusion is a type of stroke and must be treated immediately.
    # config = {"FEATURES_REPORT_PATH":"../../eye_occ_ipw_feature_report.csv",
    #               "CANDIDATE_NOVEL_CUIS_FILEPATH":"../../candidate_novel_cuis_eye_occ.csv",
    # "TARGET_NAME":"Retinal Vein Occlusion"
    # # "additional_target_cui_terms_list" : []
    # }

    link_kg_concepts(FEATURES_REPORT_PATH=config["FEATURES_REPORT_PATH"], CANDIDATE_NOVEL_CUIS_FILEPATH=config["CANDIDATE_NOVEL_CUIS_FILEPATH"],
                     TARGET_NAME=config["TARGET_NAME"],
                     SAVE_OUTPUTS = False,
                    #  sem_similarity_threshhold_score = 0.04, top_cutoff_simMin=0.65, MIN_EVIDENCE_FILTER=3,
                     )
