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


## move to util
# def link_kg_concepts(FEATURES_REPORT_PATH, CANDIDATE_NOVEL_CUIS_FILEPATH, TARGET_NAME, additional_target_cui_terms_list=[], SAVE_OUTPUTS = True, MIN_EVIDENCE_FILTER = 2,
#                      REMOVE_CUI_TERMS_LIST=['Prieto syndrome', "Polarized Reflectance Spectroscopy",
#                                             # mistaken extraction from PRS - drop it for now for cleanliness
#                                             'Standard (qualifier)', 'Standard base excess calculation technique',
#                                             'Standard of Care', 'Spatial Frequency', 'Disease',
#                                             'Statistical Frequency', 'Kind of quantity - Frequency', 'Concentration measurement',
#                                             'Concentration measurement', 'Illness (finding)', 'Concentration Ratio',
#                                             "Special", "ActInformationPrivacyReason <operations>", "Left sided", "Left", "Right",
#                                             "Table Cell Horizontal Align - left", "Query Quantity Unit - Records", "Up",
#                                             "Qualification",
#                                             "Visit", "Total", "Participant", "Overall", "Right sided", "Left sided",
#                                             "Take", "Percent (qualifier value)","Population Group",
#                                             "Diagnosis","Coding","Code",
#                                             "Average" , "Comparison" , "Lost" , "Yes - Presence findings", 
#                                             "Pharmaceutical Preparations","Physicians","Mother (person)","Father (person)","Severe (severity modifier)",
#                                             ]
#                      , input_kg_path="../../SemMed/predications.parquet", EXCLUDE_TUIS_LIST = ["T079", "T093", "T094", "T095", "T170", "T204", "T201", "T065",
#                          "T078", ], sem_similarity_threshhold_score=0.15, # 0.15
#                      top_cutoff_simMin = 0.39,top_cutoff_kgHit = 2):

#     # global nlp, df_kg_sep, df_hits, G
#     global df_kg_sep, df_hits ## maybe disable this..

#     # ### Load processed semmed db
#     # * ths version has 1 row per triple.
#     # * `Count` is the number of unique PMIDs the triple has appeared in
#     # * Idea: we could filter for triples that appear at least K times. +- filter papers with less than 1-2 citations
#     # In[5]:
#     df_kg = pd.read_parquet(input_kg_path)

#     # ### Filtered version
#     # * Could Keep cases with more than ~3 evidences (note: counts of evidences are counts of unique papers with that SVO triple).
#     # * * could laso filter by HQ papers (With external dataset linked to the PMIDs in raw data - e.g. using pubmedKG for citation counts)
#     # In[8]:
#     # print(df_kg.shape[0])  # 26M
#     # df_kg["counts"] = df_kg[["SUBJECT_CUI","OBJECT_CUI","PREDICATE"]].groupby(["SUBJECT_CUI","OBJECT_CUI"],observed=True)["PREDICATE"].transform("size") # count evidence irregardless of predicate
#     df_kg = df_kg.loc[df_kg["pair_counts"] >= MIN_EVIDENCE_FILTER].reset_index(drop=True).copy()
#     df_kg.drop(columns=["pair_counts","counts"], inplace=True, errors="ignore")
#     print("After filtering KG min count")
#     for c in df_kg.select_dtypes("category").columns:
#         # remove unobserved categories, in new filtered data
#         df_kg[c] = df_kg[c].cat.remove_unused_categories()
#     print(df_kg.shape[0])  # 4M

#     # # !pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz
#     # # !pip install 'spacy[transformers]'
#     # # !python -m spacy download en_core_web_sm
#     # # !pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz ## small
#     # ```
#     # In[11]:
#     # %%time
#     # nlp = spacy.load("en_core_sci_sm")
#     # there's also lg, and transformer based
#     nlp = spacy.load("en_core_sci_lg")
#     # nlp = spacy.load("en_core_sci_scibert")
#     # In[12]:
#     # This line takes a while, because we have to download ~1GB of data
#     # and load a large JSON file (the knowledge base). Be patient!
#     # Thankfully it should be faster after the first time you use it, because
#     # the downloads are cached.
#     # NOTE: The resolve_abbreviations parameter is optional, and requires that
#     # the AbbreviationDetector pipe has already been added to the pipeline. Adding
#     # the AbbreviationDetector pipe and setting resolve_abbreviations to True means
#     # that linking will only be performed on the long form of abbreviations.

#     # Add the abbreviation pipe to the spacy pipeline. (if using resolve_abbreviations )
#     nlp.add_pipe("abbreviation_detector")

#     nlp.add_pipe("scispacy_linker",
#                  config={"resolve_abbreviations": True,
#                          "linker_name": "umls",
#                          "max_entities_per_mention": 3  # 3, #6, #4, #5
#                      , "threshold": 0.88  ## default is 0.8, paper mentions 0.99 as thresh
#                          })


#     def test_nel(text: str = "Sepsis"):
#         doc = nlp(text)

#         # Let's look at a random entity!
#         print("All ents", doc.ents)
#         ## broken code snippet : https://github.com/allenai/scispacy/issues/355
#         for e in doc.ents:
#             if e._.kb_ents:
#                 cui = e._.kb_ents[0][0]
#                 print(e, cui)

#         print("\n--------------------------\n")
#         entity = doc.ents[0]
#         print("Name: ", entity)

#         # Each entity is linked to UMLS with a score
#         # (currently just char-3gram matching).
#         linker = nlp.get_pipe("scispacy_linker")
#         for umls_ent in entity._.kb_ents:
#             print(linker.kb.cui_to_entity[umls_ent[0]])

#         # linker.kb.cui_to_entity[umls_ent[0]][3][0] # TUI
#         # linker.kb.cui_to_entity[umls_ent[0]][4] # definition
#         # linker.kb.cui_to_entity[umls_ent[0]][1] # cui-name
#         # linker.kb.cui_to_entity[umls_ent[0]][2] # aliases
#         return entity

#         # >>> CUI: C1839259, Name: Bulbo-Spinal Atrophy, X-Linked
#         # >>> Definition: An X-linked recessive form of spinal muscular atrophy. It is due to a mutation of the
#         #   				gene encoding the ANDROGEN RECEPTOR.
#         # >>> TUI(s): T047
#         # >>> Aliases (abbreviated, total: 50):
#         #          Bulbo-Spinal Atrophy, X-Linked, Bulbo-Spinal Atrophy, X-Linked, ....

#     # In[14]:
#     # test_nel("gluten allergy")
#     # In[15]:
#     print("TARGET_NAME (For Entity-KG linking", TARGET_NAME)
#     # entity= test_nel("GALLSTONES, Cholelithiasis") # "Cholelithiasis" = Gallstone
#     # entity= test_nel("Gout") # GOUT
#     # C0007570, Name: Celiac Disease
#     entity = test_nel(TARGET_NAME)
#     # #### Target terms - may manually change
#     ## list of extracted CUIs meaning sepsis. Note that not all correct even here and with threshhold
#     list_target_cuis = [i[0] for i in entity._.kb_ents]
#     if len(additional_target_cui_terms_list) > 0:
#         list_target_cuis = list(set(list_target_cuis + additional_target_cui_terms_list))
#     print("list_target_cuis",list_target_cuis)
#     # #### get subset of KG with target in pairs
#     # (not 100% sure if ideal, but will save time when comparing features )
#     # In[26]:
#     # sepsis_cui = 'C0518988'
#     # df_kg_sep = df_kg.loc[(df_kg["SUBJECT_CUI"]==sepsis_cui) | (df_kg["OBJECT_CUI"]==sepsis_cui)].copy()
#     df_kg_sep = df_kg.loc[(df_kg["SUBJECT_CUI"].isin(list_target_cuis)) | (df_kg["OBJECT_CUI"].isin(list_target_cuis))].copy()
#     df_kg_sep.drop_duplicates(['SUBJECT_CUI', 'SUBJECT_NAME', 'OBJECT_CUI', 'OBJECT_NAME'],
#                               inplace=True)  # ignore predicate type for this filter table
#     # In[28]:
#     ## sample set of terms from gallstone prediction + filtered
#     icu_feature_terms = pd.read_csv(FEATURES_REPORT_PATH)
#     icu_feature_terms = icu_feature_terms.loc[(icu_feature_terms["feature_importance"] > 0) |(icu_feature_terms["MutualInfoTarget"] >= 0.001)|(icu_feature_terms["p_val"] <= 0.05) ]  # filter a bit - optionally
#     assert "raw_name" in icu_feature_terms.columns
#     # icu_feature_terms["raw_name"] = icu_feature_terms["name"]
#     icu_feature_terms["name"] = icu_feature_terms["name"].str.replace("_nan", " ").str.replace("_", " ",
#                                                                                                regex=False)  # .str.replace("."," ",regex=False).str.strip()
#     icu_feature_terms["name"] = icu_feature_terms["name"].str.replace("missing ", "", case=False).str.replace("(", " (", regex=False).str.replace(
#         "  ", " ", regex=False).str.replace("_Empty", "", regex=False).str.strip()
#     ## following filters may increase noise? or may help with NEL? Note it means different featre shape
#     icu_feature_terms["name"] = icu_feature_terms["name"].str.replace("left|right|Major|missing|Array [0-9]|Standard ", "", case=True,
#                                                                       regex=True).str.replace("()", " ", regex=False).str.replace("  ", " ",
#                                                                                                                                   regex=False).str.strip()
#     icu_feature_terms["name"] = icu_feature_terms["name"].str.replace(" None$", "", regex=True,
#                                                                       case=False)  ## remove some cases of "none" at end of feat.
#     icu_feature_terms["name"] = icu_feature_terms["name"].str.replace("PRS", "Genetic risk").str.replace("  ", " ",
#                                                                                                          regex=False).str.strip()  # PRS gets lots of noise.
#     icu_feature_terms["name"] = icu_feature_terms["name"].str.replace(" [0-9]{1,5}$", "", regex=True).str.strip()  # remove number at end
#     icu_feature_terms["name"] = icu_feature_terms["name"].str.replace(" No$|Yes$|Do not know$", "",
#                                                                       regex=True).str.strip()  # some noise when searching maybe
#     icu_feature_terms["name"] = icu_feature_terms["name"].str.replace("Treatment/medication code |", "medication", regex=False)
#     icu_feature_terms["name"] = icu_feature_terms["name"].str.replace("Non-cancer illness code, self-reported | ", "", regex=False).str.strip()
#     # .str.replace(" gene","")
#     # df_icu_feature_terms = icu_feature_terms.copy() # raw vals
#     ## warning - feature values ma y be wrong for cases of "missing" placeholder featues" (will have same name
#     icu_feature_terms = icu_feature_terms.drop_duplicates(subset=["name"]).reset_index(drop=True)  # some dupe terms, e.g. due to missing being removed

#     # * ToDos - get scores per CUI of a name, and aggregate - so we can later tell if a feature/name has just 1 or multiple non-novel CUIs...
#     # * could store using dict instead of lists.
#     # In[29]:
#     # EXCLUDE_TUIS_LIST = ["T079", "T093", "T094", "T095", "T170", "T204", "T201", "T065",
#     #                      "T078", ]  # List of umls cui semtypes to exclude. Rough heuristic - not validated!
#     ## more bad cases
#     # "missing" -
#     #         # CUI: C1551393, Name: Container status -  ;  TUI(s): T033
#     #         # CUI: C1705492, Name: Missing Definition: Not existing - TUI(s): T080 # Legitimate use of T80
#     ## T078	Idea or Concept  - e.g. Standard
#     ## T080	Qualitative Concept - e.g. Standard (qualifier). Brown.  - Borderline , maybe drop?
#     ### exogenous -> T082	Spatial Concept , T169	Functional Concept
#     ## get CUIs from all entites in text - may add too much noise?
#     ### It is easier and less work to just filter the output dataframe - although this may harm quality ofresults returned..
#     ## "T204" - vertebrate, invertebrate
#     ## "T201" - point in time, etc'. (Many of these findings are covered by normal diagnoses, e.g. ## "T204" -
#     ## "T065" - Educational process of instructing , Teaching aspects
#     # doc =nlp(icu_feature_terms)
#     novel_cols_candidates_names = []
#     no_entities_list = []
#     novel_candidate_cuis = []
#     novel_candidate_cuis_nomenclatures = []
#     TUIs_list = []
#     list_cui_kg_hits = []  # mark if (CUI level) novel or not (presencei n KG graph, for given target). Save # of hits (will alow filtering?)
#     list_cui_definitions = []  # all text inc synonyms, definition for each cui - convenience for  doing expanded semantic similarity filtering.
#     for f in icu_feature_terms["name"]:
#         doc = nlp(f)
#         linker = nlp.get_pipe("scispacy_linker")
#         ## could use all or first k entities? Is this even top entity?
#         if len(doc.ents) > 0:
#             for j, entity in enumerate(doc.ents):
#                 #         if linker.kb.cui_to_entity[umls_ent[0]][3][0] not in EXCLUDE_TUIS_LIST: ## filter entities by TUIs, don't count the excluded. May overfilter!
#                 # TUIs_list.append(linker.kb.cui_to_entity[umls_ent[0]][3][0]) # new
#                 ### entity = doc.ents[0] # only get first entity
#                 # print(f"Entity #{j}:{entity}")

#                 list_feature_cuis = [i[0] for i in entity._.kb_ents]
#                 # print(list_feature_cuis)

#                 ## add tui filt
#                 s1 = len(list_feature_cuis)
#                 tui_filter_mask = [linker.kb.cui_to_entity[c][3][0] not in EXCLUDE_TUIS_LIST for c in list_feature_cuis]
#                 list_feature_cuis = list(compress(list_feature_cuis, tui_filter_mask))
#                 # print(list_feature_cuis)
#                 list_cuis_nomenclatures = [linker.kb.cui_to_entity[i[0]][1] for i in entity._.kb_ents]
#                 # linker = nlp.get_pipe("scispacy_linker") #ORIG
#                 list_cuis_nomenclatures = list(compress(list_cuis_nomenclatures, tui_filter_mask))

#                 num_candidates = len(list_feature_cuis)
#                 for c in list_feature_cuis:
#                     TUIs_list.append(linker.kb.cui_to_entity[c][3][0])  # c[0]][3][0])
#                 if num_candidates > 0:
#                     for umls_ent in entity._.kb_ents:
#                         ent_name = linker.kb.cui_to_entity[umls_ent[0]][1]  # remove [1] to print all the cui data
#                         # if ent_name not in novel_candidate_cuis_nomenclatures:
#                         #     print(ent_name)
#                     df_related = df_kg_sep.loc[(df_kg_sep["SUBJECT_CUI"].isin(list_feature_cuis)) | (df_kg_sep["OBJECT_CUI"].isin(list_feature_cuis))]

#                     for cui in list_feature_cuis:
#                         num_kg_hits = df_related.loc[(df_kg_sep["SUBJECT_CUI"] == cui) | (df_related["OBJECT_CUI"] == cui)].shape[0]
#                         list_cui_kg_hits.append(num_kg_hits)
#                         # list_cui_definitions.append(linker.kb.cui_to_entity[cui][4]) # cui definition only
#                         list_cui_definitions.append(str(linker.kb.cui_to_entity[cui][1]) + ". " + str(linker.kb.cui_to_entity[cui][4]).replace("None",
#                                                                                                                                                ""))  # append all cui definition, synonms, tui etc'

#                     novel_cols_candidates_names.extend([f] * (num_candidates))
#                     novel_candidate_cuis.extend(list_feature_cuis)
#                     novel_candidate_cuis_nomenclatures.extend(list_cuis_nomenclatures)
#                 ## orig level of no wntity func:
#                 # else:
#                 #     no_entities_list.append(f)
#                 #     print(f"No Entity candidates for {f}")
#                 else:  # new, alt level, for 0 cands after filt
#                     no_entities_list.append(f)
#                     # print(f"No Entity candidates for {f}")
#                     # novel_cols_candidates_names.append([f])
#                     # novel_candidate_cuis.append([""])
#                     # novel_candidate_cuis_nomenclatures.append([""])
#         # print("-------------------------------")
#         assert len(novel_cols_candidates_names) == len(novel_candidate_cuis)
#         no_entities_list = list(set(no_entities_list))
#     print(f"{len(no_entities_list)} - No Entity feats:{no_entities_list}")

#     df_hits = pd.DataFrame({"feature_name": novel_cols_candidates_names,
#                             "cui": novel_candidate_cuis  # + list_target_cuis,
#                                , "cui_nomenclature": novel_candidate_cuis_nomenclatures,
#                             "cui_def": list_cui_definitions,
#                             "KG_Hits": list_cui_kg_hits,
#                             "TUI": TUIs_list}).drop_duplicates()  ##
#     ## merge with icu_feature_terms[["raw_name","name"]]
#     s1 = df_hits.shape[0]
#     df_hits = df_hits.loc[~df_hits["cui_nomenclature"].isin(REMOVE_CUI_TERMS_LIST)]
#     print(s1 - df_hits.shape[0], "Rows of unwanted cuis dropped")
#     for c in list_target_cuis:  ## manually append it hewre with some of the feature vals
#         df_hits._append({"cui": c, "feature_name": "target", "cui_nomenclature": linker.kb.cui_to_entity[c][1]}, ignore_index=True)
#     s1 = df_hits.shape[0]
#     print(s1, "# rows pre semantic sim filt")
#     ### semantic similarity - heuristic, remove poor scoring pairs (by semantic similarity).
#     ### Note:Could also expandthis with the CUIsdescriptions
#     df_hits = get_sentence_pairs_similarity(df=df_hits, col1="cui_nomenclature", col2="feature_name", filter=True, minFilterValue=0.98 * sem_similarity_threshhold_score
#                                             , model2Name=None)
#     # print(s1 - df_hits.shape[0], "rows dropped by cui/feature semantic sim")
#     s1 = df_hits.shape[0]
#     df_hits = get_sentence_pairs_similarity(df=df_hits, col1="cui_def", col2="feature_name", model2Name=None, filter=True,
#                                             minFilterValue=sem_similarity_threshhold_score)  # 0.07
#     print(s1 - df_hits.shape[0], "rows dropped by cui+Definition/feature semantic sim")
#     ### TODO/DONE: Could do the mix, max, etc' using more stringently filtered (by sim score) concepts?
#     ### Make pseudo col, instead of groupby join and needing more code. If decide not to use, then switch kg_hits_robust back to KG_Hits in subsequent code
#     df_hits["kg_hits_robust"] = np.where(df_hits["sim_score"] >= 0.24, df_hits["KG_Hits"], 0)
#     df_hits["feature_level_min_kg_hits"] = df_hits.groupby(["feature_name"])["kg_hits_robust"].transform("min")
#     # df_hits["feature_level_sum_kg_hits"] = df_hits.groupby(["feature_name"])["KG_Hits"].transform("sum") # max
#     # error - wrong length (groupedby, not transform(# df_hits["feature_level_sum_kg_hits"] = df_hits.groupby(["feature_name"]).apply(lambda df: sum(df.KG_Hits > 0)).values # sum of cases with hits
#     df_hits["feature_level_sum_kg_hits"] = df_hits.groupby(["feature_name"])["kg_hits_robust"].transform(lambda x: sum(x > 0))
#     # df_hits["feature_level_mean_kg_hits"] = df_hits.groupby(["feature_name"])["KG_Hits"].transform(lambda x: mean(x>0))  # todo: make work
#     df_hits["v"] = df_hits["kg_hits_robust"].clip(upper=1)
#     df_hits["feature_level_avg_kg_hits"] = df_hits.groupby(["feature_name"])["v"].transform("mean").round(1)
#     df_hits.drop(columns=["v", "kg_hits_robust"], errors="ignore", inplace=True)
#     # ## cases matching existing literature knowledge :
#     # display(df_hits.query("KG_Hits>0").drop_duplicates("cui_nomenclature"))
#     "keep features where at least 1 potentially novel cui = unmatched in known literature-KG:"
#     # df_hits = df_hits.query("feature_level_min_kg_hits==0 & feature_level_avg_kg_hits<0.7")
#     df_hits["cui"] = df_hits["cui"].astype(str)
#     df_hits.drop_duplicates(inplace=True)
#     print(df_hits[["feature_name", "cui"]].nunique())
#     print("# KG Hits:")
#     print(df_hits.query("KG_Hits>0")[["feature_name", "cui"]].nunique())
#     print("# No KG Hits for feature:")
#     print(df_hits.query("feature_level_min_kg_hits==0")[["feature_name", "cui"]].nunique())
#     # ### optional another filter step - drop by top match being found?
#     # * This may not help wit hcases of irrelevant matches.
#     # * OPT/dangeorus
#     # * IDEA: Take top match (by ner or our sim score) per entity, and if that is confident and a known link , then drop the feature.
#     #     * could change KG_hits to >1 instead of >0 ?
#     # In[35]:
#     df_hits_top = df_hits.sort_values(["feature_name", "sim_score"], ascending=False).copy()  # sort with highest similarity feature first
#     ## Also check for cases of rough exact match (cui =~ feature name, after mini cleaning
#     df_hits_top["clean_featName"] = df_hits_top["feature_name"].str.lower().str.replace(punct_pattern, '').str.strip()
#     df_hits_top["clean_cui"] = df_hits_top["cui_nomenclature"].str.lower().str.replace(punct_pattern, '').str.strip()

#     # df_hits_top = df_hits_top.query(f"sim_score>={top_cutoff_simMin} & KG_Hits>={top_cutoff_kgHit}").drop_duplicates("feature_name", keep="first")
#     df_hits_top = df_hits_top.loc[((df_hits_top["sim_score"]>=top_cutoff_simMin)\
#                                    |(df_hits_top["clean_featName"]==df_hits_top["clean_cui"]))\
#                                   & (df_hits_top["KG_Hits"]>=top_cutoff_kgHit)].drop_duplicates("feature_name", keep="first")

#     ## drop these cases from candidates, as they are high confidence and seemingly known in lit!
#     print(df_hits["feature_name"].nunique(), "# Feats before @1 filter")
#     df_hits = df_hits.loc[~df_hits["feature_name"].isin(df_hits_top["feature_name"])]
#     print(df_hits["feature_name"].nunique(), " Feats left after top 1 filter")
#     ###### Add/Keep features that had 0 hits in the KG as additional candidate novels , for next stage of filtering
#     # * Add pseudovals for cui
#     # In[36]:
#     ## TODO: Add in raw_name
#     df_hits = pd.concat([df_hits, pd.DataFrame({"feature_name": no_entities_list,
#                                                 "KG_Hits": [0] * len(no_entities_list),
#                                                 "cui_nomenclature": [""] * len(no_entities_list),
#                                                 "cui_def": [""] * len(no_entities_list),
#                                                 "cui": [""] * len(no_entities_list),
#                                                 "sim_score": [1] * len(no_entities_list)})], ignore_index=True)
#     for c in df_hits.select_dtypes("number").columns:
#         # print(c)
#         df_hits[c] = df_hits[c].fillna(0)


#     # #### Rejoin with features metadata
#     # * * Warning: missing feature proxies won't be idd correctly may replace the version of column without missings.
#     #
#     # In[42]:

#     # In[43]:
#     df_hits = df_hits.merge(icu_feature_terms[['name', 'feature_importance', 'p_val', 'corr', "MutualInfoTarget",
#                                                'raw_name', # restore
#                                                'F.Split-Lift (y==1)',
#                                                'F.Split-Support',  # 'F.Split-Target % Covered',
#                                                'F.Split-Feature Split',
#                                                ]].round(3),
#                             left_on=["feature_name"], right_on="name", how="left", validate="m:1").drop(columns=["name", "TUI"], errors="ignore")
#     # In[44]:
#     # df_hits.loc[df_hits["sim_score"]<0.18].drop_duplicates("sim_score").sort_values("feature_name")#.head(12) sim_score

#     # ### Very common/reoccurring nomenclatures - may be too broad
#     # * could remove these based on counts, tf-idf, percentile distribution.
#     # In[45]:
#     df_hits["cui_nomenclature"].value_counts().head(11)  # .index ## TFIDF - would be a good filter!
#     # In[46]:
#     # df_hits["cui_nomenclature"].value_counts().div(df_hits["feature_name"].nunique()).round(3)
#     # # In[47]:
#     # df_hits["cui_nomenclature"].value_counts().describe().round(2)


#     # linker.kb.cui_to_entity[umls_ent[0]][3][0] # TUI
#     # # linker.kb.cui_to_entity[umls_ent[0]][4] # definition
#     # linker.kb.cui_to_entity[umls_ent[0]][1] # cui-name
#     # # linker.kb.cui_to_entity[umls_ent[0]][2] # aliases
#     # In[53]:
#     # In[54]:
#     # df_hits.loc[df_hits["cui"] == "C0028754"]  # "obesity"
#     # In[55]:
#     # df_hits.loc[(df_hits["KG_Hits"] == 0) & (df_hits["feature_level_avg_kg_hits"] < 0.5)].drop_duplicates("cui_nomenclature") \
#     #     .sort_values(["feature_level_avg_kg_hits", "feature_name"]) \
#     #     .drop_duplicates("feature_name")

#     print("features with no linked entities in them:\n",no_entities_list)
#     print(len(novel_cols_candidates_names), "# novel candidate cols")
#     # print(f"{100*(len(novel_cols_candidates_names)/len(icu_feature_terms)):.2f}% candidates novel")
#     # for gallstones - 52% (78) when using TF linker, vs 66% (100) using statistical linker
#     # In[58]:
#     print("novel candidates # CUIS:", len(novel_candidate_cuis))


#     # #### Add seperate sim score between feature name (+- cui?) and the TARGET
#     # * NOTE! This differs from the OTHER sim_score (which was used for filtering NEL results); this one is more for further
#     # In[38]:
#     df_temp = df_hits[["feature_name", "cui_nomenclature"]].copy()  # .head(10) # .drop_duplicates(subset=["feature_name"])
#     df_temp["target_name"] = TARGET_NAME
#     df_hits["sim_score_target_feat"] = get_sentence_pairs_similarity(df=df_temp.copy(), col1="target_name", col2="feature_name", filter=False,
#                                                                      return_score_only=True)
#     # df_hits["sim_score_target_cui"] = get_sentence_pairs_similarity(df=df_temp.copy(),col1="cui_nomenclature",col2="target_name",filter=False,return_score_only=True)
#     df_hits["sim_score_target_cui"] = get_sentence_pairs_similarity(df=df_temp.copy(), col1="target_name", col2="cui_nomenclature",
#                                                                     model2Name=None,
#                                                                     filter=False, return_score_only=True)

#     # In[59]:
#     for c in list_target_cuis:
#         print(linker.kb.cui_to_entity[c][1])

#     # ## Graph connectivity metric

#     df_path_lengths = get_kg_connections(df_hits, df_kg, list_target_cuis)
#     print(df_path_lengths.shape,"df_path_lengths")
#     print(df_hits.shape, "df_hits")
#     # In[63]:
#     # df_hits = df_hits.merge(df_path_lengths.drop(columns=["node_degree"],
#     #                                              errors="ignore").drop_duplicates(), on="cui", how="left") #added row changed, drop dupes
#     df_hits = df_hits.merge(df_path_lengths.drop(columns=["node_degree"], errors="ignore").drop_duplicates(), on="cui", how="left")
#     # df_hits.drop_duplicates(subset=["cui"]).query("shortest_path_length<50").shortest_path_length.describe().round(1)
#     # In[65]:

#     # ### Save output report
#     # * Saves highly filtered (heuristic) candiadtes, eg.g with no kg hits. (+- path length?)
#     # In[66]:
#     if SAVE_OUTPUTS:

#         df_hits["cui"] = df_hits["cui"].astype(str)
#         print(CANDIDATE_NOVEL_CUIS_FILEPATH)
#         df_temp = df_hits.query("(KG_Hits==0) & (feature_level_min_kg_hits<=3)").drop_duplicates() #  & feature_level_avg_kg_hits<0.6
#         print(df_temp.select_dtypes("O").nunique())
#         # df_temp = df_hits.query("shortest_path_length>2 & feature_level_avg_kg_hits<=0.75")
#         df_temp.to_csv(CANDIDATE_NOVEL_CUIS_FILEPATH, index=False)
#         display(df_temp)


def get_kg_connections(df_hits, df_kg, list_target_cuis,
 get_simplePathLengths = False#True
 ):
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
