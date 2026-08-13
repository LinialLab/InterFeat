#!/usr/bin/env python
# coding: utf-8
from IPython import get_ipython
# * Loads output of tidy diagnoses
# * Filter that for desired target and time horizons.
#     * Possible filterings for less leakage: Keep diagnoses from before gallbladder and/or all from+during first ukbb intake visit (fixed date?).
#     * Could also do model on followup subcohort - future disease - prediction. (But then we will only have ~50k subjects!!) 
# * Then join that with more features
# * Then do modelling and get interesting features
# 
# -----------------
# UKBB first cohort dates: ~ 2006-2010 -
#  First repeat assessment visit (2012-13).
#  
#  https://biobank.ndph.ox.ac.uk/ukb/field.cgi?id=53

# In[1]:

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier  # non linear
from catboost import CatBoostClassifier
from sklearn.feature_selection import mutual_info_classif, SelectFromModel
from sklearn.linear_model import ElasticNet
import shap
from sklearn import set_config
set_config(transform_output="pandas") 

shap.initjs()

# !pip install dcor
# !conda install -c conda-forge feature_engine -y
# ## https://github.com/feature-engine/feature-engine-examples/blob/main/selection/Smart-Correlation-Selection.ipynb

from feature_engine.selection import SmartCorrelatedSelection

# !pip install BorutaShap
# !pip install arfs

try:
    from BorutaShap import BorutaShap
except:
    ()
try:
    import cmim
    from cmim import CMIMFeatureSelector

    CMIM_AVAILABLE = True
except:
    CMIM_AVAILABLE = False

from arfs.preprocessing import OrdinalEncoderPandas
import arfs.feature_selection.allrelevant as arfsgroot

## https://github.com/ThomasBury/arfs/blob/main/docs/notebooks/arfs_boruta_borutaShap_comparison.ipynb

from feature_engine.imputation import RandomSampleImputer
from pandas.api.types import is_datetime64_ns_dtype

# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# https://scikit-learn.org/stable/auto_examples/feature_selection/plot_feature_selection.html#id1

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
from util import *


# In[2]:

def ipw_downsampling(df: pd.DataFrame,
                     K_IPW_RATIO=9, ipw_propensity_cols_list: [] = ['age',
                                                                    'Sex',
                                                                    'age_X_sex',
                                                                    # 'Body mass index (BMI)(participant - p21001_i0)',
                                                                    # "Weight(participant - p21002_i0)",
                                                                    '(BMI) Body mass index (p21001)', # measured bmi +
                                                                    # '(BMI) Body mass index (p23104)'
                                                                    ]):
    for c in ipw_propensity_cols_list:
        assert c in df.columns, f"{c} missing"
    print(df["y"].agg(["mean", "sum", "size"]).round(2), "# target pre IPW")
    df = IPW_resample(df, propensity_cols_list=ipw_propensity_cols_list,
                      K_IPW_RATIO=K_IPW_RATIO, get_ipw_shap=False)
    ## if wew ant to see the feature importances here/now:
    # IPW_resample(df,propensity_cols_list=['Sex','age', 'age_X_sex','Body mass index (BMI)(participant - p21001_i0)'],
    #              K_IPW_RATIO =K_IPW_RATIO,get_ipw_shap = True,X=df.select_dtypes(["number"]).drop(columns="y"),)
    print(df["y"].agg(["mean", "sum", "size"]).round(2), "# target after IPW")
    return df.reset_index(drop=True)


def make_target_df( DIAG_TIDY_PATH =  "../../ukbb-hack/df_diag_tidy.parquet",#"ukbb-hack/df_diag_tidy.parquet",  # ../../
        # phenocodes_map_file_path="../../Phecode_map_v1_2_icd10_beta.csv.zip",  # ../../
phenocodes_map_file_path="Phecode_map_v1_2_icd10_beta.csv.zip", #"../../Phecode_map_v1_2_icd10_beta.csv.zip"#"../../Phecode_map_v1_2_icd10_beta.csv.zip"
phenocodes_def_file_path="phecode_definitions1.2.csv.zip",#"../../pphecode_definitions1.2.csv.zip"
EHR_FEAT_TIDY_PATH = "../../df_ukbb_aux_tidy.parquet",
        TARGET_CODES_LIST=("K80", "K81", "K82")  ## Cholelithiasis = gallstones
        , FILTER_HAS_ANY_FUTURE_DIAGS=True  ## filter to keep cases with any future/post 2010 diagnoses = future record acquisition. May be leaky or unneeded
        , FILTER_HAS_ANY_DIAG_IDS=False
        , ROUND_PHEWAS_CODE_PREC=False  ## round phentypes code prec to 1 digit
        , JOIN_CHOL_FEATS=True
        , FUTURE_DIAG_CUTOFF_YEARS=1  # 1 # in application we look at - of this val
        , FILTER_TARGET_FUTURE_HORIZON=False  # True ## filter for cases within ~3 years (2014) of ukbb - closer horizon, but less cases!
        , USE_TARGET_FROM_ANYTIME=False  # True     ### Include past cases for target, as well as future. Super leaky! use for e.g. MS
        , FILTER_FEMALES_ONLY=False # True ## Keep only females in data. For Gallstones - ROCauc is stable on this subset, as are few top features seemingly.
        , PIVOT_DIAGS_COL_NAME="phenotype"  # "code" #"phenotype"
        , K_diag_thresh_value=200  # 800#250#500
        , FAST_SAMPLE_SIZE=12_000  # 60_000
        , DO_IPW_SAMPLING=False
        # ## get diag feats by z-score, most extreme, or time since most recent value. Recentmost i much faster to extract:
        , GET_EXTREMETMOST_DIAGFEATS=False
        , GET_RECENTMOST_DIAGFEATS=True
        , FAST=False
        , REMOVE_CASES_WITH_PAST_TARGET=False  ## Remove from dataset entirely cases/patients who had one of our targets diagnosed in the past.
        , list_cols_replace_isna_feat=["Fitness test results, including ECG data",
                                       "Acceptability of each blow result (text) | Array 0"]
        , list_sequential_features=['Light - Day average',
                                    'Sedentary - Day average', 'Sleep - Day average',
                                    'Sleep - Weekday hour average', 'Sleep - Weekend hour average',
                                    'Pulse wave pressure versus time response curve']
        # some features that are sequences of numbers - need more processing to use
        , actually_numeric_cat_cols_list=['Coffee intake', 'Tea intake', 'Water intake',
                                          'Age when diabetes-related eye disease diagnosed', 'Age diabetes diagnosed', 'Age heart attack diagnosed'],
           save_targ_df_only = False
        ):
    assert GET_EXTREMETMOST_DIAGFEATS != GET_RECENTMOST_DIAGFEATS
    # global K_diag_thresh_value, CORR_THRESH, do_stat_fs_filt, do_boruta_fs, do_mi_fs_filt, USE_CAT_COLS, FAST,  FEATURES_REPORT_PATH, DO_CV, DROP_FEAT_COLS_LIST, df, c
    # In[5]:

    ## expanded drop feats list
    # DROP_FEAT_COLS_LIST = ["cutoff_date", "prediction_cutoff_year",
    #                        'PRS genetic principal components | Array 0',
    #                        'PRS genetic principal components | Array 1',
    #                        'PRS genetic principal components | Array 2',
    #                        'PRS genetic principal components | Array 3',
    #                        'PRS genetic principal components | Array 4', ]
    # # ## MS leakage related:
    # # ### If doing MS / anytime
    # # ## Don't use otherwise!!!
    # # DROP_FEAT_COLS_LIST.extend([
    # # "Other serious medical condition/disability diagnosed by doctor_Yes - you will be asked about this later by an interviewer",
    # # "Non-cancer illness code, self-reported | Array 0__multiple sclerosis",
    # # "Non-cancer illness code, self-reported | Array 1__multiple sclerosis",
    # # "Non-cancer illness code, self-reported | Array 2__multiple sclerosis",
    # # "Other serious medical condition/disability diagnosed by doctor_No",
    # # "Long-standing illness, disability or infirmity_Yes",
    # # "Long-standing illness, disability or infirmity_No",
    # # "missing_Symptoms involving nervous and musculoskeletal systems",
    # # "Symptoms involving nervous and musculoskeletal systems",
    # # "Other serious medical condition/disability diagnosed by doctor",
    # # "Non-cancer illness code, self-reported",
    # # "Long-standing illness, disability or infirmity",
    # # "Non-cancer illness code, self-reported | Array 0",
    # # "Non-cancer illness code, self-reported | Array 1",
    # # "Non-cancer illness code, self-reported | Array 2",
    # # "Number of self-reported non-cancer illnesses"])
    ######## End Hyperparameters/config variables  ######## End Hyperparameters/config variables
    # In[6]:
    # In[7]:
    #### holds most features, prefenerated from hjoining, includes date of assesment center attendnace etc' . Doesn't hold med codes.
    ## take date of attending asesment for firstim from here (it's 2009-2011 for most )
    ## we'll load the full DF later in pipeline. for now, just the metadata we need for filtering!
    df_meta = pd.read_parquet(EHR_FEAT_TIDY_PATH, columns=["Date of attending assessment centre", "eid"]).set_index("eid") #ORIG: DIAG_TIDY_PATH
    df_meta.rename(columns={"Date of attending assessment centre": "instance0_date"}, inplace=True)
    assert is_datetime64_ns_dtype(df_meta["instance0_date"])
    # In[8]:
    df_diag = pd.read_parquet(DIAG_TIDY_PATH)
    if FILTER_FEMALES_ONLY:
        df_diag = df_diag.query("Sex=='Female'")
        print("Female only")
    print(df_diag.shape[0], "# rows pre assesment date filter")
    df_diag = df_diag.join(df_meta, how="inner", on="eid")  # get instance 0 date .
    assert df_diag.shape[0] > 0
    df_diag = downcast_number_dtypes(df_diag)
    print(df_diag.nunique())
    # print(df_diag.isna().sum())
    assert df_diag.shape[0] > 0
    assert is_datetime64_ns_dtype(df_diag["instance0_date"])
    assert is_datetime64_ns_dtype(df_diag["Diagnosis Date"])
    # In[9]:

    # ###  Numcases with *any* future medical diagnoses
    # * 278,224
    # * Those without any future diagnoses - could be healthy, or could not be in system? We could drop others - this would change population distribution!
    #
    # WEIRD: Filtering ids by this seems to drop all   past diagnosis ids, and leave only future ones? Do not understand why/how
    # In[10]:
    ## seems to drop all cases with past diagnosis???
    ## I move thisdownstream and (optionally) filter pot. targets according to this , not the raw diags, due to weird issue
    # list_ids_any_future_diagnoses = df_diag.loc[df_diag["Diagnosis Date"]>=df_diag["instance0_date"]]["eid"].unique() #ORIG
    # new, uses actual intake/instance 0 date:
    list_ids_any_future_diagnoses = df_diag.loc[df_diag["Diagnosis Date"] >= df_diag["instance0_date"]]["eid"].unique()
    # print(len(list_ids_any_future_diagnoses))
    assert len(list_ids_any_future_diagnoses) > 4
    # #### disease eda
    # * K80: Cholelithiasis ={Calculus of gallbladder =~ gallbladder stones etc')
    #     * also "`Gallstone ileus` - a rare form of small bowel obstruction caused by impaction of gallstone within the small intestine."
    #
    # * Note neoplasms/cancers , and injury/perforation diagnoses

    # In[12]:
    # print(" intersection of post and past diagnosis codes")
    # len(set(df_diag.loc[df_diag["Diagnosis Date"] >= df_diag["instance0_date"]]["eid"].unique()) \
    #     .intersection(set(df_diag.loc[df_diag["Diagnosis Date"] < df_diag["instance0_date"]]["eid"].unique())))
    # In[13]:
    all_codes = df_diag.drop_duplicates(subset=["ICD_Code", "Diagnosis_Description", "eid"])[["ICD_Code", "Diagnosis_Description"]]
    # In[14]:
    df_diag_target = df_diag.loc[df_diag["ICD_Code"].str.startswith(TARGET_CODES_LIST, na=False)]
    print(df_diag_target["eid"].nunique(), "# patients with disease target at any time of their lives")
    # In[18]:
    df_target_future = df_diag_target.loc[df_diag_target["Diagnosis Date"] >= df_diag_target["instance0_date"]].drop_duplicates(["eid", "ICD_Code"])
    assert df_target_future["Diagnosis Date"].isna().sum() == 0
    # print(df_target_future["eid"].nunique())

    # #### Dates of future diagnsis (earliest first) - can use for future cutoff
    # * can sample from this also
    # * Note that it's from 2011 and onwards
    # * Q: Do we want it to be earliest? many are from UKK date only, and this may reduce performance. (Earlier = tests are more relevant, but may miss more diagnoses..)
    #     * Could not drop duplicates by eid...?
    # In[19]:
    df_target_future_dates = df_target_future[["eid", "Diagnosis Date"]].sort_values("Diagnosis Date", ascending=True).drop_duplicates(subset=["eid"])
    df_target_future_dates = df_target_future_dates.set_index(["eid"]).rename(columns={"Diagnosis Date": "cutoff_date"})
    assert df_target_future_dates["cutoff_date"].isna().sum() == 0
    # df_target_future_dates
    # ### add features of past history of disease
    # * skipping specific disease name for now. Getting earlier age
    # * * Could add feature of "time since most recent diagnoses (current date minus max from before ukbb intake)
    # In[20]:
    df_target_past = df_diag_target.loc[df_diag_target["Diagnosis Date"].dt.year <= 2010]
    print(df_target_past["eid"].nunique(), f"past (2010<=) target IDs, {df_target_past.shape[0]}, rows")

    # df_target_past = df_target_past.groupby("eid")[['Age_at_diagnosis', 'Z-score-Age_at_diagnosis']].min() # 'ICD_Code',
    df_target_past = get_max_time_vals(df_target_past, time_col="Diagnosis Date", val_cols=["Age_at_diagnosis", 'Z-score-Age_at_diagnosis'])
    df_target_past = df_target_past.add_prefix("past disease history - ")
    df_target_past["had_disease_past"] = True
    ### drop duplicates (multiple hisorical disease cases
    df_target_past = df_target_past[~df_target_past.index.duplicated(keep='first')]
    print(df_target_past.shape[0], "rows")
    # In[21]:
    ### our diagnoses records first occurence only per code!
    ### little to no overlap - (the little there is, is probably due to different specific codes )
    print("Any intersection of post and past target?:", len(set(df_target_future["eid"].unique()) \
                                                            .intersection(
        set(df_diag_target.loc[df_diag_target["Diagnosis Date"].dt.year <= 2010]["eid"].unique()))))

    # ### Possible multiple future targets
    # * take 2d level icd code ; or CCS or map to phenotypes as targets + filter by commonness
    # * Then predict on all
    #
    # * Re Phenotypes, phewas - https://github.com/spiros/phemap?tab=readme-ov-file , https://phewascatalog.org/phecodes_icd10  , https://phewascatalog.org/phecodes_icd10cm  (icd10-CM codes)
    # * Note: there are 3,885 ICD10 codesi n ukbb not mapped to phewas/phenotypes.
    # In[22]:
    if "pheno" in PIVOT_DIAGS_COL_NAME.lower(): # get, map phenotypes mapping using external data
        df_phecodes_map = pd.read_csv(phenocodes_map_file_path, usecols=['ICD10', 'PHECODE'])
        # category_number
        df_phecodes_map = df_phecodes_map.merge(pd.read_csv(phenocodes_def_file_path, usecols=["phecode",
                                                                                                       "phenotype", "sex", "category"]),
                                                left_on='PHECODE',
                                                right_on="phecode")
        df_phecodes_map.drop(columns=["PHECODE"], errors="ignore", inplace=True)
        df_phecodes_map = df_phecodes_map.rename(columns={"sex": "phenotype_sex",
                                                          "category": "phenotype_cat",
                                                          "ICD10": "ICD_Code"}).drop_duplicates().set_index("ICD_Code")
    # In[23]:
    df_future_diag = df_diag.loc[df_diag["Diagnosis Date"] >= df_diag["instance0_date"]].drop_duplicates(["eid", "ICD_Code"])
    df_future_diag = filter_min_code_counts(df_future_diag, K_value=5, codeColName="ICD_Code")
    df_future_diag = df_future_diag.filter(['eid', 'ICD_Code', 'Diagnosis_Description',
                                            'Diagnosis Date', 'Age_at_diagnosis', 'Z-score-Age_at_diagnosis'], axis=1)
    if FILTER_TARGET_FUTURE_HORIZON:
        print("FILTER_TARGET_FUTURE_HORIZON")
        df_future_diag = df_diag.loc[df_diag["Diagnosis Date"].dt.year <= 2014]
        print(df_future_diag.nunique())
        print(df_future_diag.shape[0])
    # In[24]:
    ## there are 3,885 ICD10 codesi n ukbb not mapped to phewas/phenotypes.
    ## 800 codes that occur 100 or more times. (in future cases)
    if "pheno" in PIVOT_DIAGS_COL_NAME.lower():
        print("ICD10 codes in our data lacking phenwasmapping")
        df_future_diag.loc[~df_future_diag["ICD_Code"].isin(df_phecodes_map.index)].drop_duplicates("ICD_Code")[["ICD_Code", "Diagnosis_Description"]]
        # In[25]:
        df_future_diag = df_future_diag.join(df_phecodes_map, how="left", on="ICD_Code")  # .round(1)
        ## OPT: round Phenotype codes to 1 digit precision, e.g. will gett all the gallbladders together
        if ROUND_PHEWAS_CODE_PREC:
            print(df_future_diag["phecode"].nunique())
            df_future_diag["phecode"] = df_future_diag["phecode"].round(1)
            print(df_future_diag["phecode"].nunique())
    # ## Make target DF
    # * Target, basic features.
    # * Add cutoff time here optionally
    # * Set to study level: for UNK (?) reason 2 rows double appear, so just drop duplicates by eid. - PATIENT LEVEL
    # *
    # In[26]:
    df = df_diag.drop(columns=['ICD_Code', 'Diagnosis_Description',
                               'Diagnosis Date', 'Age_at_diagnosis', 'Z-score-Age_at_diagnosis']).drop_duplicates().copy()
    df["y"] = df["eid"].isin(df_target_future["eid"]).astype(int)
    ### keep 1 record per patient
    df = df.sort_values(["y"], ascending=False).drop_duplicates(subset=["eid"]).sort_values(["eid"], ascending=False)
    df = df.join(df_target_past.drop_duplicates(), on=["eid"], how="left").drop_duplicates()
    df["had_disease_past"] = df["had_disease_past"].fillna(False).astype(bool)
    # print(df["y"].agg(["sum", "mean", "size"]).round(2))
    if REMOVE_CASES_WITH_PAST_TARGET:
        df = df.loc[df["had_disease_past"] != True].copy()
        assert df["y"].sum() > 10, "No pos cases left after removing past cases!"
        print("After removing patients with past disease history:")
        print(df["y"].agg(["sum", "mean", "size"]).round(3))
    if USE_TARGET_FROM_ANYTIME:
        ### Include past cases for target, as well as future. May be leaky!
        ### TODO may want/need to add different diagnosis temporal filtering to use the past cutoff..
        # df["y"] = df[["y","had_disease_past"]].max(axis=1)# past disease seems so lack cases - todo fix?
        df["y"] = df["eid"].isin(df_diag_target["eid"])
        print(df["y"].agg(["sum", "mean", "size"]).round(2))
        df_diag = df_diag.loc[~df_diag["ICD_Code"].str.startswith(TARGET_CODES_LIST, na=False)]  # drop exact matches from diag
        df.drop(columns=["had_disease_past"], errors="ignore", inplace=True)
    if FAST:
        df_pos = df.loc[df["y"] > 0]  # .sample(frac=0.5,random_state=3)
        df = pd.concat([df.sample(min(FAST_SAMPLE_SIZE, df.shape[0]), random_state=3), df_pos]).drop_duplicates().reset_index(drop=True).copy()
        df_diag = df_diag.loc[df_diag["eid"].isin(df["eid"])].reset_index(drop=True).copy()

        # ## added: / new - may not be needed (try to match indexes for later matchings)
        # df_target_future_dates = df_target_future_dates.loc[df_target_future_dates.index.isin(df["eid"])].reset_index(drop=True).copy()
        df_target_future_dates = df_target_future_dates.loc[df_target_future_dates.index.isin(df["eid"])].copy()
    assert df["y"].sum() > 20
    assert df.shape[0] == df["eid"].nunique(), df.shape[0] - df["eid"].nunique()
    # In[27]:
    print(df["y"].describe().round(3))

    if save_targ_df_only:
        targ_names = "".join(TARGET_CODES_LIST)
            # if JOIN_CHOL_FEATS:  ## not just chol feats!!
        print(df.shape, " Pre aux merge")
        df_aux = pd.read_parquet(EHR_FEAT_TIDY_PATH) # "df_ukbb_aux_tidy.parquet"
        df_temp = df.merge(df_aux, how="left", on="eid", suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)').copy()
        ## following should have already been done on the aux in advance, but redo them here to be safe
        df_temp.drop(list_sequential_features, axis=1, inplace=True, errors="ignore")
        df_temp.drop(list(df_temp.filter(regex='Instance [1-4]')), axis=1, inplace=True)  # Future instances are leaks currently - drop them
        del df_aux

        df_temp.sample(frac=1).to_parquet(f"df_{targ_names}.parquet",index=False)
        return df_temp
    # #### future target cutoff time for feats
    # * - has disease after ukbb intake, and set cutoff point as a few years before that, or time of ukbb intake?
    #     * ANND disease appearing after biobank visit, i.e. ignore past ?
    #     *  - Could do multipredictions per person...
    # ##### Filter and remove diagnoses after UKBB visit
    # * Set time to 2011 arbitrarily for now

    # In[28]:
    ## some (few) icd codes lack phewas match
    df_diag = df_diag.join(df_phecodes_map, how="left", on="ICD_Code").drop_duplicates().copy()  # .round(1)
    ## OPT: round Phenotype codes to 1 digit precision, e.g. will gett all the gallbladders together
    if ROUND_PHEWAS_CODE_PREC:
        print(df_diag["phecode"].nunique())
        df_diag["phecode"] = df_diag["phecode"].round(1)
    # #### Filter diagnoses to be before cutoff date (of future diagnosis)
    # ALT (did previousky) - be after 2011.
    # * Sample from dates for rest of pop. (Could sample by ages instead, but this is easier)
    #     * https://stackoverflow.com/questions/36413314/filling-missing-data-by-random-choosing-from-non-missing-values-in-pandas-datafr
    # * or at time of biobank?
    # * horizon of 1 year before? (but after UKBB?)
    #
    # Uses `df_target_future_dates`
    df_target_future_dates = df_target_future_dates.reindex(df["eid"].drop_duplicates())  # will have NaTs for missing eids
    print(df_target_future_dates.shape[0], "# cutoff dates rows after reindexing to match all eids")
    print(df_target_future_dates.isna().sum(), "df_target_future_dates.isna().sum()")
    ## random sampler impute.
    df_target_future_dates = RandomSampleImputer().fit_transform(df_target_future_dates)
    # ## sample from real dates, to use for imputing
    assert df.drop_duplicates(["eid", "y"]).shape[0] == df.shape[0] == df["eid"].nunique()
    # In[36]:
    df_diag = df_diag.join(df_target_future_dates, how="left", on="eid", validate="m:1")  # orig
    if REMOVE_CASES_WITH_PAST_TARGET:
        df_diag = df_diag.loc[~df_diag["eid"].isin(df_target_past.index)]  # drop past cases if using that filter
    assert df_diag["cutoff_date"].isna().sum() == 0, df_diag["cutoff_date"].isna().sum()
    # df_diag["age"] = df_diag["YOB"].sub(2011).abs().astype(int) # age roughly at time of naive cutoff/UKBB visit
    ## Patients age (at time of cutoff!)
    df_diag["age"] = pd.to_numeric(df_diag["cutoff_date"].dt.year.sub(df_diag["YOB"]).abs(), downcast="integer")  # age roughly at time of cutoff
    df_diag["Age_at_diagnosis"] = pd.to_numeric(df_diag["Age_at_diagnosis"], downcast="integer")  # this is age per diagnosis, not overall
    ## add age +- cutofff col to base data
    df = df.merge(df_diag[["eid", "age", "cutoff_date"]].drop_duplicates(), on="eid", how="left", validate="1:1")
    df["prediction_cutoff_year"] = df["cutoff_date"].dt.year
    # * Could set cutoff of 1 year before, etc
    # In[37]:
    # #### Set prediction horizon / time cutoff for diagnosis
    # * `cutoff_date`
    # In[38]:
    df_diag.loc[(df_diag["Diagnosis Date"].sub(df_diag["cutoff_date"]).dt.days / 365) <= -FUTURE_DIAG_CUTOFF_YEARS].drop_duplicates(
        ["cutoff_date"]).head()
    # In[39]:

    df_diag_feat = df_diag.loc[(df_diag["Diagnosis Date"].sub(
        df_diag["cutoff_date"]).dt.days / 365) <= -FUTURE_DIAG_CUTOFF_YEARS]  # dynamic - time horizon within 1 year (instead of just before
    print(df_diag_feat.shape[0], "# diag rows after FUTURE_DIAG_CUTOFF_YEARS cutoff filter")
    df_diag_feat = df_diag_feat.filter(
        ['eid', 'ICD_Code', 'Diagnosis_Description', 'Z-score-Age_at_diagnosis', "phecode", "phenotype", "Age_at_diagnosis", "age"], axis=1).copy()
    df_diag_feat["code"] = df_diag_feat["ICD_Code"].astype(str) + " - " + df_diag_feat['Diagnosis_Description'].astype(str)
    df_diag_feat.drop(columns=['ICD_Code', 'Diagnosis_Description'], errors="ignore", inplace=True)
    df_diag_feat = df_diag_feat.round(2)
    ## some phenotypes missing match, use icd code instead for now - see if it matters?
    print(df_diag_feat["phenotype"].isna().sum(), "# rows missing phenotype")
    df_diag_feat["phenotype"] = df_diag_feat["phenotype"].fillna(df_diag_feat["code"])
    # save memory
    df_diag_feat["Z-score-Age_at_diagnosis"] = pd.to_numeric(df_diag_feat["Z-score-Age_at_diagnosis"].round(2), downcast="float")
    for c in ["code", "phecode", "phenotype"]:
        df_diag_feat[c] = df_diag_feat[c].astype("category")  # incompatible with oivot table?
    # In[41]:
    if GET_EXTREMETMOST_DIAGFEATS:
        df_diag_feat = pivot_long_col_diags(df_diag_feat,
                                            K_value=1.5 * K_diag_thresh_value if FAST else K_diag_thresh_value,
                                            value_col="Z-score-Age_at_diagnosis",
                                            codeColName=PIVOT_DIAGS_COL_NAME
                                            , get_most_extreme_val=True, get_max_val=False,
                                            )

    elif GET_RECENTMOST_DIAGFEATS:
        ## warning - age/yob used here is not correct seemingly, it' time around 2011, not time of sicknessi  thin?
        df_diag_feat = pivot_long_col_diags(df_diag_feat,
                                            K_value=1.5 * K_diag_thresh_value if FAST else K_diag_thresh_value,
                                            value_col="Age_at_diagnosis",
                                            codeColName=PIVOT_DIAGS_COL_NAME
                                            , get_max_val=True,
                                            get_most_extreme_val=False,
                                            subtractionCol='age',
                                            )

    if FILTER_HAS_ANY_DIAG_IDS:
        df = df.join(df_diag_feat, how="inner", on="eid")  ### leaky filter
    else:
        # df = df.join(df_diag_feat,how="left",on="eid") # orig
        df = df.merge(df_diag_feat, how="left", on="eid")
    print(df.shape[0])
    if FILTER_HAS_ANY_FUTURE_DIAGS:
        print("FILTER_HAS_ANY_FUTURE_DIAGS")
        print(df["y"].describe().round(2))
        # list_ids_any_future_diagnoses = list(df_diag.loc[df_diag["Diagnosis Date"].dt.year>=2011]["eid"].unique())
        list_ids_any_future_diagnoses = list(df_diag.loc[df_diag["Diagnosis Date"] >= df_diag["instance0_date"]]["eid"].unique())

        print(len(list_ids_any_future_diagnoses), "#IDs with any future diag")
        df = df.loc[df["eid"].isin(list_ids_any_future_diagnoses)]
        print(df.shape[0])
        # display(df_diag[["Diagnosis Date","eid"]].agg(["min","max","count","nunique"]).round())
        print(df["y"].describe().round(2))
    df.reset_index(drop=True, inplace=True)
    # In[44]:
    list_ids_any_future_diagnoses = list(df_diag.loc[df_diag["Diagnosis Date"] >= df_diag["instance0_date"]]["eid"].unique())
    # print(len(list_ids_any_future_diagnoses), "#IDs with any future diag")
    # #### example extra data file, from hackathon (needs cleaning)
    # In[45]:
    if JOIN_CHOL_FEATS:  ## not just chol feats!!
        print(df.shape, " Pre aux merge")
        df_aux = pd.read_parquet(EHR_FEAT_TIDY_PATH) # "df_ukbb_aux_tidy.parquet"
        df = df.merge(df_aux, how="left", on="eid", suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
        ## following should have already been done on the aux in advance, but redo them here to be safe
        df.drop(list_sequential_features, axis=1, inplace=True, errors="ignore")
        df.drop(list(df.filter(regex='Instance [1-4]')), axis=1, inplace=True)  # Future instances are leaks currently - drop them
        del df_aux
    # In[46]:
    # from sklearn.preprocessing import LabelEncoder
    ## ideally we'd use ""cutoff_date"" but nvm that.  (It is in df_diag)
    ## can also change this to be useful as an interactio nterm? (age X gender)
    # df["age"] = df["YOB"].sub(2011).abs() # age roughly # old - new version done previously + uses real age
    print(df["age"].isna().sum(), "# cases without YoB")
    df["age"] = pd.to_numeric(df["age"], downcast="integer")
    ## todo - zscore of other features by gender?
    if not FILTER_FEMALES_ONLY:
        sex_vec = df["Sex"] == "Male"
        assert sex_vec.sum() != len(sex_vec)
        assert sex_vec.sum() > 0
        sex_vec = -2 * sex_vec + 1  # -1 if sex A, +1 if sex B. (Vs 0/1 interaction term).
        df["age_X_sex"] = sex_vec
        # fig = sns.displot(data=df, kind='kde', x='age', hue='y')
    # In[47]:
    # df["Sex"].value_counts(dropna=False)
    # ## change sex to numeric - default female
    # df["Sex"] = (df["Sex"]=="Female").astype(int)# make into number instead of string
    df["Sex"] = (df["Sex"] == "Female").astype(int)
    if not FILTER_FEMALES_ONLY:
        df.groupby(["y", "Sex"])["age"].describe().round(1)
        assert df["Sex"].nunique() > 1
    # In[51]:
    # #### clean some categoircals that are actually numbers (e.g. all numeric except for 1 string val
    # In[52]:
    for c in actually_numeric_cat_cols_list:
        if c in df.select_dtypes(["O", "string", "category"]).columns:
            # X[c] = pd.to_numeric(X[c].str.replace("Less than one","0").str.replace("Do not know","-1").str.replace("Unknown","-1"),errors="coerce")
            df[c] = pd.to_numeric(df[c].str.replace("Less than one", "0").str.replace("Do not know", "-1").str.replace("Unknown", "-1"),
                                  errors="coerce")
    ## drop some sequential cols
    df.drop(list_sequential_features, axis=1, inplace=True, errors="ignore")
    # ## IPW Resampling
    # * We could do this before the diag feat processing (to save time) but then we wold need to merge the bmi feature earlier.
    # * We resample by propensity according to some major confounders.
    # * Less cases, different distribution after this

    if DO_IPW_SAMPLING:
        df = ipw_downsampling(df)

    # In[55]:
    df = downcast_number_dtypes(df)  # added
    df = df.sample(frac=1).copy()
    # In[56]:
    ### drop pregnancy col, it is annoying proxy instead of gender
    df.drop(columns=["Pregnant"], inplace=True, errors="ignore")
    return df


# ## Try a model


# In[57]:

def final_matrix_and_names(pipeline, X, transformed_feature_names):
    """
    Returns
        X_final  – the design matrix *after* optional SelectKBest
        feat_names_final – the matching column names
    Works both when 'FS' (SelectKBest) exists and when it doesn’t.
    """
    # 1️⃣  run the column‑transformer
    # X_pre = pipeline.named_steps["preprocessor"].transform(X)
    X_pre = pipeline[-2].transform(X) # replace

    # 2️⃣  apply SelectKBest only if it is part of the pipeline
    if "FS" in pipeline.named_steps and hasattr(pipeline.named_steps["FS"], "get_support"):
        selector = pipeline.named_steps["FS"]
        mask = selector.get_support()              # <‑‑ the boolean mask you asked about
        X_final = X_pre[:, mask]                   # keeps sparsity
        feat_names_final = np.array(transformed_feature_names)[mask]
    else:
        X_final = X_pre
        feat_names_final = transformed_feature_names

    return X_final, feat_names_final

# In[60]:

# DROP_FEAT_COLS_LIST = ["cutoff_date", "prediction_cutoff_year",
#                            'PRS genetic principal components | Array 0',
#                            'PRS genetic principal components | Array 1',
#                            'PRS genetic principal components | Array 2',
#                            'PRS genetic principal components | Array 3',
#                            'PRS genetic principal components | Array 4', ]
#     # ## MS leakage related:
#     # ### If doing MS / anytime
#     # ## Don't use otherwise!!!
#     # DROP_FEAT_COLS_LIST.extend([
#     # "Other serious medical condition/disability diagnosed by doctor_Yes - you will be asked about this later by an interviewer",
#     # "Non-cancer illness code, self-reported | Array 0__multiple sclerosis",
#     # "Non-cancer illness code, self-reported | Array 1__multiple sclerosis",
#     # "Non-cancer illness code, self-reported | Array 2__multiple sclerosis",
#     # "Other serious medical condition/disability diagnosed by doctor_No",
#     # "Long-standing illness, disability or infirmity_Yes",
#     # "Long-standing illness, disability or infirmity_No",
#     # "missing_Symptoms involving nervous and musculoskeletal systems",
#     # "Symptoms involving nervous and musculoskeletal systems",
#     # "Other serious medical condition/disability diagnosed by doctor",
#     # "Non-cancer illness code, self-reported",
#     # "Long-standing illness, disability or infirmity",
#     # "Non-cancer illness code, self-reported | Array 0",
#     # "Non-cancer illness code, self-reported | Array 1",
#     # "Non-cancer illness code, self-reported | Array 2",
#     # "Number of self-reported non-cancer illnesses"])


def model_features(df: pd.DataFrame, FAST=False
                   , CORR_THRESH=0.9
                   , do_stat_fs_filt=True
                   , do_boruta_fs=False
                   , do_mi_fs_filt=False
                   , DO_CV=False
                   , USE_CAT_COLS=True
                   , SAVE_OUTPUT=True,
                   K_diag_thresh_value=200
                   , FEATURES_REPORT_PATH="gallstone_chol_ipw_feature_report.csv",
                   DROP_FEAT_COLS_LIST: [] = ["cutoff_date", "prediction_cutoff_year",
                                              'PRS genetic principal components | Array 0',
                                              'PRS genetic principal components | Array 1',
                                              'PRS genetic principal components | Array 2',
                                              'PRS genetic principal components | Array 3',
                                              'PRS genetic principal components | Array 4',
                                              # "Method of diagnosis when first had COVID-19",
                                               ]
                   , do_shap = True, return_res:bool=True,
                   pval_ceiling=0.2,
                    get_feature_importances=True,
                   add_missings_indicator:bool = False,
                   do_no_selection = False ,GET_EXTRA_STATS=True,
                   keep_top_k_only:bool = False
                   ):
    if not USE_CAT_COLS:
        to_drop_text_cols_list = df.select_dtypes(["O", "string", "category"]).columns.tolist()
    else:
        to_drop_text_cols_list = []  # empty list, so none will be dropped from df

    # * Warning/note: "Date of attending assessment centre" as difference from cutoff date  = strong feature!
    # In[62]:
    ## keep or drop categorical/text cols
    X, list_features_high_cardinality, text_cols_list, y = convert_ukbb_df_to_x_y(df, DROP_FEAT_COLS_LIST, K_diag_thresh_value,
                                                                                  to_drop_text_cols_list)

    # In[67]:
    cfs = SmartCorrelatedSelection(
        threshold=min(1.12 * CORR_THRESH, 0.99),
        # method=dcor.distance_correlation ## slower but more flexible, and leaves more cases
        # ,method="kendall",
        # ,method= "spearman",
        ).fit(OrdinalEncoderPandas(return_pandas_categorical=True).set_output(transform="default").fit_transform(X).sample(frac=0.66))  
    ## warning! This doesn't keep the text columns natively!
    # keep_feats = list(set(cfs.get_feature_names_out()+["Sex","Age"]+X.select_dtypes(["O","string","category","boolean"]).columns.tolist()))
    keep_feats = list(set(cfs.get_feature_names_out() + ["Sex", "Age"] + list_features_high_cardinality))  ##
    X = X.filter(keep_feats, axis=1).copy()
    del cfs
    # In[68]:
    ## used in FS parts
    # print("raw X cols", X.shape[1])
    X_encoded = OrdinalEncoderPandas(return_pandas_categorical=True).set_output(transform="default").fit_transform(X.drop(columns=list_features_high_cardinality, errors="ignore"))
    X_encoded = SimpleImputer().set_output(transform="pandas").fit_transform(X_encoded)
    for c in X.select_dtypes(["O", "string", "category"]).columns.tolist():
        if c in X_encoded.columns:
            X_encoded[c] = X_encoded[c].astype("category")
    # In[69]:
    # # ## optional big leaky initial FS
    ##TODO: filter by absolute MI values? +- text cols. (Also, is it really good for untransformed continous featureS?)
    ##NOTEL: MI doesn't catch non linear feature-feature interactions... !
    ## ~2 min
    if do_mi_fs_filt:
        # imputer = SimpleImputer() #  strategy='mean' # add_indicator=False # - would make more cols

        print("X_encoded cols", X_encoded.shape[1])
        ## filter by abs mi value.
        discrete_features = (X_encoded.dtypes == "category").values
        print(sum(discrete_features), "# discrete feats")
        mi_vals_mask = mutual_info_classif(X_encoded, y,  # n_neighbors=5,
                                           discrete_features=discrete_features
                                           # discrete_features=X_encoded.select_dtypes(["O","category"]).columns # fails
                                           ) > 0.001
        print(sum(mi_vals_mask))
        keep_feats = list(set(X_encoded.loc[:, mi_vals_mask].columns.tolist() + ["Sex", "Age"] + list_features_high_cardinality))

        # keep_feats = list(set(fs2.get_feature_names_out().tolist()+["Sex","Age"]+list_features_high_cardinality))
        ## if using ordinal encoder then don't need to force all cats to be kept

        X = X.filter(keep_feats, axis=1).copy()
        X_encoded = X_encoded.filter(keep_feats, axis=1).copy()
        print(len(keep_feats), "# MI selected features in initial leaky filter")
        # print(X.shape[1])
    # In[70]:
    ## commented out - new
    # X_encoded = OrdinalEncoderPandas(return_pandas_categorical=True).fit_transform(X.drop(columns=list_features_high_cardinality,errors="ignore"))
    # In[71]:

    # ## optional big leaky initial FS, keep best per cluster. - extreme leaky - cannot be used like this if doing CV!
    ## takes ~ 3-10 minutes
    if do_stat_fs_filt:
        print(X.shape)
        ## Note! Need to manually readd text features / list_features_high_cardinality back in, they're excluded from X_encoded

        cfs = SmartCorrelatedSelection(threshold=0.99 * CORR_THRESH  # 0.94,
                                       , selection_method="model_performance",
                                       # threshold= 0.99999
                                       # method=dcor.distance_correlation ## slower but more flexible, and leaves more cases
                                       method="spearman",  # "kendall",
                                       estimator=DecisionTreeClassifier(random_state=42)  # min_samples_leaf=3
                                       ).fit(X_encoded, y)
        # keep_feats = list(set(cfs.get_feature_names_out()+["Sex","Age"]+X.select_dtypes(["O","string","category","boolean"]).columns.tolist()))
        keep_feats = list(set(cfs.get_feature_names_out() + ["Sex", "age"] + list_features_high_cardinality))  ##
        X = X.filter(keep_feats, axis=1).copy()
        X_encoded = X_encoded.filter(keep_feats, axis=1).copy()
        del cfs
        print(X.shape[1])

        text_cols_list = [x for x in text_cols_list if x in X.columns]
    # In[72]:
    ## do boruta-shap (All relevant ) leaky FS . Warning! Will take time
    ## https://github.com/ThomasBury/arfs/blob/main/docs/notebooks/arfs_boruta_borutaShap_comparison.ipynb
    ## Note! Need to manually readd text features / list_features_high_cardinality back in, they're excluded from X_encoded
    if do_boruta_fs:
        # !pip install arfs
        # X_encoded = OrdinalEncoderPandas().fit_transform(X
        # from lightgbm import LGBMClassifier
        ## LightGBMError: Do not support special JSON characters in feature name

        # from BorutaShap import BorutaShap

        # bs_feat_selector = BorutaShap(model = HistGradientBoostingClassifier(categorical_features="from_dtype"),
        #                               importance_measure="shap", classification=True,percentile=80)
        bs_feat_selector = arfsgroot.Leshy(  # estimator = HistGradientBoostingClassifier(categorical_features="from_dtype"),
            # cat_features=X_encoded.select_dtypes("category").columns.tolist()
            # LGBMClassifier(random_state=42, verbose=-1),
            estimator=CatBoostClassifier(random_state=42, verbose=0, # ,task_type="GPU"
                max_depth=7),
            # LGBMClassifier(random_state=42)#, verbose=-1),
            n_estimators=50 if FAST else 300 # "auto",  # 600,#
            ,keep_weak=True, perc=33, max_iter=7 if FAST else 13,
            random_state=0, importance="fastshap" )
        # find all relevant features
        bs_feat_selector.fit(X=X_encoded, y=y)  # 100

        # # Returns Boxplot of features
        # bs_feat_selector.plot(X_size=12, figsize=(12, 8), y_scale="log", which_features="accepted") # "all"
        # https://github.com/ThomasBury/arfs/blob/main/docs/notebooks/arfs_classification.ipynb
        print(f"# selected features: {len(bs_feat_selector.get_feature_names_out())}")
        print(f"The naive ranking: {bs_feat_selector.ranking_absolutes_}")

        print(X.shape)
        keep_feats = list(set(bs_feat_selector.get_feature_names_out().tolist() + ["Sex", "age"] + list_features_high_cardinality))  ##
        X = X.filter(keep_feats, axis=1).copy()
        X_encoded = X_encoded.filter(keep_feats, axis=1)
        print(X.shape)
    # In[73]:
    X.filter(text_cols_list, axis=1).head(2)
    print(X.filter(text_cols_list, axis=1).nunique())
    # In[77]:
    clf_model = CatBoostClassifier(
        auto_class_weights="SqrtBalanced",  # SqrtBalanced "Balanced"
        early_stopping_rounds=60,
        # cat_features=X.select_dtypes(["O","string","category"]).columns.to_list(),
        verbose=False, eval_fraction=0.06,
        task_type="GPU"
        # subsample=0.8
        )
    model = clf_model
    ## Example of using shap with a sklearn pipe/transforms -
    ## https://github.com/shap/shap/issues/2611
    # ### SKlearn Pipeline - also get categorical/text feats
    #
    # * https://github.com/shap/shap/issues/2611
    #     * - do pipeline transform sep from model ?
    # In[78]:
    X, categorical_cols, text_cols, numerical_cols = get_coltypes_list(X, return_X=True)
    # In[79]:
    set_config(transform_output="default")  # "pandas"

    # ## CAtboost RFE
    # # Split data
    # X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42,stratify=y)
    # summary = clf_model3.select_features(X_train,y_train,eval_set=(X_test,y_test),train_final_model=False,features_for_select=f"0-{X.shape[1] - 1}",
    #         num_features_to_select=100,logging_level="Silent",plot=True,)["selected_features_names"]
    # ### Catboost SHAP
    # * Much faster.more effecient, but lacks explanations for specific categorical features
    # * XGboost/lgbm maybe supports that? (But not sure if in global view)
    # In[83]:
    # X, categorical_cols,text_cols,numerical_cols = get_coltypes_list(X,return_X =True)
    categorical_cols, text_cols, numerical_cols = get_coltypes_list(X, return_X=False)
    shap_values = None
    # shap_values = get_cb_shap(X, y, categorical_cols, text_cols) # was run here orig
    # In[84]:
    if do_stat_fs_filt:
        shap_values = get_cb_shap(X, y, categorical_cols, text_cols) # added here
        print(X.shape[1])
        vals = shap_values
        feature_importance = pd.DataFrame(list(zip(X.columns.tolist(),  # feature_names,
                                                   vals)),
                                          columns=['name', 'feature_importance']).sort_values('feature_importance')

        # drop features unused by this model- optional. P value doesn't mean features used (e.g. if redundant)
        feature_importance = feature_importance.loc[
            (feature_importance["feature_importance"].abs() > 0) | feature_importance["name"].isin(text_cols)].copy()

        X = X.filter(feature_importance.name, axis=1).copy()  # keep only used features
        print(X.shape[1])
        categorical_cols, text_cols, numerical_cols = get_coltypes_list(X, return_X=False)
    # ### Pipeline explainer
    # In[85]:
    # Preprocessing for categorical data
    if do_no_selection: ## do no FS on these
        categorical_pipeline = Pipeline(steps=[
            # ('bool_to_str', BooleanToStringTransformer()),  # Convert booleans to strings
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing', add_indicator=add_missings_indicator)),  # ,add_indicator=True
            ('onehot', OneHotEncoder(handle_unknown='ignore', min_frequency=30,
                                     sparse_output=False,  ## sparse outputs can cause issues wit hshap maybe?
                                     max_categories=95)),
            ],
            )
        # Preprocessing for text data
        ## warning - multiple text columns needs different processing!
        text_pipeline = Pipeline(steps=[
            ('vectorizer', TfidfVectorizer(min_df=40, max_df=0.9, ngram_range=(1, 2), max_features=700,stop_words="english",
                                           ))
            ],
            )

    else:
        categorical_pipeline = Pipeline(steps=[
            # ('bool_to_str', BooleanToStringTransformer()),  # Convert booleans to strings
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing', add_indicator=add_missings_indicator)),  # ,add_indicator=True
            ('onehot', OneHotEncoder(handle_unknown='ignore', min_frequency=30,
                                     sparse_output=False,  ## sparse outputs can cause issues wit hshap maybe?
                                     max_categories=95)),
            ("FS_cat",SelectFdr(alpha=0.7)) # score_func=chi2,
            # ("FS_cat",SelectKBest(score_func=chi2, k=200)),
            ],
            # memory="cat_pipeline_cache"
            )
        # Preprocessing for text data
        ## warning - multiple text columns needs different processing!
        text_pipeline = Pipeline(steps=[
            # ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), # causes error
            # ## CountVectorizer
            ('vectorizer', TfidfVectorizer(min_df=30, max_df=0.9, ngram_range=(1, 2), max_features=700,stop_words="english",
                                           )),
            ("FS_text", SelectFdr(alpha=0.7))  # score_func=chi2,
            ],
            )

    # Preprocessing for numerical data
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean', add_indicator=add_missings_indicator)),
        ('scaler', StandardScaler()),
        # ('scaler', MaxAbsScaler()),
        # ("FS_num",SelectKBest(score_func=mutual_info_classif, k=600)),
        # ("FS", SelectFromModel(ElasticNet(l1_ratio=0.2), threshold="0.02*mean")),
        ])
    ## warning - multiple text columns needs different processing!
    ## https://github.com/scikit-learn/scikit-learn/issues/16148
    # *[(f'text_{f}', text_transformer, f) for f in text_features],
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(  # n_jobs=2, # ma ycrash?
        # verbose_feature_names_out=False,
        transformers=[
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols),
            # ('text', text_pipeline, text_cols) # ORIG
            *[(f'text_{f}', text_pipeline, f) for f in text_cols],
            ],
        remainder='passthrough', verbose=True
        )
    # Create a preprocessing and modeling pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               # ("FS",SelectKBest(score_func=mutual_info_classif, k=900)),
                               # ("FS",SelectFromModel(ElasticNet(l1_ratio=0.2),threshold = "5.05*mean")), ## may noit actually affect shap ?
                               # ("CorrFS",SmartCorrelatedSelection(threshold= 0.98,#method=dcor.distance_correlation,
                               #                                    selection_method="variance")),
                               ('classifier',
                                clf_model
                                # HistGradientBoostingClassifier(min_samples_leaf=30,max_depth=8,random_state=42,categorical_features="from_dtype")
                                )],
                        # memory="pipeline_cache"
                        )
    if keep_top_k_only: ## not working - sparse issues + feature names not supported
        print("Keeping top K features")
        ## try mrmr? 
        try:
            from feature_engine.selection import MRMR
        except:()
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ("FS",SelectKBest(score_func=mutual_info_classif, k=500)), # could use mrmr also or a faster fs metric/method?
                           # ("FS",MRMR(max_features=200,method="MIQ", regression=False, random_state=3,n_jobs=-2,confirm_variables=False)),

                           # ("FS",SelectFromModel(ElasticNet(l1_ratio=0.2),threshold = "5.05*mean")), ## may noit actually affect shap ?
                           # ("CorrFS",SmartCorrelatedSelection(threshold= 0.98,#method=dcor.distance_correlation,
                           #                                    selection_method="variance")),
                           ('classifier',
                            clf_model
                            )])
    # Fit the pipeline
    pipeline.fit(X, y)
    print(f"Fitted with {len(pipeline[:-1].get_feature_names_out())} feat")
    # transformed_feature_names = [x.replace("num__", "").replace("cat__", "").replace("text_", "").replace("missingindicator", "missing").replace("sklearn","").replace("  "," ").strip() for x in
    #                          pipeline.named_steps['preprocessor'].get_feature_names_out()]  # remove prefixes ## ORIG

    raw_feature_names = pipeline[:-1].get_feature_names_out() # new 

    transformed_feature_names = [x.replace("num__", "").replace("cat__", "").replace("text_", "").replace("missingindicator", "missing").replace("sklearn","").replace("  "," ").strip() for x in
                                 raw_feature_names]  # remove prefixes

    if get_feature_importances:                             
        # if do_shap:
        if shap_values is None:
            shap_values = get_cb_shap(X, y, categorical_cols, text_cols)
        shap_values = run_SHAP_pipeline(X, pipeline, shap_values, transformed_feature_names, FAST)
        ## plt.savefig('fig_tes1.svg', bbox_inches='tight',dpi=100)
    # ### features
    # * warning - cutoff_Date is strong feature (with high cutoff date -> lower chance of disease)
    # In[88]:
    if DO_CV:
        cv_eval(X, y)

    # ### Get and filter top features (by utility)
    if get_feature_importances:
        feature_importance = make_feature_importance_utility_selection(X=X, y=y, pipeline=pipeline, feature_names=transformed_feature_names,
     shap_values=shap_values, SAVE_OUTPUT=SAVE_OUTPUT,FEATURES_REPORT_PATH=FEATURES_REPORT_PATH,pval_ceiling=pval_ceiling,GET_EXTRA_STATS=GET_EXTRA_STATS)
    if return_res:
        if get_feature_importances:
            res_dict = {"feature_importance_df":feature_importance,"X":X,"y":y,"pipeline":pipeline,"feature_names":transformed_feature_names}
        else:
            ## return pipeline only ? 
            res_dict = {"X":X,"y":y,"pipeline":pipeline,"feature_names":transformed_feature_names}
        return res_dict


import scipy.sparse as sp
import pandas as pd

def to_pandas(X_mat, feature_names): # added 
    """
    Convert a dense ndarray **or** a SciPy sparse matrix that comes out of
    a ColumnTransformer into a pandas DataFrame with the right column names.
    Keeps sparsity when possible.
    """
    if sp.issparse(X_mat):
        return pd.DataFrame.sparse.from_spmatrix(X_mat, columns=feature_names)
    if X_mat.ndim == 2 and X_mat.shape[1] == 1 and hasattr(X_mat[0, 0], "shape"):
        # we received the "object-with-CSR-per-row" situation ➜ stack into one CSR
        X_mat = sp.vstack(X_mat[:, 0]).tocsr()
        return pd.DataFrame.sparse.from_spmatrix(X_mat, columns=feature_names)

    return pd.DataFrame(X_mat, columns=feature_names)

def make_feature_importance_utility_selection(X, y, pipeline, feature_names:[], shap_values, SAVE_OUTPUT=False, FEATURES_REPORT_PATH="feature_importance.csv",pval_ceiling=0.2,GET_EXTRA_STATS=True):
    # ### Get and filter top features (by utility)
    #
    # Q: Missing features: maybe remove indicator maybe from name?
    #
    # * Note: `LIFT` here is relative to the _IPW subset_ (if that is being used).
    # * * TODO - we could reextract features over whole population (or subset of feats)... and get lift/feats on that

    # In[111]:
    # X_trans = pipeline.named_steps['preprocessor'].transform(X)  # ,output="pandas") #ORIG

     ## #  new: 
    # if 'FS' in pipeline.named_steps: #  # also new
    #     X_trans = pipeline.named_steps['FS'].transform(X)
    # else:
    #     X_trans = pipeline.named_steps['preprocessor'].transform(X)

    X_trans = Pipeline(pipeline.steps[:-1]).transform(X) # also new, try

    assert X_trans.shape[1] == len(
        feature_names), "Transformed feature names mismatch: X_trans: {X_trans.shape[1]}; feature_names: {len(feature_names)}"

    print(X_trans.shape)

    # # Old – fails when X_trans is CSR
    # X_trans = pd.DataFrame(X_trans, columns=feature_names)

    # New
    X_trans = to_pandas(X_trans, feature_names)


    # X_trans.columns = X_trans.columns.str.replace("missing","",case=False).str.replace("__"," ",regex=False).str.strip() # was enabled , disable here
    assert len(set(X_trans.columns)) == len(X_trans.columns), "non unique col names"
    try:
        vals = np.abs(shap_values.values).mean(0)
    except:
        print("Shap shape issue?!?", shap_values.shape)
        vals = shap_values

    feature_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                      columns=['name', 'feature_importance'])
    ## Add uncorrected P-value
    fs = SelectFdr().fit(SimpleImputer().fit_transform(X_trans), y)
    feature_importance["p_val"] = fs.pvalues_
    # # drop features unused by this model- optional. P value doesn't mean features used (e.g. if redundant)
    feature_importance = feature_importance.sort_values(by=['feature_importance'],
                                                        ascending=False).reset_index(drop=True).round(5)
    ## add correlation of each feature with target
    corr_series = X_trans.select_dtypes("number").corrwith(y.astype(int), method="spearman").round(3)  # originally was kendall
    corr_series.name = "corr"
    feature_importance = feature_importance.merge(corr_series, left_on="name", right_index=True, how="left")
    ## add correlation of each feature with target
    corr_series = X_trans.select_dtypes("number").corrwith(y.astype(int), method="kendall").round(3)  # spearman
    corr_series.name = "corr_kendal"
    feature_importance = feature_importance.merge(corr_series, left_on="name", right_index=True, how="left")

    ## support (non nans cases) per feature - meaningless with imputation
    support_series = X_trans.count()
    support_series.name = "support"
    feature_importance = feature_importance.merge(support_series, left_on="name", right_index=True, how="left")
    print(feature_importance.shape[0],"#all Features")
    feature_importance = feature_importance.loc[(feature_importance["p_val"] < pval_ceiling) | (
            feature_importance["feature_importance"] >= 0.001) ].copy() # | (feature_importance["corr"].abs() >= 0.01)
    print(feature_importance.shape[0],"# minimally filtered features")
    # ## positive class support (non nans cases) as % per feature
    # pos_mask = (y>0)
    # # support_series = 100*(X_trans.loc[pos_mask].count()/y.sum()).round(3) # ORIG
    # support_series = 100*(X_trans[pos_mask].count()/y.sum()).round(3)
    # support_series.name = "%Support_positiveClass"
    # feature_importance = feature_importance.merge(support_series,left_on="name",right_index=True,how="left")
    ## Mutual info. Assumes things. not mrmr. Slow to calc?
    try:
        # # ALT IV: Doesn't seem t o work that great by default
        # from feature_engine.selection import SelectByInformationValue
        # SelectByInformationValue(bins=5, strategy='equal_frequency', threshold=0.01)
        feature_importance["MutualInfoTarget"] = mutual_info_classif(X_trans.filter(feature_importance["name"], axis=1), y,
                                                                     n_jobs=-2)  # - needs newer sklearn version
    except:
        feature_importance["MutualInfoTarget"] = mutual_info_classif(X_trans.filter(feature_importance["name"], axis=1), y)
    # drop features unused by this model- optional. P value doesn't mean features used (e.g. if redundant). MI is univariate
    # feature_importance = feature_importance.loc[
    #     (feature_importance["feature_importance"] > 0) | (feature_importance["MutualInfoTarget"] >= 0.002) | (
    #             feature_importance["p_val"] < 0.01)].copy()
    # print(feature_importance.shape[0])
    feature_importance = feature_importance.loc[(feature_importance["MutualInfoTarget"] >= 0.001) | (feature_importance["p_val"] < pval_ceiling) | (
            feature_importance["feature_importance"] > 0.0005)].copy()
    print(feature_importance.shape[0], "# feat after second filt")
    ## added here - some cleaning of name. done here instead of earlier to allowfeat compat
    ## based on/replaces: X_trans.columns = X_trans.columns.str.replace("missing","",case=False).str.replace("__"," ",regex=False).str.strip()
    feature_importance["raw_name"] = feature_importance["name"]
    feature_importance["name"] = feature_importance["name"].str.replace("missing", "", case=False).str.replace("__", " ", regex=False).str.strip()
    #################
    # ## Optionally: do additional correlation FS. threshold is arbitrary here..
    # ## note - takes time to run. 6 min e.g., after which 631/801 feats left
    # if do_stat_fs_filt:
    #     X_trans = X_trans.filter(feature_importance["raw_name"],axis=1).copy()
    #     print(X_trans.shape)
    #     ## Note! Need to manually re-add text features / list_features_high_cardinality back in, they're excluded from X_encoded
    #     X_trans = SmartCorrelatedSelection(threshold= 0.88,selection_method = "model_performance",
    #                                # method=dcor.distance_correlation ## slower but more flexible, and leaves more cases
    #                                method=  "spearman", #"kendall",
    #                                 estimator= RandomForestClassifier(n_estimators=30,min_samples_leaf=8,n_jobs=-2,max_depth=6), #DecisionTreeClassifier() #min_samples_leaf=3
    #                                ).fit_transform(X_trans,y)
    #     print(X_trans.shape,"After extra corr FS")
    #     feature_importance = feature_importance.loc[feature_importance["raw_name"].isin(X_trans.columns)].reset_index(drop=True)
    #     print(feature_importance.shape)
    #     assert feature_importance.shape[0]==X_trans.shape[1]
    ##################

    ##################
    ## do cmim
    if GET_EXTRA_STATS:
        if CMIM_AVAILABLE:
            print("cmim")
            cmim = CMIMFeatureSelector(task='classification')
            cmim.fit_transform(X_trans.filter(feature_importance["raw_name"]), y)
            pipeline_cmi_scores = cmim.cmi_scores_

            feature_importance["cmim"] = pipeline_cmi_scores
            feature_importance["cmim"] = feature_importance["cmim"].round(5)
        ##################

        ## get lift of optimal split per feature; for selected subset
        print("Getting lift stats")
        df_lift = get_optimal_splits_results(X_trans.filter(feature_importance["raw_name"], axis=1), y, max_depth=3, criterion='gini',
                                             min_support_pct=1, focus_on_lift=True, )
        df_lift = df_lift.filter(['Feature', 'Lift (y==1)', 'Support',
                                  # 'Support (%)',
                                  'Target % Covered',
                                  'Feature Split', ], axis=1).set_index('Feature').add_prefix("F.Split-")
        ## add optimal split stats columns , with a prefix (F.Split-)
        feature_importance = feature_importance.merge(df_lift, right_index=True, left_on="raw_name", how="inner", validate="1:1")
    feature_importance = feature_importance.round(5)
    if SAVE_OUTPUT:
        print(FEATURES_REPORT_PATH, "Saved")
        feature_importance.to_csv(FEATURES_REPORT_PATH, index=False)

    return feature_importance


def run_SHAP_pipeline(X, pipeline, shap_values, transformed_feature_names:[] = None, FAST=False,save_fig=False):
    # # Generate SHAP values
    if transformed_feature_names is None:
        # if 'FS' in pipeline.named_steps: ##NOTE: could replace these with pipeline[:-1].get_feature_names_out() ? 
        #     feature_names = pipeline.named_steps['FS'].get_feature_names_out()
        # else:
        #     feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        feature_names = pipeline[:-1].get_feature_names_out()

        # transformed_feature_names= [x.replace("num__", "").replace("cat__", "").replace("text_", "").replace("missingindicator", "missing") for x in
        #                          pipeline.named_steps['preprocessor'].get_feature_names_out()]
        transformed_feature_names= [x.replace("num__", "").replace("cat__", "").replace("text_", "").replace("missingindicator", "missing") for x in
                         feature_names]
        print("transformed_feature_names is None")
        # print("feature_names",feature_names) 
        # print("transformed_feature_names",transformed_feature_names) 
    try:
        if FAST:
            explainer = shap.TreeExplainer(pipeline.named_steps['classifier'],
                                           # pipeline.named_steps['preprocessor'].transform(X.sample(min(200, X.shape[0]))))  # ORIG, worked
                                           Pipeline(pipeline.steps[:-1]).transform(X.sample(min(500, X.shape[0])))) # NEW
        else:
            explainer = shap.TreeExplainer(pipeline.named_steps['classifier'],
                                           # pipeline.named_steps['preprocessor'].transform(X.sample(min(4_000, X.shape[0])))) #ORIG, worked
                                           Pipeline(pipeline.steps[:-1]).transform(X.sample(min(6_000, X.shape[0])))) # NEW

    except:
        print("Non interventional explainer")  # needed if inuts are sparse/mixed sparse (due to ohe) maybe?
        explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])
    print("Get Shap vals:")
    if FAST:
        X2 = X.sample(n=min(2_000, X.shape[0]))

    else:
        X2 = X.sample(n=min(85_000, X.shape[0]))

    # shap_values = explainer(pipeline.named_steps['preprocessor'].transform(X2)) # ORIG
    shap_values = explainer(Pipeline(pipeline.steps[:-1]).transform(X2)) # NEW
    shap.summary_plot(shap_values, feature_names=transformed_feature_names, max_display=25)

    if save_fig: # not tested + need to handle figure name + setting import in parent function
        plt.savefig('SHAP_fig.svg', bbox_inches='tight', dpi=300)
    return shap_values


def convert_ukbb_df_to_x_y(df, DROP_FEAT_COLS_LIST=[], K_diag_thresh_value=200, to_drop_text_cols_list=[],
                           pipe_delim_columns=['Illnesses of mother', 'Illnesses of siblings', 'Illnesses of adopted father']):
    X = df.drop(columns=["y", "eid", "YOB", "prediction_cutoff_year"] + to_drop_text_cols_list  # +DROP_FEAT_COLS_LIST - done after
                , errors="ignore").dropna(axis=1, thresh=10 + (K_diag_thresh_value // 5)).copy()
    if "Sex" in X.select_dtypes(["O", "string"]).columns:
        X["Sex"] = (X["Sex"] == "Female").astype(int)  # make into number instead of string
    pipe_delim_columns = [x for x in pipe_delim_columns if x in X.columns]
    X = split_pipe_delimited_columns(X, pipe_delim_columns, prefix=True).dropna(
        thresh=50,
        axis=1)
    ##todo: move this to before/ to df part
    ## this is still leaving in future data!
    ### Warning/note: "Date of attending assessment centre" as difference from cutoff date  = strong feature!
    # date_cols_list = ["Date of attending assessment centre"]# ,"cutoff_date" # cutoff_date already present as "prediction_cutoff_year"
    date_cols_list = X.select_dtypes(["datetime"]).columns.difference(
        ["instance0_date", "cutoff_date", "Date of attending assessment centre"]).tolist()
    for c in date_cols_list:
        X[c] = np.where(X[c] <= X["cutoff_date"], X["cutoff_date"].sub(X[c]).dt.days // 30, np.NaN)  # -1) # months before, if exists, otherwise -1
        ## todo: remove future "date X exists" features added in our premunging..
    X.drop(columns=X.select_dtypes(["datetime"]).columns, errors="ignore", inplace=True)
    X.drop(columns=DROP_FEAT_COLS_LIST, errors="ignore", inplace=True)
    text_cols_list = X.select_dtypes(["O", "string", "category"]).columns.tolist()
    print(X.shape)
    # X = DropDuplicateFeatures().fit_transform(X) # same shape
    ## some impouting:
    # for c in text_cols_list:
    #     # X[c] = X[c].fillna(r'""').astype("category") # fillna - needed for catboost
    #     X[c] = X[c].astype("category")
    # print(X.shape[1],"Without duplicate feats")
    ## this is duplicated down below
    m = X.filter(text_cols_list, axis=1).nunique() > 90  # 70
    list_features_high_cardinality = list(X.filter(text_cols_list, axis=1).nunique()[m].index)
    y = df["y"].copy()  # .reset_index(drop=True)
    print(df["y"].agg(["mean", "sum", "size"]).round(3))
    # ##### This FS does NOT necessarily impove model perf!
    # In[63]:
    X.drop(columns=DROP_FEAT_COLS_LIST, errors="ignore", inplace=True)
    return X, list_features_high_cardinality, text_cols_list, y


# In[99]:
#
# ### evaluate contrib of novel features - needs reconfigging + runs cv
# if DO_CV:
#     ### warning: needs novel feature files to be defined
#     evaluate_novel_features_contribution(X=X,y=y,novel_candidates_filename="chol_candidates_search_results.csv")

if __name__ == "__main__":
    TARGET_CODES_LIST = ("K80", "K81", "K82")  ## Cholelithiasis = gallstones

    # In[100]:
    Fast_Run = False#True
    df = make_target_df(TARGET_CODES_LIST=TARGET_CODES_LIST,FAST=Fast_Run)
    print("-------"*5,"\nTarget extracted\n","-------"*8)
    print(df.shape)

    # #### make supplemntary - list of features
    # columns = df.columns.tolist()
    # with open('./Outputs/Figures/Supplement/S1-FeatureList.txt', 'w') as f:
    #     for column in columns:
    #         f.write(f"{column}\n")

    df = ipw_downsampling(df,K_IPW_RATIO=9)
    print(df.shape,"IPW downsampled")
    # In[101]:
    res_dict = model_features(df=df, FAST=Fast_Run,do_boruta_fs=False, SAVE_OUTPUT = True,
                   FEATURES_REPORT_PATH = "gallstone_ipw_broad_feature_report.csv",)
