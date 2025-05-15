import os
# Set the environment variable
os.environ['NX_CUGRAPH_AUTOCONFIG'] = 'True'
## make networkx use gpu
## https://developer.nvidia.com/blog/networkx-introduces-zero-code-change-acceleration-using-nvidia-cugraph/

from IPython.core.display_functions import display
from arfs.preprocessing import OrdinalEncoderPandas
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier # could also try rulesFit ? 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
# import shap  # Assuming you have SHAP installed
from math import log10, floor   
from sklearn import datasets
import networkx as nx
from itertools import combinations

import spacy
import scispacy
from itertools import compress
from scispacy.linking import EntityLinker
from scispacy.abbreviation import AbbreviationDetector ## https://github.com/allenai/scispacy?tab=readme-ov-file#example-usage
# from Link_semmed_cuis import link_kg_concepts, get_kg_connections
from Link_semmed_cuis import get_kg_connections

from sklearn.feature_selection import f_classif, SelectFpr, chi2, SelectFdr, VarianceThreshold, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, SelectPercentile
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict
# https://scikit-learn.org/stable/auto_examples/feature_selection/plot_feature_selection.html#id1
from sklearn.pipeline import make_pipeline
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
from catboost import (CatBoostRegressor, Pool,
                      EShapCalcType, EFeaturesSelectionAlgorithm, EFstrType)
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import string
rem = string.punctuation
punct_pattern = r"[{}]".format(rem)
try: ## add to medrag
    import simdjson as json
except ImportError:
    try:
        import msgspec as json
    except ImportError:
        import json
import re
from tqdm import tqdm
import nltk
lemma = nltk.wordnet.WordNetLemmatizer()
# import sys, os ## append parent path to dir to allow import
# parent_directory = os.path.abspath('..')
# sys.path.append(parent_directory)

import logging

# Set up logger to suppress spaCy's warnings
logger = logging.getLogger("spacy")
logger.setLevel(logging.ERROR)
def anti_join_df(left:pd.DataFrame, right:pd.DataFrame, key):
    """
    https://www.adventuresinmachinelearning.com/mastering-the-art-of-anti-joins-in-pandas-eliminating-rows-with-ease/
    """
    result = pd.merge(left, right, on=key, how='outer', indicator=True, suffixes=('', '_drop'))
    # Filtering to keep only the rows from the left dataframe that do not have a match in the right dataframe
    anti_join = result[result['_merge'] == 'left_only'].copy()
    
    # Dropping the merge indicator and any columns from the right dataframe using regex to match '_drop' suffix
    anti_join.drop(labels=['_merge'] + anti_join.filter(regex='_drop$').columns.tolist(), axis=1, inplace=True)
    
    return anti_join

def get_sentence_pairs_similarity(df,col1:str="cui_nomenclature",col2:str="feature_name",model1Name='FremyCompany/BioLORD-2023',model2Name = 'sentence-transformers/all-MiniLM-L12-v2',filter=False,minFilterValue=0,
    return_score_only:bool = False):
    df = df.copy()
    # Load model
    model = SentenceTransformer(model1Name)
    
    # model2 = SentenceTransformer("allenai-specter") ## note - needs maybe preproc. Could use another model instead

    # # model_3 = SentenceTransformer('S-PubMedBert-MS-MARCO-SCIFACT')
    # models = [SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2'),SentenceTransformer('FremyCompany/BioLORD-2023') ]
    
    embeddings1 = model.encode(df[col1].values, convert_to_tensor=True,normalize_embeddings=True)
    embeddings2 = model.encode(df["feature_name"].values, convert_to_tensor=True,normalize_embeddings=True)
    df["sim_score"] = torch.nn.functional.cosine_similarity(embeddings1, embeddings2).tolist()

    if model2Name is not None:
        model2 = SentenceTransformer(model2Name)
        embeddings1 = model2.encode(df[col1].values, convert_to_tensor=True,normalize_embeddings=True)
        embeddings2 = model2.encode(df["feature_name"].values, convert_to_tensor=True,normalize_embeddings=True)
        df["sim_score2"]  = torch.nn.functional.cosine_similarity(embeddings1, embeddings2).tolist()
        
        df["sim_score"] = df[["sim_score","sim_score2"]].min(axis=1)
    df["sim_score"] =df["sim_score"].round(3)
    if return_score_only:
        return df["sim_score"].round(2).values
    
    df.drop(columns=["sim_score2"],inplace=True,errors="ignore")
    
    # df.sort_values(["sim_score"]).query("sim_score<0.3") # "1.0" values = identical
    if filter:
        s1 = df.shape[0]
        df = df.loc[df["sim_score"]>minFilterValue].copy()
        print(df.shape[0]-s1,"#Rows dropped after similarity filter")
    return df



def round_to_significant_digits(x, significant_digits=2):
    """
    # Examples
    print(round_to_significant_digit(120.12345))   # Output: 120.1
    print(round_to_significant_digit(0.000123))    # Output: 0.0001
    """
    
    if x == 0:
        return 0
    if  np.isinf(x):
        return x
    else:
        # Determine the factor to round to the nearest significant digit
        factor = -int(floor(log10(abs(x)))) + (significant_digits - 1)
        # Use round function to round the number to the calculated factor
        return round(x, factor)


###### optimal splits code - modified from feature interactions

def get_split_mask(tree, feature, node_id):
    """
    Recursively determines the data points that satisfy the split conditions
    leading to a specific node in the decision tree. Handles multi-level trees.
    """
    if node_id == 0:  # Root node
        return np.ones_like(feature, dtype=bool).flatten()  # Flatten here

    parent_id = (node_id - 1) // 2  # Get parent node ID
    parent_threshold = tree.threshold[parent_id]
    is_left_child = tree.children_left[parent_id] == node_id

    # Get the mask for the parent node
    parent_mask = get_split_mask(tree, feature, parent_id)

    # Apply the current node's condition to the parent mask
    if is_left_child:
        current_mask = feature <= parent_threshold  # Go left if <= threshold
    else:
        current_mask = feature > parent_threshold  # Go right if > threshold

    # Combine the parent mask and current mask using logical AND
    mask = np.logical_and(parent_mask, current_mask.flatten())

    return mask.flatten()  # Flatten the final mask

def optimal_split_info(feature, target, handle_missing=True, print_results=False, return_results=True, max_depth=2,
                       criterion='gini', min_support_pct=0.05, min_support=20, focus_on_lift=True,
                       feature_name="feature",missing_approach_median=False):
    """
    Calculates lift and support for the optimal split(s) of a feature based on the
    specified criterion, allowing for multiple splits (max_depth > 1).
    By default imputes min val -99 for missings. 
    Returns only the best split (or range) based on the criterion and minimum support.
    Handles range-style splits and NaN values (using imputation).
    """

    if handle_missing and np.issubdtype(feature.dtype, np.number):
        if missing_approach_median:
            fill_value = np.nanmedian(feature)
        else:
            fill_value = np.nanmin(feature) - 99
        feature = np.where(np.isnan(feature), fill_value, feature)

    feature = feature.reshape(-1, 1)  # Reshape for decision tree
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=0, criterion=criterion,min_samples_split=min_support, min_samples_leaf=min_support)
    clf.fit(feature, target)

    overall_freq_1 = sum(target) / len(target)
    total_target_count = sum(target)  # Total count of target=1

    # Extract split conditions and create ranges
    ranges = []
    thresholds = []  # Store thresholds for each split
    for i in range(clf.tree_.node_count):
        if clf.tree_.children_left[i] != clf.tree_.children_right[i]:  # Not a leaf node
            thresholds.append(clf.tree_.threshold[i])
    
    # Sort thresholds and create ranges
    thresholds.sort()
    if len(thresholds) > 0:
        ranges.append((float('-inf'), thresholds[0]))  # First range
        for j in range(len(thresholds) - 1):
            ranges.append((thresholds[j], thresholds[j+1]))  # Intermediate ranges
        ranges.append((thresholds[-1], float('inf')))  # Last range
    else:
        ranges.append((float('-inf'), float('inf'))) # Single range (no splits)

    # Evaluate lift and support for each range
    best_range = None
    best_lift = -float('inf')  # Initialize with negative infinity
    best_support = 0
    best_support_pct = 0
    best_split_description = ""  # Initialize

    for range_start, range_end in ranges:
        mask = (feature >= range_start) & (feature < range_end)
        mask = mask.flatten() # Flatten the mask

        if mask.sum() == 0:
            continue

        target_array = target.values if isinstance(target, pd.Series) else target

        greater_freq_1 = sum(target_array[mask]) / len(target_array[mask])
        lift_1 = greater_freq_1 / overall_freq_1
        support = len(target_array[mask])
        support_pct = (100 * support) / len(feature)
        # Calculate target percentage covered
        target_pct_covered = (100 * sum(target_array[mask])) / total_target_count 

        if support < min_support or support_pct < min_support_pct:
            continue

        if (focus_on_lift and lift_1 > best_lift) or (not focus_on_lift and support_pct > best_support_pct):
            best_lift = lift_1
            best_support = support
            best_support_pct = support_pct
            # best_range = (round(range_start,4), round(range_end,4))
            # best_range = (range_start,range_end)
            best_range = (round_to_significant_digits(range_start),round_to_significant_digits(range_end))
            best_target_pct_covered = target_pct_covered  # Store target % covered
            
            # Create split description (using feature_name and range)
            if best_range[0] == float('-inf') and best_range[1] == float('inf'):
                best_split_description = "All values"
            elif best_range[0] == float('-inf'):
                best_split_description = f"{feature_name} < {best_range[1]:.2f}"
            elif best_range[1] == float('inf'):
                best_split_description = f"{feature_name} >= {best_range[0]:.2f}"
            else:
                best_split_description = f"{best_range[0]:.2f} <= {feature_name} < {best_range[1]:.2f}"


    if best_range is not None:
        result = {
            # 'Split Condition': best_range,  # Store the range as a tuple
            'Lift (y==1)': best_lift,
            'Support': best_support,
            'Support (%)': best_support_pct,
            'Feature Split': best_split_description,  # Add split description
             'Target % Covered': best_target_pct_covered,  # Add target % covered
            "nan_imputed_val":fill_value
        }
    else:
        result = {}  # Return empty dictionary if no valid split found

    return result



def get_optimal_splits_results(df,y,max_depth:int=1, criterion='gini', min_support_pct=0.5,
                                          focus_on_lift=True,):

    results = []
    for feature_name in df.columns:
        feature = df[feature_name].values#.reshape(-1, 1)
        split_result = optimal_split_info(feature, y, max_depth=max_depth, criterion=criterion, min_support_pct=min_support_pct,
                                          focus_on_lift=focus_on_lift,feature_name=feature_name)  # You can change the criterion here
        if split_result:
            results.append({'Feature': feature_name, **split_result})

    df_results = pd.DataFrame(results).round(2)

    return df_results


def example_splits_usage_titanic():
    # Load the Titanic dataset
    titanic_data = datasets.fetch_openml(name='titanic', version=1)

    # Select features and target variable
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    target = 'Survived'

    X = titanic_data[features]
    y = titanic_data[target]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle missing values using SimpleImputer (more flexible)
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    num_cols = X_train.select_dtypes(include=np.number).columns
    cat_cols = X_train.select_dtypes(include='object').columns

    X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
    X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])

    # One-hot encode categorical features using OneHotEncoder (more efficient)
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_features = encoder.fit_transform(X_train[cat_cols]).toarray()
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(cat_cols))
    X_train = pd.concat([X_train, encoded_df], axis=1).drop(cat_cols, axis=1)
    print(get_optimal_splits_results(X_train,y_train))



## code/funcs from 2-ukbb


def get_most_extreme(df, value_col="Z-score-Age_at_diagnosis"):
    idx = df[value_col].abs().idxmax()
    return df.loc[idx]


def get_min_time_vals(df, time_col="Diagnosis Date", val_cols=['Z-score-Age_at_diagnosis']):
    # Group by 'eid' and get the index of the row with the latest date for each group
    idx = df.groupby('eid')[time_col].idxmin()
    # Use the index to filter the rows with the latest date
    return df.loc[idx, ['eid'] + val_cols].set_index('eid')


def get_max_time_vals(df, time_col="Diagnosis Date", val_cols=['Z-score-Age_at_diagnosis']):
    # Group by 'eid' and get the index of the row with the latest date for each group
    idx = df.groupby('eid')[time_col].idxmax()
    # Use the index to filter the rows with the latest date
    return df.loc[idx, ['eid'] + val_cols].set_index('eid')


def filter_min_code_counts(df, K_value=200, codeColName="code"):
    # Count occurrences and filter codes as before
    code_counts_refined = df[codeColName].value_counts()
    print(len(code_counts_refined), "# all code_counts")
    codes_to_keep_refined = code_counts_refined[code_counts_refined >= K_value].index
    print(len(codes_to_keep_refined), "# codes_to_keep_refined")

    df = df.loc[df[codeColName].isin(codes_to_keep_refined)].reset_index(drop=True)
    if df[codeColName].dtype == "category":
        df[codeColName] = df[codeColName].cat.remove_unused_categories()
    return df


def pivot_long_col_diags(df, K_value=3_00, value_col="Z-score-Age_at_diagnosis", codeColName="code",
                         get_most_extreme_val=True,
                         get_max_val=False,  # may be best to do seperately
                         subtractionCol="age",
                         # try_polars = False
                         ):
    assert get_most_extreme_val != get_max_val, f"get_most_extreme_val({get_most_extreme_val}) must differ from get_max_val{get_max_val}"
    df = df.dropna(subset=[codeColName, value_col], axis=0)  # .drop_duplicates() # added
    df = filter_min_code_counts(df=df, K_value=K_value, codeColName=codeColName)  # .reset_index(drop=True)

    if get_most_extreme_val:
        df = df.groupby(['eid', codeColName], observed=True)
        # Define a function to get the row with the most extreme Z-score for each eid-code pair
        # Apply the function and reset index
        df = df.apply(get_most_extreme, value_col=value_col).reset_index(drop=True)

    elif get_max_val:  # not tested
        df = df.sort_values(by=["eid", value_col, codeColName], ascending=False).drop_duplicates(["eid", value_col, codeColName])
        df[value_col] = df[value_col].sub(df[subtractionCol])

    df = df.pivot_table(
        values=value_col,
        index="eid",
        columns=codeColName,
        observed=True
    )

    df = df.reset_index().set_index("eid")
    return df


from sklearn.base import BaseEstimator, TransformerMixin


# Custom transformer to convert boolean to string
class BooleanToStringTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.astype(str)


def downcast_number_dtypes(df):
    ## downcast datatypes 
    for col in df.select_dtypes('number'):
        df[col] = pd.to_numeric(df[col], downcast='integer')
        if df[col].dtype == 'float':
            df[col] = pd.to_numeric(df[col], downcast='float')
    print(df.dtypes)
    return df


def split_pipe_delimited_columns(df, columns, prefix=True):
    for col in columns:
        # df[col] = df[col].str.replace("None of the above", "None")
        split_cols = df[col].str.split('|', expand=True)
        if prefix:
            split_cols = split_cols.add_prefix(f'{col}_')
        df = df.drop(columns=[col]).join(split_cols)
    return df


def cv_eval(X, y, n_cv=5, model=None):
    score_metrics = ["roc_auc", 'accuracy', 'precision', "recall", "average_precision", "f1"]  # ,"balanced_accuracy"
    if model is None:
        model = CatBoostClassifier(iterations=500,
                                   auto_class_weights="SqrtBalanced"  # SqrtBalanced "Balanced"
                                   , early_stopping_rounds=50
                                   # task_type="GPU",
                                   , verbose=False,
                                   # cat_features=(categorical_cols+text_cols), ),
                                   cat_features=X.select_dtypes(["O", "string", "category", "object"]).columns.to_list(), )
    native_result = cross_validate(
        # pipeline,
        model,
        X, y, cv=n_cv
        # ,cv=3 if FAST else 4
        , scoring=score_metrics)  # ,n_jobs=3,

    for metric in score_metrics:
        metric_mean = np.mean(native_result[f"test_{metric}"])
        metric_std = np.std(native_result[f"test_{metric}"])
        print(f"{metric}: {100 * metric_mean:.2f} +-{metric_std:.3f} SD")


## "Body mass index (BMI)(participant - p23104_i0)"
def IPW_resample(df, propensity_cols_list=['Sex', 'age', 'age_X_sex', "Body mass index (BMI)(participant - p23104_i0)"],
                 K_IPW_RATIO=9, X=None, get_ipw_shap=False):
    # K_IPW_RATIO = 1 if FAST else 3 #3

    X_ipw = OrdinalEncoderPandas().fit_transform(df[propensity_cols_list])
    X_ipw["eid"] = df["eid"]
    clf_cal = CalibratedClassifierCV(
        estimator=HistGradientBoostingClassifier(max_iter=300, categorical_features="from_dtype"  # ["Sex"] #
                                                 , random_state=42, ),
        # estimator=LogisticRegression(),
        method="isotonic"
    )

    dmg_propensity = cross_val_predict(estimator=clf_cal,
                                       X=X_ipw.drop(columns=["eid"], errors="ignore"), n_jobs=-1, cv=8,
                                       y=df["y"], method='predict_proba', )[:, 1]
    df_ipw = df.copy()
    df_ipw["propensity"] = dmg_propensity  # [:,1]

    # positive_class_propensity_cutoff = df_ipw.loc[df_ipw["y"]>0]["propensity"].quantile(0.001)
    # print(positive_class_propensity_cutoff)
    # # df_ipw_subset = df_ipw.loc[(df_ipw["propensity"]>=positive_class_propensity_cutoff)|(df_ipw["y"]>0)]
    df_ipw_subset = df_ipw.loc[(df_ipw["propensity"] > 0) | (df_ipw["y"] > 0)]
    print(df_ipw_subset.shape[0])
    # df_ipw_subset.groupby(["y"])["propensity"].plot(kind="kde")

    ## IPW weighted sampling! 
    df_pos = df_ipw_subset.loc[df_ipw_subset["y"] > 0]
    num_pos = df_ipw_subset["y"].sum()
    df_neg = df_ipw_subset.loc[df_ipw_subset["y"] == 0]
    df_neg = df_neg.sample(n=min((K_IPW_RATIO * num_pos), df_neg.shape[0]), weights="propensity")

    df_ipw_subset = pd.concat([df_pos, df_neg]).drop(columns=["propensity"]).drop_duplicates()
    print("After IPW downsampling:", df_ipw_subset.shape[0])
    # df_ipw_subset.groupby(["y"])["propensity"].plot(kind="kde")

    df_subset_ipw = df.loc[df["eid"].isin(df_ipw_subset["eid"])].dropna(axis=1, thresh=50).copy()

    if get_ipw_shap:
        print("Getting IPW based SHAP")
        X_subset_ipw = df_subset_ipw.filter(X.columns.tolist(), axis=1).dropna(axis=1, thresh=100)

        categorical_cols, text_cols, numerical_cols = get_coltypes_list(X_subset_ipw)

        _ = get_cb_shap(X=X_subset_ipw,
                        y=df_subset_ipw["y"],
                        categorical_cols=categorical_cols, text_cols=text_cols)
    return df_ipw_subset


# IPW_resample(df,propensity_cols_list=['Sex','age', 'age_X_sex', "Body mass index (BMI)(participant - p23104_i0)"],
#                  K_IPW_RATIO =1,X=X,get_ipw_shap = True)

def get_coltypes_list(X: pd.DataFrame, return_X=False):
    text_cols_list = X.select_dtypes(["O", "string", "category"]).columns.tolist()

    # X.filter(text_cols_list,axis=1).loc[X.filter(text_cols_list,axis=1).nunique()>15]
    m = X.filter(text_cols_list, axis=1).nunique() > 90  # 20
    list_features_high_cardinality = list(X.filter(text_cols_list, axis=1).nunique()[m].index)
    # print(list_features_high_cardinality)
    categorical_cols = X[text_cols_list].columns.difference(list_features_high_cardinality).to_list()

    # Define categorical, numerical and text columns
    # categorical_cols = categorical_cols  # Update with your actual categorical columns
    text_cols = list_features_high_cardinality  # Update with your actual text columns
    numerical_cols = X.select_dtypes(include=['number', "boolean"]).columns.difference(categorical_cols + text_cols).to_list()

    ## remove some cases where seemingly col missing in actual data?
    categorical_cols = [x for x in categorical_cols if x in X.columns]
    text_cols = [x for x in text_cols if x in X.columns]
    print("text_cols", text_cols)

    if return_X:
        for c in text_cols:
            X[c] = X[c].fillna("None").astype(str)  # .astype("O")#.astype("category") #.astype(str)
        X[categorical_cols] = X[categorical_cols].astype(str).fillna("Empty").astype("category")  # handle mixed Object/Booleans
        # X[categorical_cols] = X[categorical_cols].astype(str).astype("category") # try without imputing
        return X, categorical_cols, text_cols, numerical_cols
    else:
        return categorical_cols, text_cols, numerical_cols


# categorical_cols,text_cols,numerical_cols = get_coltypes_list(X)

def get_cb_shap(X, y, categorical_cols, text_cols, return_shapVal=True,get_interactions=False):
    # X2 = X.head(X.shape[0]//30).copy().reset_index(drop=True)
    X2 = X

    clf_model2 = CatBoostClassifier(  # depth=5,
        #iterations=800,
         auto_class_weights="SqrtBalanced"  # "SqrtBalanced" # SqrtBalanced "Balanced"
        , early_stopping_rounds=90
        # task_type="GPU",
        # cat_features=X2.select_dtypes(["O","string","category","object"]).columns.to_list(),
        , verbose=False,
        eval_fraction=0.05,
        cat_features=(categorical_cols + text_cols),
        # text_cols - seemingly broken in current catboost?
        # cat_features=categorical_cols,#+text_cols,
        # text_features=["Treatment/medication code | Array 0"] #text_cols ## seemingly broken? 
        # text_features=text_cols 
    )

    clf_model2.fit(X2, y.head(X2.shape[0]), plot=False)
    explainer = shap.Explainer(clf_model2)

    print("fitted")

    shap_values = explainer(X2, y=y.head(X2.shape[0]))

    if get_interactions:
    # Calculate top 10 feat interactions
        feat_interactions = clf_model2.get_feature_importance(type=EFstrType.Interaction, prettified=True)

        top_interactions = feat_interactions[:10].round(2)
    summary = {X2.columns.get_loc(c):c for c in X2.columns}
    if get_interactions:
        top_interactions['First Feature Index'] = top_interactions['First Feature Index'].apply(lambda x: summary[x])
        top_interactions['Second Feature Index'] = top_interactions['Second Feature Index'].apply(lambda x: summary[x])
        top_interactions.columns = ['First Feature', 'Second Feature', 'Interaction']
        print(top_interactions)
    ## overall feat imp
    display(shap.plots.beeswarm(shap_values, max_display=30))  # ,log_scale=True
    if return_shapVal:
        shap_values = np.abs(shap_values.values).mean(0)  # get abs val
        return shap_values

def wrangle_df_icu(icu_feature_terms: pd.DataFrame,filter_feat_imp=True):
    print(icu_feature_terms.shape,"# feats preclean")
    if filter_feat_imp:
        icu_feature_terms = icu_feature_terms.loc[(icu_feature_terms["feature_importance"] > 0.0009) |(icu_feature_terms["MutualInfoTarget"] >= 0.001)|(icu_feature_terms["p_val"] < 0.06) ]  # filter a bit - optionally
        print(icu_feature_terms.shape,"# feats after mini filt and load")
    icu_feature_terms = icu_feature_terms.reset_index(drop=True).copy()
    
    assert "raw_name" in icu_feature_terms.columns
    # icu_feature_terms["raw_name"] = icu_feature_terms["name"]
    icu_feature_terms["name"] = icu_feature_terms["name"].str.replace("_nan", " ").str.replace("_", " ",
                                                                                               regex=False)  # .str.replace("."," ",regex=False).str.strip()
    icu_feature_terms["name"] = icu_feature_terms["name"].str.replace("missing ", "", case=False).str.replace("(", " (", regex=False).str.replace(
        "  ", " ", regex=False).str.replace("_Empty", "", regex=False).str.strip()
    ## following filters may increase noise? or may help with NEL? Note it means different featre shape
    icu_feature_terms["name"] = icu_feature_terms["name"].str.replace("left|right|Major|missing|Array [0-9]|Standard ", "", case=True,
                                                                      regex=True).str.replace("()", " ", regex=False).str.replace("  ", " ",
                                                                                                                                  regex=False).str.strip()
    icu_feature_terms["name"] = icu_feature_terms["name"].str.replace(" None$", "", regex=True,
                                                                      case=False)  ## remove some cases of "none" at end of feat.
    icu_feature_terms["name"] = icu_feature_terms["name"].str.replace("PRS", "Genetic risk").str.replace("  ", " ",
                                                                                                         regex=False).str.strip()  # PRS gets lots of noise.
    icu_feature_terms["name"] = icu_feature_terms["name"].str.replace(" [0-9]{1,5}$", "", regex=True).str.strip()  # remove number at end
    icu_feature_terms["name"] = icu_feature_terms["name"].str.replace(" No$|Yes$|Do not know$", "",
                                                                      regex=True).str.strip()  # some noise when searching maybe
    icu_feature_terms["name"] = icu_feature_terms["name"].str.replace("Treatment/medication code |", "medication", regex=False)
    icu_feature_terms["name"] = icu_feature_terms["name"].str.replace("Non-cancer illness code, self-reported | ", "", regex=False).str.strip()

    # df_icu_feature_terms = icu_feature_terms.copy() # raw vals
    ## warning - feature values ma y be wrong for cases of "missing" placeholder featues" (will have same name
    ##NEW: Add "raw_name" - removed, caused many join issue 
    icu_feature_terms = icu_feature_terms.drop_duplicates(subset=["name"]).reset_index(drop=True)  # some dupe terms, e.g. due to missing being removed
    return icu_feature_terms

def evaluate_novel_features_contribution(X,y,novel_candidates_filename = "chol_candidates_search_results.csv"):
    
    # novel_candidates_filename = "candidate_novel_cuis_chol.csv" # = KG filtered, but not literature searched
    df_novel_candidates = pd.read_csv(novel_candidates_filename)
    try:
        df_novel_candidates = df_novel_candidates.query(
            "feature_level_sum_kg_hits<=1 & feature_level_avg_kg_hits<0.4")  ## stringent filter - ~ half as many features
    except:
        df_novel_candidates = df_novel_candidates.loc[
            (df_novel_candidates["Co-occurrence Count"] == 0) & (df_novel_candidates.feature_level_avg_kg_hits < 0.3)]
    if "Co-occurrence Count" in df_novel_candidates.columns.tolist():
        df_novel_candidates = df_novel_candidates.loc[df_novel_candidates["Co-occurrence Count"] == 0]
    df_novel_candidates.drop_duplicates(subset=["feature_name"], inplace=True)
    novel_names = df_novel_candidates["feature_name"].unique()
    print(len(novel_names),"# novel_names features")
    print(X.filter(novel_names,axis=1).shape[1].dropna(axis=1,how="all").shape[1],"# novel features present in X")
    # In[102]:
    # X.filter(novel_names, axis=1).shape[1]

    # print(X[X.columns.difference([i for i in X.columns if i not in (novel_names)])].shape[1])
    # X[X.columns.difference([i for i in X.columns if i not in (novel_names)])].columns
    # In[106]:
    # X_no_novels = X[[i for i in X.columns if i not in (novel_names)]]
    # if DO_CV:
    print("CV without novels")
    cv_eval(X[[i for i in X.columns if i not in (novel_names)]], y)
    
    print("CV ONLY novels")
    cv_eval(X[novel_names], y)
    print("CV ALL novels")
    cv_eval(X, y)


def test_nel(text,nlp,do_print=False):
    doc = nlp(text)
    # Let's look at a random entity!
    if do_print: print("All ents", doc.ents)
    ## broken code snippet : https://github.com/allenai/scispacy/issues/355
    for e in doc.ents:
        if e._.kb_ents:
            cui = e._.kb_ents[0][0]
            print(e, cui)

    if do_print:print("\n--------------------------\n")
    entity = doc.ents[0]
    if do_print:print("Name: ", entity)

    # Each entity is linked to UMLS with a score
    # (currently just char-3gram matching).
    if do_print:
        linker = nlp.get_pipe("scispacy_linker")
        for umls_ent in entity._.kb_ents:
            print(linker.kb.cui_to_entity[umls_ent[0]])
    return entity

def get_multiple_nel(text, nlp,do_print=False):
    "modified to get multiple entities , not just first"
    doc = nlp(text)
    if do_print:print("All entities:", doc.ents)
    
    entities_with_cuis = []
    
    for e in doc.ents:
        if e._.kb_ents:
            cuis = [cui[0] for cui in e._.kb_ents]
            if do_print:print(f"Entity: {e}, CUIs: {cuis}")
            entities_with_cuis.append({
                'entity': e,
                'cuis': cuis
            })
    
    if do_print:
        print("\n--------------------------\n")
        linker = nlp.get_pipe("scispacy_linker")
        for ent in entities_with_cuis:
            print(f"Details for Entity: {ent['entity']}")
            for umls_ent in ent['cuis']:
                print(linker.kb.cui_to_entity[umls_ent])
    
    return entities_with_cuis

### copy from Link_Semmed_cuis.py: (fat version)
def link_kg_concepts(FEATURES_REPORT_PATH:str, CANDIDATE_NOVEL_CUIS_FILEPATH:str, TARGET_NAME:str, additional_target_cui_terms_list=[], SAVE_OUTPUTS = True, MIN_EVIDENCE_FILTER = 2,
    df_features:pd.DataFrame= None, do_feat_imp_filt:bool=True,
                     REMOVE_CUI_TERMS_LIST=['Prieto syndrome', "Polarized Reflectance Spectroscopy",
                                            # mistaken extraction from PRS - drop it for now for cleanliness
                                            'Standard (qualifier)', 'Standard base excess calculation technique',
                                            'Standard of Care', 'Spatial Frequency', 'Disease',
                                            'Statistical Frequency', 'Kind of quantity - Frequency', 'Concentration measurement',
                                            'Concentration measurement', 'Illness (finding)', 'Concentration Ratio',
                                            "Special", "ActInformationPrivacyReason <operations>", "Left sided", "Left", "Right",
                                            "Table Cell Horizontal Align - left", "Query Quantity Unit - Records", "Up",
                                            "Qualification",
                                            "Visit", "Total", "Participant", "Overall", "Right sided", "Left sided",
                                            "Take", "Percent (qualifier value)","Population Group",
                                            "Diagnosis","Coding","Code",
                                            "Average" , "Comparison" , "Lost" , "Yes - Presence findings", 
                                            "Pharmaceutical Preparations","Physicians","Mother (person)","Father (person)","Severe (severity modifier)",
                                            ]
                     , input_kg_path="../SemMed/predications.parquet", EXCLUDE_TUIS_LIST = ["T079", "T093", "T094", "T095", "T170", "T204", "T201", "T065",
                         "T078", ], sem_similarity_threshhold_score=0.15, # 0.15
                     top_cutoff_simMin = 0.39,top_cutoff_kgHit = 2):

    # global nlp, df_kg_sep, df_hits, G
    global df_kg_sep, df_hits ## maybe disable this..

    # ### Load processed semmed db
    # * ths version has 1 row per triple.
    # * `Count` is the number of unique PMIDs the triple has appeared in
    # * Idea: we could filter for triples that appear at least K times. +- filter papers with less than 1-2 citations
    # In[5]:
    df_kg = pd.read_parquet(input_kg_path)

    # ### Filtered version
    # * Could Keep cases with more than ~3 evidences (note: counts of evidences are counts of unique papers with that SVO triple).
    # * * could laso filter by HQ papers (With external dataset linked to the PMIDs in raw data - e.g. using pubmedKG for citation counts)
    # In[8]:
    # print(df_kg.shape[0])  # 26M
    # df_kg["counts"] = df_kg[["SUBJECT_CUI","OBJECT_CUI","PREDICATE"]].groupby(["SUBJECT_CUI","OBJECT_CUI"],observed=True)["PREDICATE"].transform("size") # count evidence irregardless of predicate
    df_kg = df_kg.loc[df_kg["pair_counts"] >= MIN_EVIDENCE_FILTER].reset_index(drop=True).copy()
    df_kg.drop(columns=["pair_counts","counts"], inplace=True, errors="ignore")
    print("After filtering KG min count")
    for c in df_kg.select_dtypes("category").columns:
        # remove unobserved categories, in new filtered data
        df_kg[c] = df_kg[c].cat.remove_unused_categories()
    print(df_kg.shape[0])  # 4M

    # # !pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz
    # # !pip install 'spacy[transformers]'
    # # !python -m spacy download en_core_web_sm
    # # !pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz ## small
    # ```
    # In[11]:
    # %%time
    # nlp = spacy.load("en_core_sci_sm")
    # there's also lg, and transformer based
    nlp = spacy.load("en_core_sci_lg")
    # nlp = spacy.load("en_core_sci_scibert")
    # In[12]:
    # This line takes a while, because we have to download ~1GB of data
    # and load a large JSON file (the knowledge base). Be patient!
    # Thankfully it should be faster after the first time you use it, because
    # the downloads are cached.
    # NOTE: The resolve_abbreviations parameter is optional, and requires that
    # the AbbreviationDetector pipe has already been added to the pipeline. Adding
    # the AbbreviationDetector pipe and setting resolve_abbreviations to True means
    # that linking will only be performed on the long form of abbreviations.

    # Add the abbreviation pipe to the spacy pipeline. (if using resolve_abbreviations )
    nlp.add_pipe("abbreviation_detector")

    nlp.add_pipe("scispacy_linker",
                 config={"resolve_abbreviations": True,
                         "linker_name": "umls",
                         "max_entities_per_mention": 3  # 3, #6, #4, #5
                     , "threshold": 0.88  ## default is 0.8, paper mentions 0.99 as thresh
                         })


    def test_nel(text: str = "Sepsis"):
        doc = nlp(text)

        # Let's look at a random entity!
        print("All ents", doc.ents)
        ## broken code snippet : https://github.com/allenai/scispacy/issues/355
        for e in doc.ents:
            if e._.kb_ents:
                cui = e._.kb_ents[0][0]
                print(e, cui)

        print("\n--------------------------\n")
        entity = doc.ents[0]
        print("Name: ", entity)

        # Each entity is linked to UMLS with a score
        # (currently just char-3gram matching).
        linker = nlp.get_pipe("scispacy_linker")
        for umls_ent in entity._.kb_ents:
            print(linker.kb.cui_to_entity[umls_ent[0]])

        # linker.kb.cui_to_entity[umls_ent[0]][3][0] # TUI
        # linker.kb.cui_to_entity[umls_ent[0]][4] # definition
        # linker.kb.cui_to_entity[umls_ent[0]][1] # cui-name
        # linker.kb.cui_to_entity[umls_ent[0]][2] # aliases
        return entity

        # >>> CUI: C1839259, Name: Bulbo-Spinal Atrophy, X-Linked
        # >>> Definition: An X-linked recessive form of spinal muscular atrophy. It is due to a mutation of the
        #                   gene encoding the ANDROGEN RECEPTOR.
        # >>> TUI(s): T047
        # >>> Aliases (abbreviated, total: 50):
        #          Bulbo-Spinal Atrophy, X-Linked, Bulbo-Spinal Atrophy, X-Linked, ....
    print("TARGET_NAME (For Entity-KG linking", TARGET_NAME)
    # entity= test_nel("GALLSTONES, Cholelithiasis") # "Cholelithiasis" = Gallstone
    # entity= test_nel("Gout") # GOUT
    # C0007570, Name: Celiac Disease
    entity = test_nel(TARGET_NAME)
    # #### Target terms - may manually change
    ## list of extracted CUIs meaning sepsis. Note that not all correct even here and with threshhold
    list_target_cuis = [i[0] for i in entity._.kb_ents]
    if len(additional_target_cui_terms_list) > 0:
        list_target_cuis = list(set(list_target_cuis + additional_target_cui_terms_list))
    print("list_target_cuis",list_target_cuis)
    # #### get subset of KG with target in pairs
    # (not 100% sure if ideal, but will save time when comparing features )
    # In[26]:
    # sepsis_cui = 'C0518988'
    # df_kg_sep = df_kg.loc[(df_kg["SUBJECT_CUI"]==sepsis_cui) | (df_kg["OBJECT_CUI"]==sepsis_cui)].copy()
    df_kg_sep = df_kg.loc[(df_kg["SUBJECT_CUI"].isin(list_target_cuis)) | (df_kg["OBJECT_CUI"].isin(list_target_cuis))].copy()
    df_kg_sep.drop_duplicates(['SUBJECT_CUI', 'SUBJECT_NAME', 'OBJECT_CUI', 'OBJECT_NAME'],
                              inplace=True)  # ignore predicate type for this filter table
    # In[28]:
    ## sample set of terms from gallstone prediction + filtered
    if df_features is None:
        icu_feature_terms = pd.read_csv(FEATURES_REPORT_PATH)
    else:
        icu_feature_terms = df_features ##TODO: refactor name of variable
    if do_feat_imp_filt:
        icu_feature_terms = icu_feature_terms.loc[(icu_feature_terms["feature_importance"] > 0.0001) |(icu_feature_terms["MutualInfoTarget"] >= 0.001)|(icu_feature_terms["p_val"] <= 0.05) ]  # filter a bit - optionally
    assert "raw_name" in icu_feature_terms.columns
    # icu_feature_terms["raw_name"] = icu_feature_terms["name"]
    icu_feature_terms["name"] = icu_feature_terms["name"].str.replace("_nan", " ").str.replace("_", " ",
                                                                                               regex=False)  # .str.replace("."," ",regex=False).str.strip()
    icu_feature_terms["name"] = icu_feature_terms["name"].str.replace("missing ", "", case=False).str.replace("(", " (", regex=False).str.replace(
        "  ", " ", regex=False).str.replace("_Empty", "", regex=False).str.strip()
    ## following filters may increase noise? or may help with NEL? Note it means different featre shape
    icu_feature_terms["name"] = icu_feature_terms["name"].str.replace("left|right|Major|missing|Array [0-9]|Standard ", "", case=True,
                                                                      regex=True).str.replace("()", " ", regex=False).str.replace("  ", " ",
                                                                                                                                  regex=False).str.strip()
    icu_feature_terms["name"] = icu_feature_terms["name"].str.replace(" None$", "", regex=True,
                                                                      case=False)  ## remove some cases of "none" at end of feat.
    icu_feature_terms["name"] = icu_feature_terms["name"].str.replace("PRS", "Genetic risk").str.replace("  ", " ",
                                                                                                         regex=False).str.strip()  # PRS gets lots of noise.
    icu_feature_terms["name"] = icu_feature_terms["name"].str.replace(" [0-9]{1,5}$", "", regex=True).str.strip()  # remove number at end
    icu_feature_terms["name"] = icu_feature_terms["name"].str.replace(" No$|Yes$|Do not know$", "",
                                                                      regex=True).str.strip()  # some noise when searching maybe
    icu_feature_terms["name"] = icu_feature_terms["name"].str.replace("Treatment/medication code |", "medication", regex=False)
    icu_feature_terms["name"] = icu_feature_terms["name"].str.replace("Non-cancer illness code, self-reported | ", "", regex=False).str.strip()
    # .str.replace(" gene","")
    # df_icu_feature_terms = icu_feature_terms.copy() # raw vals
    ## warning - feature values ma y be wrong for cases of "missing" placeholder featues" (will have same name
    icu_feature_terms = icu_feature_terms.drop_duplicates(subset=["name"]).reset_index(drop=True)  # some dupe terms, e.g. due to missing being removed

    # * ToDos - get scores per CUI of a name, and aggregate - so we can later tell if a feature/name has just 1 or multiple non-novel CUIs...
    # * could store using dict instead of lists.
    # In[29]:
    # EXCLUDE_TUIS_LIST = ["T079", "T093", "T094", "T095", "T170", "T204", "T201", "T065",
    #                      "T078", ]  # List of umls cui semtypes to exclude. Rough heuristic - not validated!
    ## more bad cases
    # "missing" -
    #         # CUI: C1551393, Name: Container status -  ;  TUI(s): T033
    #         # CUI: C1705492, Name: Missing Definition: Not existing - TUI(s): T080 # Legitimate use of T80
    ## T078 Idea or Concept  - e.g. Standard
    ## T080 Qualitative Concept - e.g. Standard (qualifier). Brown.  - Borderline , maybe drop?
    ### exogenous -> T082   Spatial Concept , T169  Functional Concept
    ## get CUIs from all entites in text - may add too much noise?
    ### It is easier and less work to just filter the output dataframe - although this may harm quality ofresults returned..
    ## "T204" - vertebrate, invertebrate
    ## "T201" - point in time, etc'. (Many of these findings are covered by normal diagnoses, e.g. ## "T204" -
    ## "T065" - Educational process of instructing , Teaching aspects
    # doc =nlp(icu_feature_terms)
    novel_cols_candidates_names = []
    no_entities_list = []
    novel_candidate_cuis = []
    novel_candidate_cuis_nomenclatures = []
    TUIs_list = []
    list_cui_kg_hits = []  # mark if (CUI level) novel or not (presencei n KG graph, for given target). Save # of hits (will alow filtering?)
    list_cui_definitions = []  # all text inc synonyms, definition for each cui - convenience for  doing expanded semantic similarity filtering.
    for f in icu_feature_terms["name"]:
        doc = nlp(f)
        linker = nlp.get_pipe("scispacy_linker")
        ## could use all or first k entities? Is this even top entity?
        if len(doc.ents) > 0:
            for j, entity in enumerate(doc.ents):
                #         if linker.kb.cui_to_entity[umls_ent[0]][3][0] not in EXCLUDE_TUIS_LIST: ## filter entities by TUIs, don't count the excluded. May overfilter!
                # TUIs_list.append(linker.kb.cui_to_entity[umls_ent[0]][3][0]) # new
                ### entity = doc.ents[0] # only get first entity
                # print(f"Entity #{j}:{entity}")

                list_feature_cuis = [i[0] for i in entity._.kb_ents]
                # print(list_feature_cuis)

                ## add tui filt
                s1 = len(list_feature_cuis)
                tui_filter_mask = [linker.kb.cui_to_entity[c][3][0] not in EXCLUDE_TUIS_LIST for c in list_feature_cuis]
                list_feature_cuis = list(compress(list_feature_cuis, tui_filter_mask))
                # print(list_feature_cuis)
                list_cuis_nomenclatures = [linker.kb.cui_to_entity[i[0]][1] for i in entity._.kb_ents]
                # linker = nlp.get_pipe("scispacy_linker") #ORIG
                list_cuis_nomenclatures = list(compress(list_cuis_nomenclatures, tui_filter_mask))

                num_candidates = len(list_feature_cuis)
                for c in list_feature_cuis:
                    TUIs_list.append(linker.kb.cui_to_entity[c][3][0])  # c[0]][3][0])
                if num_candidates > 0:
                    for umls_ent in entity._.kb_ents:
                        ent_name = linker.kb.cui_to_entity[umls_ent[0]][1]  # remove [1] to print all the cui data
                        # if ent_name not in novel_candidate_cuis_nomenclatures:
                        #     print(ent_name)
                    df_related = df_kg_sep.loc[(df_kg_sep["SUBJECT_CUI"].isin(list_feature_cuis)) | (df_kg_sep["OBJECT_CUI"].isin(list_feature_cuis))]

                    for cui in list_feature_cuis:
                        num_kg_hits = df_related.loc[(df_kg_sep["SUBJECT_CUI"] == cui) | (df_related["OBJECT_CUI"] == cui)].shape[0]
                        list_cui_kg_hits.append(num_kg_hits)
                        # list_cui_definitions.append(linker.kb.cui_to_entity[cui][4]) # cui definition only
                        list_cui_definitions.append(str(linker.kb.cui_to_entity[cui][1]) + ". " + str(linker.kb.cui_to_entity[cui][4]).replace("None",
                                                                                                                                               ""))  # append all cui definition, synonms, tui etc'

                    novel_cols_candidates_names.extend([f] * (num_candidates))
                    novel_candidate_cuis.extend(list_feature_cuis)
                    novel_candidate_cuis_nomenclatures.extend(list_cuis_nomenclatures)
                ## orig level of no wntity func:
                # else:
                #     no_entities_list.append(f)
                #     print(f"No Entity candidates for {f}")
                else:  # new, alt level, for 0 cands after filt
                    no_entities_list.append(f)
                    # print(f"No Entity candidates for {f}")
                    # novel_cols_candidates_names.append([f])
                    # novel_candidate_cuis.append([""])
                    # novel_candidate_cuis_nomenclatures.append([""])
        # print("-------------------------------")
        assert len(novel_cols_candidates_names) == len(novel_candidate_cuis)
        no_entities_list = list(set(no_entities_list))
    print(f"{len(no_entities_list)} - No Entity feats:{no_entities_list}")

    df_hits = pd.DataFrame({"feature_name": novel_cols_candidates_names,
                            "cui": novel_candidate_cuis  # + list_target_cuis,
                               , "cui_nomenclature": novel_candidate_cuis_nomenclatures,
                            "cui_def": list_cui_definitions,
                            "KG_Hits": list_cui_kg_hits,
                            "TUI": TUIs_list}).drop_duplicates()  ##
    ## merge with icu_feature_terms[["raw_name","name"]]
    s1 = df_hits.shape[0]
    df_hits = df_hits.loc[~df_hits["cui_nomenclature"].isin(REMOVE_CUI_TERMS_LIST)]
    print(s1 - df_hits.shape[0], "Rows of unwanted cuis dropped")
    for c in list_target_cuis:  ## manually append it hewre with some of the feature vals
        df_hits._append({"cui": c, "feature_name": "target", "cui_nomenclature": linker.kb.cui_to_entity[c][1]}, ignore_index=True)
    s1 = df_hits.shape[0]
    print(s1, "# rows pre semantic sim filt")
    ### semantic similarity - heuristic, remove poor scoring pairs (by semantic similarity).
    ### Note:Could also expandthis with the CUIsdescriptions
    df_hits = get_sentence_pairs_similarity(df=df_hits, col1="cui_nomenclature", col2="feature_name", filter=True, minFilterValue=0.99 * sem_similarity_threshhold_score
                                            , model2Name=None)
    # print(s1 - df_hits.shape[0], "rows dropped by cui/feature semantic sim")
    s1 = df_hits.shape[0]
    df_hits = get_sentence_pairs_similarity(df=df_hits, col1="cui_def", col2="feature_name", model2Name=None, filter=True,
                                            minFilterValue=sem_similarity_threshhold_score)  # 0.07
    print(s1 - df_hits.shape[0], "rows dropped by cui+Definition/feature semantic sim")
    ### TODO/DONE: Could do the mix, max, etc' using more stringently filtered (by sim score) concepts?
    ### Make pseudo col, instead of groupby join and needing more code. If decide not to use, then switch kg_hits_robust back to KG_Hits in subsequent code
    df_hits["kg_hits_robust"] = np.where(df_hits["sim_score"] >= 0.24, df_hits["KG_Hits"], 0)
    df_hits["feature_level_min_kg_hits"] = df_hits.groupby(["feature_name"])["kg_hits_robust"].transform("min")
    # df_hits["feature_level_sum_kg_hits"] = df_hits.groupby(["feature_name"])["KG_Hits"].transform("sum") # max
    # error - wrong length (groupedby, not transform(# df_hits["feature_level_sum_kg_hits"] = df_hits.groupby(["feature_name"]).apply(lambda df: sum(df.KG_Hits > 0)).values # sum of cases with hits
    df_hits["feature_level_sum_kg_hits"] = df_hits.groupby(["feature_name"])["kg_hits_robust"].transform(lambda x: sum(x > 0))
    # df_hits["feature_level_mean_kg_hits"] = df_hits.groupby(["feature_name"])["KG_Hits"].transform(lambda x: mean(x>0))  # todo: make work
    df_hits["v"] = df_hits["kg_hits_robust"].clip(upper=1)
    df_hits["feature_level_avg_kg_hits"] = df_hits.groupby(["feature_name"])["v"].transform("mean").round(1)
    df_hits.drop(columns=["v", "kg_hits_robust"], errors="ignore", inplace=True)
    # ## cases matching existing literature knowledge :
    # display(df_hits.query("KG_Hits>0").drop_duplicates("cui_nomenclature"))
    "keep features where at least 1 potentially novel cui = unmatched in known literature-KG:"
    # df_hits = df_hits.query("feature_level_min_kg_hits==0 & feature_level_avg_kg_hits<0.7")
    df_hits["cui"] = df_hits["cui"].astype(str)
    df_hits.drop_duplicates(inplace=True)
    print(df_hits[["feature_name", "cui"]].nunique())
    print("# KG Hits:")
    print(df_hits.query("KG_Hits>0")[["feature_name", "cui"]].nunique())
    print("# No KG Hits for feature:")
    print(df_hits.query("feature_level_min_kg_hits==0")[["feature_name", "cui"]].nunique())
    # ### optional another filter step - drop by top match being found?
    # * This may not help wit hcases of irrelevant matches.
    # * OPT/dangeorus
    # * IDEA: Take top match (by ner or our sim score) per entity, and if that is confident and a known link , then drop the feature.
    #     * could change KG_hits to >1 instead of >0 ?
    # In[35]:
    df_hits_top = df_hits.sort_values(["feature_name", "sim_score"], ascending=False).copy()  # sort with highest similarity feature first
    ## Also check for cases of rough exact match (cui =~ feature name, after mini cleaning
    df_hits_top["clean_featName"] = df_hits_top["feature_name"].str.lower().str.replace(punct_pattern, '').str.strip()
    df_hits_top["clean_cui"] = df_hits_top["cui_nomenclature"].str.lower().str.replace(punct_pattern, '').str.strip()

    # df_hits_top = df_hits_top.query(f"sim_score>={top_cutoff_simMin} & KG_Hits>={top_cutoff_kgHit}").drop_duplicates("feature_name", keep="first")
    df_hits_top = df_hits_top.loc[((df_hits_top["sim_score"]>=top_cutoff_simMin)\
                                   |(df_hits_top["clean_featName"]==df_hits_top["clean_cui"]))\
                                  & (df_hits_top["KG_Hits"]>=top_cutoff_kgHit)].drop_duplicates("feature_name", keep="first")

    ## drop these cases from candidates, as they are high confidence and seemingly known in lit!
    print(df_hits["feature_name"].nunique(), "# Feats before @1 filter")
    df_hits = df_hits.loc[~df_hits["feature_name"].isin(df_hits_top["feature_name"])]
    print(df_hits["feature_name"].nunique(), " Feats left after top 1 filter")
    ###### Add/Keep features that had 0 hits in the KG as additional candidate novels , for next stage of filtering
    # * Add pseudovals for cui
    # In[36]:
    ## TODO: Add in raw_name
    df_hits = pd.concat([df_hits, pd.DataFrame({"feature_name": no_entities_list,
                                                "KG_Hits": [0] * len(no_entities_list),
                                                "cui_nomenclature": [""] * len(no_entities_list),
                                                "cui_def": [""] * len(no_entities_list),
                                                "cui": [""] * len(no_entities_list),
                                                "sim_score": [1] * len(no_entities_list)})], ignore_index=True)
    for c in df_hits.select_dtypes("number").columns:
        # print(c)
        df_hits[c] = df_hits[c].fillna(0)


    # #### Rejoin with features metadata
    # * * Warning: missing feature proxies won't be idd correctly may replace the version of column without missings.
    #
    # In[42]:

    # In[43]:
    df_hits = df_hits.merge(icu_feature_terms.filter(['name', 'feature_importance', 'p_val', 'corr', "MutualInfoTarget",
                                               'raw_name', # restore
                                               'F.Split-Lift (y==1)',
                                               'F.Split-Support',  # 'F.Split-Target % Covered',
                                               'F.Split-Feature Split',
                                               ],axis=1).round(4),
                            left_on=["feature_name"], right_on="name", how="left", validate="m:1").drop(columns=["name", "TUI"], errors="ignore")
    # In[44]:
    # df_hits.loc[df_hits["sim_score"]<0.18].drop_duplicates("sim_score").sort_values("feature_name")#.head(12) sim_score

    # ### Very common/reoccurring nomenclatures - may be too broad
    # * could remove these based on counts, tf-idf, percentile distribution.
    # In[45]:
    # df_hits["cui_nomenclature"].value_counts().head(11)  # .index ## TFIDF - would be a good filter!
    # In[46]:
    # df_hits["cui_nomenclature"].value_counts().div(df_hits["feature_name"].nunique()).round(3)
    # # In[47]:
    # df_hits["cui_nomenclature"].value_counts().describe().round(2)


    # linker.kb.cui_to_entity[umls_ent[0]][3][0] # TUI
    # # linker.kb.cui_to_entity[umls_ent[0]][4] # definition
    # linker.kb.cui_to_entity[umls_ent[0]][1] # cui-name
    # # linker.kb.cui_to_entity[umls_ent[0]][2] # aliases
    # In[53]:
    # In[54]:
    # df_hits.loc[df_hits["cui"] == "C0028754"]  # "obesity"
    # In[55]:

    print("features with no linked entities in them:\n",no_entities_list)
    print(len(novel_cols_candidates_names), "# novel candidate cols")
    # print(f"{100*(len(novel_cols_candidates_names)/len(icu_feature_terms)):.2f}% candidates novel")
    # for gallstones - 52% (78) when using TF linker, vs 66% (100) using statistical linker
    # In[58]:
    print("novel candidates # CUIS:", len(novel_candidate_cuis))


    # #### Add seperate sim score between feature name (+- cui?) and the TARGET
    # * NOTE! This differs from the OTHER sim_score (which was used for filtering NEL results); this one is more for further
    # In[38]:
    df_temp = df_hits[["feature_name", "cui_nomenclature"]].copy()  # .head(10) # .drop_duplicates(subset=["feature_name"])
    df_temp["target_name"] = TARGET_NAME
    df_hits["sim_score_target_feat"] = get_sentence_pairs_similarity(df=df_temp.copy(), col1="target_name", col2="feature_name", filter=False,
                                                                     return_score_only=True)
    # df_hits["sim_score_target_cui"] = get_sentence_pairs_similarity(df=df_temp.copy(),col1="cui_nomenclature",col2="target_name",filter=False,return_score_only=True)
    df_hits["sim_score_target_cui"] = get_sentence_pairs_similarity(df=df_temp.copy(), col1="target_name", col2="cui_nomenclature",
                                                                    model2Name=None,
                                                                    filter=False, return_score_only=True)

    # In[59]:
    for c in list_target_cuis:
        print(linker.kb.cui_to_entity[c][1])

    # ## Graph connectivity metric

    df_path_lengths = get_kg_connections(df_hits, df_kg, list_target_cuis)
    print(df_path_lengths.shape,"df_path_lengths")
    print(df_hits.shape, "df_hits")
    # In[63]:
    # df_hits = df_hits.merge(df_path_lengths.drop(columns=["node_degree"],
    #                                              errors="ignore").drop_duplicates(), on="cui", how="left") #added row changed, drop dupes
    df_hits = df_hits.merge(df_path_lengths.drop(columns=["node_degree"], errors="ignore").drop_duplicates(), on="cui", how="left")
    # df_hits.drop_duplicates(subset=["cui"]).query("shortest_path_length<50").shortest_path_length.describe().round(1)
    # In[65]:

    # ### Save output report
    # * Saves highly filtered (heuristic) candiadtes, eg.g with no kg hits. (+- path length?)
    # In[66]:
    if SAVE_OUTPUTS:

        df_hits["cui"] = df_hits["cui"].astype(str)
        print(CANDIDATE_NOVEL_CUIS_FILEPATH)
        df_temp = df_hits.query("(KG_Hits==0) & (feature_level_min_kg_hits<=3)").drop_duplicates() #  & feature_level_avg_kg_hits<0.6
        print(df_temp.select_dtypes("O").nunique())
        # df_temp = df_hits.query("shortest_path_length>2 & feature_level_avg_kg_hits<=0.75")
        df_temp.to_csv(CANDIDATE_NOVEL_CUIS_FILEPATH, index=False)
        display(df_temp)
    else:
        return df_hits


## slim in that some parts disabled/commented out (semantic similarity filter etc'): renamed from link_kg_concepts       
def link_kg_concepts_slim(FEATURES_REPORT_PATH, CANDIDATE_NOVEL_CUIS_FILEPATH, TARGET_NAME, additional_target_cui_terms_list=[], SAVE_OUTPUTS = False, MIN_EVIDENCE_FILTER = 2,
                     REMOVE_CUI_TERMS_LIST=['Prieto syndrome', "Polarized Reflectance Spectroscopy",
                                            # mistaken extraction from PRS - drop it for now for cleanliness
                                            'Standard (qualifier)', 'Standard base excess calculation technique',
                                            'Standard of Care', 'Spatial Frequency', 'Disease',
                                            'Statistical Frequency', 'Kind of quantity - Frequency', 'Concentration measurement',
                                            'Concentration measurement', 'Illness (finding)', 'Concentration Ratio',
                                            "Special", "ActInformationPrivacyReason <operations>", "Left sided", "Left", "Right",
                                            "Table Cell Horizontal Align - left", "Query Quantity Unit - Records", "Up",
                                            "Qualification",
                                            "Visit", "Total", "Participant", "Overall", "Right sided", "Left sided",
                                            "Take", "Percent (qualifier value)","Population Group",
                                            "Diagnosis","Coding","Code",
                                            "Average" , "Comparison" , "Lost" , "Yes - Presence findings", 
                                            "Pharmaceutical Preparations","Physicians","Mother (person)","Father (person)","Severe (severity modifier)",
                                            ]
                     , input_kg_path="../SemMed/predications.parquet", EXCLUDE_TUIS_LIST = ["T079", "T093", "T094", "T095", "T170", "T204", "T201", "T065",
                         "T078", ], sem_similarity_threshhold_score=0.15,
                     # top_cutoff_simMin = 0.39,top_cutoff_kgHit = 2,
                     FAST=False,
                    return_df=True
                    , get_simplePathLengths = True):
    # global nlp, df_kg_sep, df_hits, G
    global df_kg_sep, df_hits ## maybe disable this..

    # ### Load processed semmed db
    # * ths version has 1 row per triple.
    # * `Count` is the number of unique PMIDs the triple has appeared in
    df_kg = pd.read_parquet(input_kg_path)

    # ### Filtered version

    df_kg = df_kg.loc[df_kg["pair_counts"] >= MIN_EVIDENCE_FILTER].reset_index(drop=True).copy()
    df_kg.drop(columns=["pair_counts","counts"], inplace=True, errors="ignore")
    print("After filtering KG min count")
    for c in df_kg.select_dtypes("category").columns:
        # remove unobserved categories, in new filtered data
        df_kg[c] = df_kg[c].cat.remove_unused_categories()
    print(df_kg.shape[0])  # 4M

    # # !pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz
    # # !pip install 'spacy[transformers]'
    # ```
    # there's also lg, and transformer based
    if FAST:
        nlp = spacy.load("en_core_sci_sm")
    else:
        nlp = spacy.load("en_core_sci_lg")
        # nlp = spacy.load("en_core_sci_scibert") # error in later spacy version?
    
    # This line takes a while, because we have to download ~1GB of data
    # and load a large JSON file (the knowledge base). 
    
    # Add the abbreviation pipe to the spacy pipeline. (if using resolve_abbreviations )
    nlp.add_pipe("abbreviation_detector")
    nlp.add_pipe("scispacy_linker",
                 config={"resolve_abbreviations": True,
                         "linker_name": "umls",
                         "max_entities_per_mention": 3  # 3, #6, #4, #5
                     , "threshold": 0.88  ## default is 0.8, paper mentions 0.99 as thresh
                         })

    print("TARGET_NAME (For Entity-KG linking", TARGET_NAME)
    # new func
    # entity = get_multiple_nel(TARGET_NAME,nlp=nlp)
    entity = test_nel(TARGET_NAME,nlp=nlp)
    # #### Target terms - may manually change
    ## list of extracted CUIs meaning sepsis. Note that not all correct even here and with threshhold
    list_target_cuis = [i[0] for i in entity._.kb_ents]
    if len(additional_target_cui_terms_list) > 0:
        list_target_cuis = list(set(list_target_cuis + additional_target_cui_terms_list))
    print("list_target_cuis",list_target_cuis)
    # #### get subset of KG with target in pairs
    # (not 100% sure if ideal, but will save time when comparing features )
    df_kg_sep = df_kg.loc[(df_kg["SUBJECT_CUI"].isin(list_target_cuis)) | (df_kg["OBJECT_CUI"].isin(list_target_cuis))].copy()
    df_kg_sep.drop_duplicates(['SUBJECT_CUI', 'SUBJECT_NAME', 'OBJECT_CUI', 'OBJECT_NAME'],
                              inplace=True)  # ignore predicate type for this filter table
    # In[28]:
    ## sample set of terms from gallstone prediction + filtered
    icu_feature_terms = pd.read_csv(FEATURES_REPORT_PATH)

    if FAST:
        icu_feature_terms = icu_feature_terms.loc[icu_feature_terms["feature_importance"] > 0].head(25)
    icu_feature_terms = wrangle_df_icu(icu_feature_terms)
    ##############################################################
    ##ORIG code deeted,  , single etity per text - deleted
    ##...
    ##############################################################
    ## slt/new - for multiple entities per text
    
    # Initialize lists to collect data
    novel_cols_candidates_names = []
    no_entities_list = []
    novel_candidate_cuis = []
    novel_candidate_cuis_nomenclatures = []
    TUIs_list = []
    list_cui_kg_hits = []
    list_cui_definitions = []
    
    # Iterate over each feature name
    for f in icu_feature_terms["name"]:
        doc = nlp(f)
        linker = nlp.get_pipe("scispacy_linker")
        
        # Check if any entities are found in the feature name
        if doc.ents:
            # Flag to check if at least one CUI passes the TUI filter
            feature_has_valid_cui = False
            
            # Iterate over all entities in the document
            for entity in doc.ents:
                # Extract all CUIs for the current entity
                list_feature_cuis = [cui[0] for cui in entity._.kb_ents] if entity._.kb_ents else []
                
                if not list_feature_cuis:
                    continue  # Skip entities without CUIs
                
                # Apply TUI filter to exclude certain CUIs
                tui_filter_mask = [
                    linker.kb.cui_to_entity[c][3][0] not in EXCLUDE_TUIS_LIST 
                    for c in list_feature_cuis
                ]
                filtered_cuis = list(compress(list_feature_cuis, tui_filter_mask))
                
                # If no CUIs pass the TUI filter, skip to the next entity
                if not filtered_cuis:
                    continue
                
                # Mark that this feature has at least one valid CUI
                feature_has_valid_cui = True
                
                # Extract nomenclatures for the filtered CUIs
                list_cuis_nomenclatures = [
                    linker.kb.cui_to_entity[c][1] for c in filtered_cuis
                ]
                
                # Append TUIs for the filtered CUIs
                TUIs_list.extend([linker.kb.cui_to_entity[c][3][0] for c in filtered_cuis])
                
                # Filter the knowledge graph for related CUIs
                df_related = df_kg_sep.loc[
                    df_kg_sep["SUBJECT_CUI"].isin(filtered_cuis) | 
                    df_kg_sep["OBJECT_CUI"].isin(filtered_cuis)
                ]
                
                # Iterate over each filtered CUI
                for cui in filtered_cuis:
                    # Count KG hits for the current CUI
                    ## add .loc
                    num_kg_hits = df_related.loc[
                        (df_kg_sep["SUBJECT_CUI"] == cui) | 
                        (df_related["OBJECT_CUI"] == cui)
                    ].shape[0]
                    list_cui_kg_hits.append(num_kg_hits)
                    
                    # Collect definitions and synonyms for the current CUI
                    cui_definition = linker.kb.cui_to_entity[cui][4]
                    cui_def_text = f"{linker.kb.cui_to_entity[cui][1]}. {cui_definition}" if cui_definition else linker.kb.cui_to_entity[cui][1]
                    list_cui_definitions.append(cui_def_text)
                
                # Extend lists with the current feature name and associated CUIs
                novel_cols_candidates_names.extend([f] * len(filtered_cuis))
                novel_candidate_cuis.extend(filtered_cuis)
                novel_candidate_cuis_nomenclatures.extend(list_cuis_nomenclatures)
            
            # If no valid CUIs were found for any entity in the feature, add to no_entities_list
            if not feature_has_valid_cui:
                no_entities_list.append(f)
        else:
            # If no entities are found in the feature name, add to no_entities_list
            no_entities_list.append(f)
    
    # After processing all feature names, remove duplicates from no_entities_list
    no_entities_list = list(set(no_entities_list))
    print(f"{len(no_entities_list)}:\n No Entity feats: {no_entities_list}")
    
    # Create the DataFrame with collected data
    df_hits = pd.DataFrame({
        "feature_name": novel_cols_candidates_names,
        "cui": novel_candidate_cuis,
        "cui_nomenclature": novel_candidate_cuis_nomenclatures,
        "cui_def": list_cui_definitions,
        "KG_Hits": list_cui_kg_hits,
        "TUI": TUIs_list
    }).drop_duplicates()

    ##############################################################
    ## merge with icu_feature_terms[["raw_name","name"]]
    s1 = df_hits.shape[0]
    df_hits = df_hits.loc[~df_hits["cui_nomenclature"].isin(REMOVE_CUI_TERMS_LIST)]
    print(s1 - df_hits.shape[0], "Rows of unwanted cuis dropped")
    for c in list_target_cuis:  ## manually append it hewre with some of the feature vals
        df_hits._append({"cui": c, "feature_name": "target", "cui_nomenclature": linker.kb.cui_to_entity[c][1]}, ignore_index=True)
    s1 = df_hits.shape[0]
    # print(s1, "# rows pre semantic sim filt")
    ### semantic similarity - heuristic, remove poor scoring pairs (by semantic similarity).
    # ### Note:Could also expandthis with the CUIsdescriptions
    # df_hits = get_sentence_pairs_similarity(df=df_hits, col1="cui_nomenclature", col2="feature_name", filter=True, minFilterValue=0.98 * sem_similarity_threshhold_score
    #                                         , model2Name=None)

    # df_hits.drop(columns=["v", "kg_hits_robust"], errors="ignore", inplace=True)
    # "keep features where at least 1 potentially novel cui = unmatched in known literature-KG:"
    # # df_hits = df_hits.query("feature_level_min_kg_hits==0 & feature_level_avg_kg_hits<0.7")
    df_hits["cui"] = df_hits["cui"].astype(str)
    df_hits.drop_duplicates(inplace=True)
    print(df_hits[["feature_name", "cui"]].nunique())
    print("# KG Hits:")
    print(df_hits.query("KG_Hits>0")[["feature_name", "cui"]].nunique())
    ## remove top hits filter 
    ###### Add/Keep features that had 0 hits in the KG as additional candidate novels , for next stage of filtering
    # * Add pseudovals for cui

    ## TODO: Add in raw_name
    df_hits = pd.concat([df_hits, pd.DataFrame({"feature_name": no_entities_list,
                                                "KG_Hits": [0] * len(no_entities_list),
                                                "cui_nomenclature": [""] * len(no_entities_list),
                                                "cui_def": [""] * len(no_entities_list),
                                                "cui": [""] * len(no_entities_list),
                                                "sim_score": [1] * len(no_entities_list)})], ignore_index=True)
    
    ## add:
    """Not enough! Doesn't handle empty features being added if present? """
    df_hits.drop_duplicates(subset=["feature_name","cui","KG_Hits"],inplace=True) ## added
    
    for c in df_hits.select_dtypes("number").columns:
        # print(c)
        df_hits[c] = df_hits[c].fillna(0)

    # #### Rejoin with features metadata
    # * * Warning: missing feature proxies won't be idd correctly may replace the version of column without missings.
    #
    # In[42]:

    # In[43]:
    df_hits = df_hits.merge(icu_feature_terms[['name', 'feature_importance', 'p_val', 'corr', "MutualInfoTarget",
                                               'raw_name', # restore
                                               ]].round(3),
                            left_on=["feature_name"], right_on="name", how="left", validate="m:1").drop(columns=["name", "TUI"], errors="ignore")

    # ### Very common/reoccurring nomenclatures - may be too broad
    # * could remove these based on counts, tf-idf, percentile distribution.

    print(f"# {len(no_entities_list)}features with no linked entities in them:\n")
    print(len(novel_cols_candidates_names), "# novel candidate")
    
    # print(f"{100*(len(novel_cols_candidates_names)/len(icu_feature_terms)):.2f}% candidates novel")
    # for gallstones - 52% (78) when using TF linker, vs 66% (100) using statistical linker
    # In[58]:
    print("novel candidates # CUIS:", len(novel_candidate_cuis))


    # # #### Add seperate sim score between feature name (+- cui?) and the TARGET
    # # * NOTE! This differs from the OTHER sim_score (which was used for filtering NEL results); this one is more for further
    # # In[38]:
    # df_temp = df_hits[["feature_name", "cui_nomenclature"]].copy()  # .head(10) # .drop_duplicates(subset=["feature_name"])
    # df_temp["target_name"] = TARGET_NAME
    # df_hits["sim_score_target_feat"] = get_sentence_pairs_similarity(df=df_temp.copy(), col1="target_name", col2="feature_name", filter=False,
    #                                                                  return_score_only=True)
    # # df_hits["sim_score_target_cui"] = get_sentence_pairs_similarity(df=df_temp.copy(),col1="cui_nomenclature",col2="target_name",filter=False,return_score_only=True)
    # df_hits["sim_score_target_cui"] = get_sentence_pairs_similarity(df=df_temp.copy(), col1="target_name", col2="cui_nomenclature",
    #                                                                 model2Name=None,
    #                                                                 filter=False, return_score_only=True)

    # In[59]:
    for c in list_target_cuis:
        print(linker.kb.cui_to_entity[c][1])

    # ## Graph connectivity metric
    df_path_lengths = get_kg_connections(df_hits=df_hits, df_kg=df_kg, list_target_cuis=list_target_cuis, get_simplePathLengths = get_simplePathLengths)
    print(df_path_lengths.shape,"df_path_lengths")
    print(df_hits.shape, "df_hits")
    # In[63]:
    df_hits = df_hits.merge(df_path_lengths.drop(columns=["node_degree"], errors="ignore").drop_duplicates(), on="cui", how="left")

    # ### Save output report
    # * Saves highly filtered (heuristic) candiadtes, eg.g with no kg hits. (+- path length?)
    # In[66]:
    if SAVE_OUTPUTS:
        df_hits["cui"] = df_hits["cui"].astype(str)
        print(CANDIDATE_NOVEL_CUIS_FILEPATH)

    if return_df:
        return df_hits



# # Function to get predictions from MedRAG # get_predictions
def get_predictions_from_medrag(medrag, results, question_key, options_key, snippets=None,return_snippets=False,K=25):
    predictions = []
    explanations = []
    if return_snippets: snippets_list = []
    for index, row in enumerate(tqdm((results), total=len(results), desc="Processing rows")):
    # for index, row in enumerate(results):
        question = row[question_key]
        options = row[options_key]
        snippet = snippets[index] if snippets is not None else None
        answer, retrieved_snippets, _ = medrag.answer(question=question, options=options, snippets=snippet,
                                                      k=K)# if FAST_RUN else 30)
        try:
            if not answer.endswith("}"): ## what about " at end?
                if "}" not in answer:
                    answer += "}" 
            # Use regex to replace two or more consecutive closing curly braces with just one
            answer = re.sub(r'\}+', '}', answer)

            # Fix missing commas between key-value pairs
            answer = re.sub(r'(")\s*(\n\s*)?(?="[\w_]+":)', r'\1,\2,', answer)

            # # replace all occurrences of single quote with double quote in the JSON string s and in the latter case will not replace escaped single-quotes.
            # p = re.compile('(?<!\\\\)\'')
            # answer = p.sub('\"', answer)
             
            # json_ans = json.loads(re.search(r'{.+}', answer, re.IGNORECASE).group(0)) # prev
            json_ans = json.loads(re.search(r'{.*?}', answer,  re.DOTALL | re.IGNORECASE).group(0))
            pred = json_ans.get('answer_choice', None)
            predictions.append(pred)
            explanations.append(json_ans.get('step_by_step_thinking', None))
        except Exception as e:
            # replace all occurrences of single quote with double quote in the JSON string s and in the latter case will not replace escaped single-quotes.
            p = re.compile('(?<!\\\\)\'')
            answer = p.sub('\"', answer)
            try:
                json_ans = json.loads(re.search(r'{.*?}', answer,  re.DOTALL | re.IGNORECASE).group(0))
                pred = json_ans.get('answer_choice', None)
                predictions.append(pred)
                explanations.append(json_ans.get('step_by_step_thinking', None))
            except:
                predictions.append(None)
                explanations.append( None)
                print(f"Failed to parse answer: {e}\n{answer}")
        if return_snippets: snippets_list.append(retrieved_snippets)
    if return_snippets:
        return predictions,explanations,snippets_list
    else:
        return predictions,explanations


def generate_boring_prompts(data: pd.DataFrame):
    results = []
    for index, row in data.iterrows():
        # Clean and prepare data
        feature_name_raw = row['raw_name'].replace('_', '-').strip()

        feature_name_clean = row['feature_name']
        
        target = row['Target']
        corr = row['corr']
        # Clean target string (keeping 'OR' in)
        target_clean = target.replace('(', '').replace(')', '').strip()
        # Clean feature names ? 
        feature_name_clean = feature_name_clean.replace('_', ' ').strip()
      
        # Determine direction of effect
        if corr > 0:
            direction = 'positive'
        elif corr < 0:
            direction = 'negative'
        else:
            direction = 'neutral'
        
        boring_question = (
            f"Evaluate the feature '{row.raw_name}' in relation to predicting the target disease: '{target}'. The feature has {direction} correlation with the target disease). \n "
            f"Is the association between the feature '{feature_name_clean}' (raw: '{feature_name_raw}') "
            f"and '{target_clean}', already known, boring, obvious (trivial, e.g. is a known risk / protective factor) or part of known risk factors (or related pathways or mechanisms)?"
            # f"Does this feature provide new insights or contradict established understanding? Is it Novel?"
        )
        boring_options = {

            "A": "Yes, it is well known or explained by existing knowledge.", #  by existing literature # surprising, not well-documented,
            "B": "No, it is not well known or explained by existing knowledge." # or easily explained or trivially explainable." #  by existing, related factors or pathways/mechanisms/biology
        }

        
        # Store the prompts
        results.append({
            'feature':  row['feature_name'],
            # 'prompt': prompt
            'raw_feature': row['raw_name'],
            "target": row['Target'],
            "boring_prompt":boring_question,
            "boring_options":boring_options,
        })
        
    return results

# boring_prompts = generate_boring_prompts(df.head(50))


import re
from difflib import get_close_matches
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

def clean_text(text):
    # Remove text inside parentheses along with the parentheses
    cleaned = re.sub(r'\(.*?\)', '', text)
    # Remove "None of the above" (case-insensitive)
    cleaned = re.sub(r'(?i)None of the above', '', cleaned)
    # Remove single-digit numbers as standalone words
    cleaned = re.sub(r'\b\d\b', '', cleaned)
    # Strip whitespace
    cleaned = cleaned.strip()
    return cleaned

def deduplicate_texts(
    texts, 
    use_difflib=True, 
    string_cutoff=0.98,
    distance_threshold=0.5
):
    """
    Deduplicate texts using a two-step process (optional difflib first, then semantic clustering),
    while preserving the original texts. 
    
    Steps:
    1. Clean texts with regex-based rules for clustering only (originals are preserved).
    2. (Optional) Use difflib-based merging on cleaned texts to quickly reduce near-duplicates.
    3. Use semantic clustering on the reduced set to further deduplicate.
    4. Return original texts as final output.

    Parameters:
        texts (list of str): The input texts.
        use_difflib (bool): Whether to use difflib-based merging before semantic clustering.
        string_cutoff (float): Cutoff for difflib.get_close_matches.
        distance_threshold (float): Distance threshold for AgglomerativeClustering.
        
    Returns:
        list of str: Deduplicated texts (original, unmodified).
    >>>
        texts = [
        'Ethnic background British', 'Ethnic background Mixed',
        'Ethnic background',
        'Doctor diagnosed COPD (chronic obstructive pulmonary disease)',
        'Illnesses of mother 3 None (group 1)',
        'Ethnic background White and Black Caribbean',
        'Illnesses of siblings 1 Severe depression', 'osteoarthritis',
        'Illnesses of siblings 1 Do not know (group 1)',
        'Illnesses of siblings 2 Do not know (group 2)',
        'Illnesses of mother 3 Do not know (group 2)',
        'Ethnic background Indian', '(BD) bipolar disorder genetic risk',
        'Ethnic background African', 'Ethnic background Chinese',
        'Ever addicted to any substance or behaviour',
        'Ever addicted to illicit or recreational drugs',
        'Illnesses of siblings', 'Tobacco use disorder', 'sclerosis',
        'Illnesses of siblings 2 Do not know (group 1)',
        'Essential hypertension',
        'Illnesses of siblings 1 Do not know (group 2)',
        'Illnesses of siblings 3 Do not know (group 1)',
        'Illnesses of siblings 3 Do not know (group 2)',
    ]
    final_texts = deduplicate_texts(texts, use_difflib=True, string_cutoff=0.95, distance_threshold=1.0)
    """
    print(len(texts), "# inputs")

    # Pre-clean all texts (for clustering purposes only, no changes to originals)
    cleaned_texts = [clean_text(t) for t in texts]

    # Optional difflib-based merging step
    if use_difflib:
        print("# Applying string-level merging (difflib) first...")
        reduced_originals = []
        reduced_cleaned = []

        for orig, ctext in zip(texts, cleaned_texts):
            match = get_close_matches(ctext, reduced_cleaned, n=2, cutoff=string_cutoff)
            if not match:
                # No match, so add this text
                reduced_originals.append(orig)
                reduced_cleaned.append(ctext)
            else:
                # If there's a match, find its index
                existing_clean = match[0]
                idx = reduced_cleaned.index(existing_clean)
                existing_orig = reduced_originals[idx]

                # Pick the shortest original text
                if len(orig) < len(existing_orig):
                    reduced_originals[idx] = orig
                    reduced_cleaned[idx] = ctext
        texts_for_semantic = reduced_originals
    else:
        texts_for_semantic = texts

    # Re-clean for semantic step (in case we skipped difflib or changed sets)
    cleaned_for_semantic = [clean_text(t) for t in texts_for_semantic]
    
    model = SentenceTransformer("FremyCompany/BioLORD-2023") #'sentence-transformers/all-MiniLM-L12-v2'
    embeddings = model.encode(cleaned_for_semantic, show_progress_bar=False)

    # print("# Performing semantic clustering...")
    clusterer = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=distance_threshold, 
        linkage='average', 
        metric='euclidean'
    )
    labels = clusterer.fit_predict(embeddings)

    # print("# Selecting shortest representative from each semantic cluster...")
    cluster_map = {}
    for label, orig_text in zip(labels, texts_for_semantic):
        if label not in cluster_map or len(orig_text) < len(cluster_map[label]):
            cluster_map[label] = orig_text

    final = list(cluster_map.values())
    print(len(final), "# Deduplicated.")

    return final