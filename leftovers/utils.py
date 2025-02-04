from IPython.core.display_functions import display
from arfs.preprocessing import OrdinalEncoderPandas
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd

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
                       criterion='gini', min_support_pct=0.05, min_support=15, focus_on_lift=True,
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


def pivot_long_col_diags(df, K_value=5_00, value_col="Z-score-Age_at_diagnosis", codeColName="code",
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
        df[col] = df[col].str.replace("None of the above", "None")
        split_cols = df[col].str.split('|', expand=True)
        if prefix:
            split_cols = split_cols.add_prefix(f'{col}_')
        df = df.drop(columns=[col]).join(split_cols)
    return df


def cv_eval(X, y, n_cv=4, model=None):
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
                 K_IPW_RATIO=4, X=None, get_ipw_shap=False):
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
    m = X.filter(text_cols_list, axis=1).nunique() > 70  # 20
    list_features_high_cardinality = list(X.filter(text_cols_list, axis=1).nunique()[m].index)
    # print(list_features_high_cardinality)
    categorical_cols = X[text_cols_list].columns.difference(list_features_high_cardinality).to_list()

    # Define categorical, numerical and text columns
    # categorical_cols = categorical_cols  # Update with your actual categorical columns
    text_cols = list_features_high_cardinality  # Update with your actual text columns
    numerical_cols = X.select_dtypes(include=['number', "boolean"]).columns.difference(categorical_cols + text_cols).to_list()
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

def get_cb_shap(X, y, categorical_cols, text_cols, return_shapVal=True):
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


    # Calculate top 10 feat interactions
    feat_interactions = clf_model2.get_feature_importance(type=EFstrType.Interaction, prettified=True)

    top_interactions = feat_interactions[:10].round(2)
    summary = {X2.columns.get_loc(c):c for c in X2.columns}

    top_interactions['First Feature Index'] = top_interactions['First Feature Index'].apply(lambda x: summary[x])
    top_interactions['Second Feature Index'] = top_interactions['Second Feature Index'].apply(lambda x: summary[x])
    top_interactions.columns = ['First Feature', 'Second Feature', 'Interaction']
    print(top_interactions)
    ## overall feat imp
    display(shap.plots.beeswarm(shap_values, max_display=30))  # ,log_scale=True
    if return_shapVal:
        shap_values = np.abs(shap_values.values).mean(0)  # get abs val
        return shap_values



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
