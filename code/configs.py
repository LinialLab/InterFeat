# # Example configuration dictionary - gallstones
config_gall = {
    "TARGET_CODES_LIST" :("K80", "K81", "K82"),  ## Cholelithiasis = gallstones 
    "do_IPW": True,
    "do_boruta_fs" : True,#False,
    "K_IPW_RATIO":9,
    'targets': [
        "Cholelithiasis",
        "Gallstone",
        "Gallbladder disease",
        "cholecystitis",
        "Cholangitis"
        ],
    "FEATURES_REPORT_PATH":"gallstone_ipw_broad_feature_report.csv",
    'QUERY_CANDIDATES_FILE': 'candidate_novel_cuis_chol.csv',
    "CANDIDATE_NOVEL_CUIS_FILEPATH": "candidate_novel_cuis_chol.csv", # output path for features/util # same as QUERY_CANDIDATES_FILE ? 
    'DO_MINI_COMBINED_PATH_FILT': False
    ,'SAVE_OUTPUTS': True,
    "TARGET_NAME" :"GALLSTONES, Cholelithiasis",
    "additional_target_cui_terms_list" : ["C0008325", "C0008311"],
    
    'OUTPUT_RES_PREFIX': 'gallstone_',
    # 'full_results_filename': 'candidates_search_results.csv', # OUTPUT_RES_PREFIX is added atop this 
    # 'filtered_results_filename': 'review_interesting_candidates_results.csv',# OUTPUT_RES_PREFIX is added atop this 
    # # 'cooc_count_filter_val': 10,  'SIGNIFICANT_PVAL': 0.3,
    # "DIAG_TIDY_TABLE_PATH" : "../../df_ukbb_aux_tidy.parquet", #"../ukbb-hack/df_diag_tidy.parquet",
    }

config_gallbladder2 = {
    "TARGET_CODES_LIST" :("K80",),  ## Cholelithiasis = Gallbladder disease
    "do_IPW": True,
    "do_boruta_fs" : False,
    "K_IPW_RATIO":10,
    'targets': [
        "Cholelithiasis",
        # "Gallstone",
        "Gallbladder disease",
        # "cholecystitis",
        # "Cholangitis"
        ],
    "FEATURES_REPORT_PATH":"gallbladder_ipw_broad_feature_report.csv",
    'QUERY_CANDIDATES_FILE': 'candidate_novel_cuis_gall.csv',
    "CANDIDATE_NOVEL_CUIS_FILEPATH": "candidate_novel_cuis_gall.csv", # output path for features/util # same as QUERY_CANDIDATES_FILE ? 
    'DO_MINI_COMBINED_PATH_FILT': False
    ,'SAVE_OUTPUTS': True,
    "TARGET_NAME" :"Cholelithiasis, Gallbladder",
    # "additional_target_cui_terms_list" : ["C0008325", "C0008311"],
    
    'OUTPUT_RES_PREFIX': 'gallbladder_',
    # 'full_results_filename': 'candidates_search_results.csv', # OUTPUT_RES_PREFIX is added atop this 
    # 'filtered_results_filename': 'review_interesting_candidates_results.csv',# OUTPUT_RES_PREFIX is added atop this 
    # # 'cooc_count_filter_val': 10,  'SIGNIFICANT_PVAL': 0.3,
    # "DIAG_TIDY_TABLE_PATH" : "../../df_ukbb_aux_tidy.parquet", #"../ukbb-hack/df_diag_tidy.parquet",
    }



config_asthma = {
    "TARGET_CODES_LIST": ("J45"),
    "do_IPW": True,
    "do_boruta_fs": False,
    "K_IPW_RATIO": 9,
    "targets": ["asthma"],
    "FEATURES_REPORT_PATH": "asthma_feature_report.csv",
    "QUERY_CANDIDATES_FILE": "candidate_novel_asthma.csv",
    "CANDIDATE_NOVEL_CUIS_FILEPATH": "candidate_novel_asthma.csv",
    "TARGET_NAME": "Asthma",
    # "additional_target_cui_terms_list": ["C5139492"],
    "OUTPUT_RES_PREFIX": "asthma_",
}

config_psoriasis = {
    "TARGET_CODES_LIST": ("L40","L21.9"),
    "do_IPW": True,
    "do_boruta_fs": False,
    "K_IPW_RATIO": 11,
    "targets": ["psoriasis","Seborrheic dermatitis"],
    "FEATURES_REPORT_PATH": "psoriasis_feature_report.csv",
    "QUERY_CANDIDATES_FILE": "candidate_novel_psoriasis.csv",
    "CANDIDATE_NOVEL_CUIS_FILEPATH": "candidate_novel_psoriasis.csv",
    "TARGET_NAME": "Psoriasis",
    "OUTPUT_RES_PREFIX": "Psoriasis_",
}


config_celiac = {
    "TARGET_CODES_LIST": ("K90",),
    "do_IPW": True,
    "do_boruta_fs": False,
    "K_IPW_RATIO": 9,
    "targets": ["celiac AND disease", "Coeliac", "gluten AND allergy", "gluten AND enteropathy"],
    "FEATURES_REPORT_PATH": "celiac_feature_report.csv",
    "QUERY_CANDIDATES_FILE": "candidate_novel_celiac.csv",
    "CANDIDATE_NOVEL_CUIS_FILEPATH": "candidate_novel_celiac.csv",
    "TARGET_NAME": "Coeliac disease",
    "additional_target_cui_terms_list": ["C5139492"],
    "OUTPUT_RES_PREFIX": "celiac_",
}
config_gout = {
    "TARGET_CODES_LIST": ("M10", "M1A"),
    "do_IPW": True,
    "do_boruta_fs": False,
    "K_IPW_RATIO": 9,
    "targets": ["Gout"],
    "FEATURES_REPORT_PATH": "gout_feature_report.csv",
    "QUERY_CANDIDATES_FILE": "candidate_novel_cuis_gout.csv",
    "CANDIDATE_NOVEL_CUIS_FILEPATH": "candidate_novel_cuis_gout.csv",
    "TARGET_NAME": "Gout",
    "additional_target_cui_terms_list": [],
    "OUTPUT_RES_PREFIX": "gout_",
}
# config_ms = {
#     "TARGET_CODES_LIST": ("G35",),
#     "do_IPW": True,
#     "do_boruta_fs": False,
#     "K_IPW_RATIO": 9,
#     "targets": ["Multiple sclerosis"],
#     "FEATURES_REPORT_PATH": "MS_feature_report.csv",
#     "QUERY_CANDIDATES_FILE": "candidate_novel_MS.csv",
#     "CANDIDATE_NOVEL_CUIS_FILEPATH": "candidate_novel_MS.csv",
#     "TARGET_NAME": "Multiple Sclerosis",
#     "additional_target_cui_terms_list": [],
#     "OUTPUT_RES_PREFIX": "ms_",
# }
config_spine = {
    "TARGET_CODES_LIST": ("M51.3", "M51.2", "M51.0", "M51.1", "M51.3", "M47"),
    "do_IPW": True,
    "do_boruta_fs": False,
    "K_IPW_RATIO": 9,
    "targets": ["disc displacement", "disc degeneration", "disc disorder", "back AND pain", "spine disease", "Spondylosis"],
    "FEATURES_REPORT_PATH": "spine_feature_report.csv",
    "QUERY_CANDIDATES_FILE": "candidate_novel_spine.csv",
    "CANDIDATE_NOVEL_CUIS_FILEPATH": "candidate_novel_spine.csv",
    "TARGET_NAME": "Spine degeneration",
    "additional_target_cui_terms_list": ["C0158266", "C0850918", "C0021818", "C0038019", "C0158252", "C0423673", "C0024031"],
    "OUTPUT_RES_PREFIX": "spine_",
}
config_oesophagus = {
    "TARGET_CODES_LIST": ("C15",),
    "do_IPW": True,
    "do_boruta_fs": False,
    "K_IPW_RATIO": 9,
    "targets": ["oesophagus cancer"],
    "FEATURES_REPORT_PATH": "oesophagus_feature_report.csv",
    "QUERY_CANDIDATES_FILE": "candidate_novel_oesophagus.csv",
    "CANDIDATE_NOVEL_CUIS_FILEPATH": "candidate_novel_oesophagus.csv",
    "TARGET_NAME": "Esophageal cancer",
    "additional_target_cui_terms_list": [],
    "OUTPUT_RES_PREFIX": "oesophagus_",
}
config_heart = {
    "TARGET_CODES_LIST": ("I21.9",),
    "do_IPW": True,
    "do_boruta_fs": False,
    "K_IPW_RATIO": 9,
    "targets": ["Heart attack"],
    "FEATURES_REPORT_PATH": "heart_feature_report.csv",
    "QUERY_CANDIDATES_FILE": "candidate_novel_cuis_heart.csv",
    "CANDIDATE_NOVEL_CUIS_FILEPATH": "candidate_novel_cuis_heart.csv",
    "TARGET_NAME": "Heart attack",
    "additional_target_cui_terms_list": [],
    "OUTPUT_RES_PREFIX": "heart_",
}
config_eye_occ = {
    "TARGET_CODES_LIST": ("H34",),
    "do_IPW": True,
    "do_boruta_fs": False,
    "K_IPW_RATIO": 9,
    "targets": ["Retinal Vein Occlusion", "retinal artery occlusion","CRVO"], # Central retinal artery occlusio
    "FEATURES_REPORT_PATH": "eye_occ_feature_report.csv",
    "QUERY_CANDIDATES_FILE": "candidate_novel_cuis_eye_occ.csv",
    "CANDIDATE_NOVEL_CUIS_FILEPATH": "candidate_novel_cuis_eye_occ.csv",
    "TARGET_NAME": "Retinal Vein Occlusion",
    "additional_target_cui_terms_list": [],
    "OUTPUT_RES_PREFIX": "eye_occ_",
}

config_depression = {
    "TARGET_CODES_LIST": ("F32","F33"),
    "do_IPW": False,
    "do_boruta_fs": True,
    "targets": ["Depression","Depressive disorder"],
    "FEATURES_REPORT_PATH": "depression_report.csv",
    "QUERY_CANDIDATES_FILE": "candidate_novel_depression.csv",
    "CANDIDATE_NOVEL_CUIS_FILEPATH": "candidate_novel_depression.csv",
    "TARGET_NAME": "Depression",
    "additional_target_cui_terms_list": [],
    "OUTPUT_RES_PREFIX": "depression_",
}