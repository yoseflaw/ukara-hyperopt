import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import xgboost as xgb
from scripts.tuning import UkaraDataset, ModelOptimizer


def main():
    params = {
        "Dataset A": {
            "root_folder": "../input/A/",
            "slang_file": "../lib/isc.v4.csv",
            "vocab_file": "../lib/vocab_A.txt",
            "models": {
                "rf": {
                    "class": RandomForestClassifier,  # 0.089
                    "weight": 1,
                    "preprocessing": {
                        "vectorizer": CountVectorizer,
                        "vectorizer_params": {
                            "ngram_range": (1, 2),
                            "min_df": 3,
                            "max_df": 0.81,
                            "binary": False
                        },
                        "svd_dim": 74,
                        "lemmatize": True,
                        "use_extra_features": True,
                        "fix_vocab": "slang_and_vocab",
                        "slang_file": "../lib/isc.v4.csv",
                        "vocab_file": "../lib/vocab_A.txt"
                    },
                    "init_params": {
                        "n_estimators": 200,
                        "max_features": "sqrt",
                        "max_depth": 9,
                        "min_samples_split": 4,
                        "min_samples_leaf": 3,
                        "bootstrap": False,
                        "class_weight": "balanced_subsample",
                    },
                    "fit_params": {}
                },
                "xgb": {
                    "class": xgb.XGBClassifier,  # 0.084
                    "weight": 1,
                    "preprocessing": {
                        "vectorizer": CountVectorizer,
                        "vectorizer_params": {
                            "ngram_range": (1, 2),
                            "min_df": 3,
                            "max_df": 0.805,
                            "binary": False
                        },
                        "svd_dim": 76,
                        "lemmatize": False,
                        "use_extra_features": True,
                        "fix_vocab": "slang_and_vocab",
                        "slang_file": "../lib/isc.v4.csv",
                        "vocab_file": "../lib/vocab_A.txt"
                    },
                    "init_params": {
                        "n_estimators": 200,
                        "learning_rate": 0.196,
                        "max_depth": 29,
                        "gamma": 0.654,
                        "colsample_bytree": 0.83,
                        "subsample": 0.734
                    },
                    "fit_params": {}
                }
            }
        },
        "Dataset B": {
            "root_folder": "../input/B/",
            "slang_file": "../lib/isc.v4.csv",
            "vocab_file": "../lib/vocab_B.txt",
            "models": {
                "rf": {
                    "class": RandomForestClassifier,  # 0.194
                    "weight": 1,
                    "preprocessing": {
                        "vectorizer": CountVectorizer,
                        "vectorizer_params": {
                            "ngram_range": (1, 1),
                            "min_df": 2,
                            "max_df": 0.84,
                            "binary": True
                        },
                        "svd_dim": 94,
                        "lemmatize": True,
                        "use_extra_features": True,
                        "fix_vocab": "slang_and_vocab",
                        "slang_file": "../lib/isc.v4.csv",
                        "vocab_file": "../lib/vocab_B.txt"
                    },
                    "init_params": {
                        "n_estimators": 200,
                        "max_features": "auto",
                        "max_depth": 7,
                        "min_samples_split": 5,
                        "min_samples_leaf": 4,
                        "bootstrap": False,
                        "class_weight": "balanced_subsample"
                    },
                    "fit_params": {}
                },
                "xgb": {
                    "class": xgb.XGBClassifier,  # 0.198
                    "weight": 1,
                    "preprocessing": {
                        "vectorizer": CountVectorizer,
                        "vectorizer_params": {
                            "ngram_range": (1, 1),
                            "min_df": 3,
                            "max_df": 0.93,
                            "binary": True
                        },
                        "svd_dim": 101,
                        "lemmatize": True,
                        "use_extra_features": True,
                        "fix_vocab": "slang_and_vocab",
                        "slang_file": "../lib/isc.v4.csv",
                        "vocab_file": "../lib/vocab_B.txt"
                    },
                    "init_params": {
                        "n_estimators": 125,
                        "learning_rate": 0.21,
                        "max_depth": 32,
                        "gamma": 0.62,
                        "colsample_bytree": 0.9,
                        "subsample": 0.89,
                        "random_state": 42
                    },
                    "fit_params": {}
                }
            }
        }
    }
    n_estimators = list(range(50, 301, 25))
    results = []
    for dataset_name in params:
        mdl = ModelOptimizer(
            UkaraDataset(params[dataset_name]["root_folder"],
                         params[dataset_name]["slang_file"],
                         params[dataset_name]["vocab_file"])
        )
        for model_name in params[dataset_name]["models"]:
            for n_estimator in n_estimators:
                losses = []
                for i in range(10):
                    params[dataset_name]["models"][model_name]["init_params"]["n_estimators"] = n_estimator
                    cv_results = mdl.cv(params[dataset_name]["models"][model_name])
                    losses.append(cv_results['loss'])
                result = (dataset_name, model_name, n_estimator, np.mean(losses), 2*np.std(losses))
                print(result)
                results.append(result)
    print(pd.DataFrame(results,
                       columns=["dataset", "clf", "n_estimator", "mean", "CI95%"]))
    # plt.plot(
    #     [result[1] for result in results],
    #     [result[2] for result in results]
    # )
    # plt.title(model_name + "-a")
    # plt.show()


if __name__ == "__main__":
    main()