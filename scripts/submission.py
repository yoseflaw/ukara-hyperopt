import difflib
import os
from copy import copy
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hyperopt import hp, fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
from hyperopt.pyll.base import scope
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_curve
import xgboost as xgb
from sklearn.svm import SVC
from spacy.lang.id import Indonesian

from scripts.tuning import (Preprocessor, FeatureExtractor, UkaraDataset,
                            slang_correction, typo_correction)


class SlangTypoCorrector(BaseEstimator, TransformerMixin):
    nlp: Indonesian
    mode: str
    slang_dict: Dict
    vocab_list: List

    def __init__(self, mode: str, slang_file: str, vocab_file: str) -> None:
        self.mode = mode
        self.slang_file = slang_file
        self.vocab_file = vocab_file
        self.nlp = Indonesian()
        isc = pd.read_csv(slang_file)
        stof_df = isc[isc["in-dictionary"] == 1][["slang", "formal"]].groupby("slang")["formal"].apply(
            lambda x: list(x)[0])
        self.slang_dict = stof_df.to_dict()
        with open(vocab_file, "r") as fvocab:
            self.vocab_list = []
            for word in fvocab.readlines():
                clean_word = word.strip()
                if clean_word not in self.vocab_list:
                    self.vocab_list.append(clean_word)

    def fit(self, x: pd.Series, y: pd.Series):
        return self

    def transform(self, x: pd.Series):
        if self.mode == "slang_only":
            return x.apply(lambda sentence: slang_correction(sentence, self.nlp, self.slang_dict))
        elif self.mode == "slang_and_typo":
            return x.apply(
                lambda sentence: typo_correction(
                    slang_correction(sentence, self.nlp, self.slang_dict),
                    self.nlp,
                    self.vocab_list
                )
            )
        else:
            return x


class VotingOptimizer(object):

    def __init__(self, dataset: UkaraDataset) -> None:
        self.dataset = dataset

    def make_preprocessing(self, params) -> Pipeline:
        prep = Preprocessor(lemmatize=params["preprocessing"]["lemmatize"])
        vectorizer = params["preprocessing"]["vectorizer"]
        default_preprocessing = Pipeline([
            (
                "formalizer",
                SlangTypoCorrector(
                    mode=params["preprocessing"]["fix_vocab"],
                    slang_file=params["preprocessing"]["slang_file"],
                    vocab_file=params["preprocessing"]["vocab_file"]
                )
            ),
            ("vectorizer", vectorizer(
                tokenizer=prep.tokenizer,
                **params["preprocessing"]["vectorizer_params"]
            )),
            ("svd", TruncatedSVD(params["preprocessing"]["svd_dim"], random_state=42))
        ])
        if params["preprocessing"]["use_extra_features"]:
            text_preprocessing = default_preprocessing
            extra_vars_preprocessing = Pipeline([
                ("feature_extraction", FeatureExtractor()),
                ("scaler", StandardScaler())
            ])
            preprocessing = FeatureUnion([
                ("text_preprocessing", text_preprocessing),
                ("extra_vars_preprocessing", extra_vars_preprocessing)
            ])
        else:
            preprocessing = default_preprocessing
        return preprocessing

    def make_model(self, params) -> Pipeline:
        clf = Pipeline([
            ("preprocessing", self.make_preprocessing(params)),
            ("classifier", params["class"](**params["init_params"]))
        ])
        return clf

    def make_voting(self, config, voting, is_optimizing) -> VotingClassifier:
        estimators = []
        weights = []
        for model_name in config["models"]:
            estimators.append((model_name, self.make_model(config["models"][model_name])))
            if is_optimizing:
                weights.append(config["stacking_params"]["weight"][model_name])
            else:
                weights.append(config["models"][model_name]["weight"])
        clf = VotingClassifier(estimators, voting, weights)
        return clf

    def cv(self, config: Dict) -> Dict:
        x_train, y_train = self.dataset.x_train, self.dataset.y_train
        voting_clfs = self.make_voting(config, config["stacking_params"]["voting"], is_optimizing=True)
        cv_results = cross_validate(voting_clfs, x_train, y_train, scoring="f1", cv=5,
                                    return_train_score=False,
                                    fit_params=config["stacking_params"]["fit_params"])
        return {"loss": 1 - np.mean(cv_results["test_score"]), "status": STATUS_OK}

    def cv_score(self, config: Dict) -> Dict:
        x_train, y_train = self.dataset.x_train, self.dataset.y_train
        voting_clfs = self.make_voting(config, config["stacking_params"]["voting"], is_optimizing=False)
        cv_results = cross_validate(voting_clfs, x_train, y_train, scoring="f1", cv=5,
                                    return_train_score=False,
                                    fit_params=config["stacking_params"]["fit_params"])
        return cv_results["test_score"]

    def optimize(self, space, trials, algo, max_evals):
        result = fmin(fn=self.cv, space=space, algo=algo, max_evals=max_evals, trials=trials)
        return result, trials

    def get_f1s(self, config, voting):
        k_cv = 5
        x_train, y_train = self.dataset.x_train, self.dataset.y_train
        cv = StratifiedKFold(n_splits=k_cv)
        clf = self.make_voting(config, voting, is_optimizing=False)
        base_thresholds = np.linspace(0.4, 0.8, 80)
        interp_f1s = []
        for train_idx, test_idx in cv.split(x_train, y_train):
            probs = clf.fit(x_train[train_idx], y_train[train_idx]).predict_proba(x_train[test_idx])
            precision, recall, thresholds = precision_recall_curve(y_train[test_idx], probs[:, 1])
            f1 = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
            interp_f1 = np.interp(base_thresholds, thresholds, f1)
            interp_f1s.append(interp_f1)
        interp_f1s = np.array(interp_f1s)
        f1_data = np.zeros((interp_f1s.shape[1], 2))
        f1_data[:, 0] = np.mean(interp_f1s, axis=0)
        f1_data[:, 1] = np.std(interp_f1s, axis=0)
        f1_scores = pd.DataFrame(
            data=f1_data,
            index=pd.Index(base_thresholds, name="threshold"),
            columns=["mean_f1", "std_f1"]
        )
        print(f1_scores.sort_values("mean_f1", ascending=False).head(10))
        # f1_scores.plot()
        # plt.show()
        return 0

    def predict(self, config, voting, mode):
        clf = self.make_voting(config, voting, is_optimizing=False)
        # cv_results = cross_validate(clf, dataset.x_train, dataset.y_train,
        #                             scoring="f1",
        #                             cv=5,
        #                             return_train_score=False)
        # print(cv_results)
        clf.fit(self.dataset.x_train, self.dataset.y_train)
        if mode == "dev":
            if voting == "soft":
                probs = clf.predict_proba(self.dataset.x_dev)
                preds = np.int8(probs[:, 1] >= config["threshold"])
            else:
                preds = clf.predict(self.dataset.x_dev)
            preds_df = pd.DataFrame(preds, index=self.dataset.x_dev.index, columns=["PREDICTION"])
            preds_df["LABEL"] = self.dataset.y_dev
            preds_df["ERROR"] = abs(self.dataset.y_dev.values - preds)
            preds_df["RESPONSE"] = self.dataset.x_dev
            return preds_df.reset_index()
        if mode == "pseudo":
            probs = clf.predict_proba(self.dataset.x_dev)
            preds_df = pd.DataFrame(self.dataset.x_dev, index=self.dataset.x_dev.index, columns=["RESPONSE"])
            preds_df["LABEL"] = np.int8(probs[:, 1] >= config["threshold"])
            preds_df["PSEUDO"] = np.abs(probs[:, 1] - 0.5)
            preds_neg_df = preds_df[self.dataset.y_dev == 0].sort_values(
                "PSEUDO", ascending=False).head(int(len(preds_df[self.dataset.y_dev == 0]) * 0.2))
            preds_pos_df = preds_df[self.dataset.y_dev == 1].sort_values(
                "PSEUDO", ascending=False).head(int(len(preds_df[self.dataset.y_dev == 1]) * 0.2))
            preds_df = pd.concat([preds_neg_df, preds_pos_df])
            return preds_df[["RESPONSE", "LABEL"]].reset_index()
        elif mode == "test":
            if voting == "soft":
                probs = clf.predict_proba(self.dataset.x_test)
                preds = np.int8(probs[:, 1] >= config["threshold"])
            else:
                preds = clf.predict(self.dataset.x_test)
            preds_df = pd.DataFrame(preds, index=self.dataset.x_test.index, columns=["LABEL"])
            return preds_df.reset_index()
        return None


configs = {
    "Dataset A": { # ensemble: 0.082
        "root_folder": "../input/A/",
        "stacking_params": {
            "max_evals": 30,
            # "voting": hp.choice("voting", ["soft", "hard"]),
            "voting": "soft",
            "weight": {
                "logreg": 0,
                "rf": hp.uniform("weight_rf", 0, 1),
                "xgb": 1,
                "svm": hp.uniform("weight_svm", 0, 1)
            },
            "fit_params": {}
        },
        "models": {
            "logreg": {
                "class": LogisticRegression,  # 0.097
                "weight": 0,
                "preprocessing": {
                    "vectorizer": TfidfVectorizer,
                    "vectorizer_params" : {
                        "ngram_range": (1, 3),
                        "min_df": 1,
                        "max_df": 0.92,
                        "binary": True
                    },
                    "svd_dim": 72,
                    "lemmatize": True,
                    "use_extra_features": True,
                    "fix_vocab": "slang_and_vocab",
                    "slang_file": "../lib/isc.v4.csv",
                    "vocab_file": "../lib/vocab_A.txt"
                },
                "init_params": {
                    "C": 45,
                    "solver": "lbfgs",
                    "max_iter": 1000,
                    "class_weight": None
                },
                "fit_params": {}
            },
            "rf": {
                "class": RandomForestClassifier,  # 0.089
                "weight": 0.771,
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
                    "n_estimators": 250,
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
                "weight": 0.881,
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
                    "n_estimators": 75,
                    "learning_rate": 0.196,
                    "max_depth": 29,
                    "gamma": 0.654,
                    "colsample_bytree": 0.83,
                    "subsample": 0.734
                },
                "fit_params": {}
            },
            "svm": {
                "class": SVC,  # 0.089
                "weight": 1,
                "preprocessing": {
                    "vectorizer": TfidfVectorizer,
                    "vectorizer_params": {
                        "ngram_range": (1, 3),
                        "min_df": 2,
                        "max_df": 0.851,
                        "binary": True
                    },
                    "svd_dim": 99,
                    "lemmatize": True,
                    "use_extra_features": True,
                    "fix_vocab": "slang_and_vocab",
                    "slang_file": "../lib/isc.v4.csv",
                    "vocab_file": "../lib/vocab_A.txt"
                },
                "init_params": {
                    "C": 316,
                    "kernel": "rbf",
                    "gamma": 0.046,
                    "probability": True,
                    "class_weight": "balanced"
                },
                "fit_params": {}
            }
        }
    },
    "Dataset B": {  # ensemble: 0.176
        "root_folder": "../input/B/",
        "stacking_params": {
            "max_evals": 30,
            # "voting": hp.choice("voting", ["soft", "hard"]),
            "voting": "soft",
            "weight": {
                "logreg": hp.uniform("weight_logreg", 0, 0.3),
                "rf": hp.uniform("weight_rf", 0, 0.3),
                "xgb": hp.uniform("weight_xgb", 0, 0.3),
                "svm": 1
            },
            "fit_params": {}
        },
        "models": {
            "logreg": {
                "class": LogisticRegression,  # 0.2
                "weight": 0.075,
                "preprocessing": {
                    "vectorizer": TfidfVectorizer,
                    "vectorizer_params": {
                        "ngram_range": (1, 2),
                        "min_df": 1,
                        "max_df": 0.97,
                        "binary": False
                    },
                    "svd_dim": 75,
                    "lemmatize": True,
                    "use_extra_features": True,
                    "fix_vocab": "slang_and_vocab",
                    "slang_file": "../lib/isc.v4.csv",
                    "vocab_file": "../lib/vocab_B.txt"
                },
                "init_params": {
                    "C": 98.81,
                    "solver": "lbfgs",
                    "max_iter": 1000,
                    "class_weight": "balanced"
                },
                "fit_params": {}
            },
            "rf": {
                "class": RandomForestClassifier,  # 0.194
                "weight": 0.166,
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
                    "n_estimators": 175,
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
                "weight": 0.294,
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
                    "n_estimators": 50,
                    "learning_rate": 0.21,
                    "max_depth": 32,
                    "gamma": 0.62,
                    "colsample_bytree": 0.9,
                    "subsample": 0.89,
                    "random_state": 42
                },
                "fit_params": {}
            },
            "svm": {
                "class": SVC,  # 0.178
                "weight": 1,
                "preprocessing": {
                    "vectorizer": TfidfVectorizer,
                    "vectorizer_params": {
                        "ngram_range": (1, 2),
                        "min_df": 1,
                        "max_df": 0.95,
                        "binary": False
                    },
                    "svd_dim": 102,
                    "lemmatize": True,
                    "use_extra_features": True,
                    "fix_vocab": "slang_and_vocab",
                    "slang_file": "../lib/isc.v4.csv",
                    "vocab_file": "../lib/vocab_B.txt"
                },
                "init_params": {
                    "C": 90.2,
                    "kernel": "rbf",
                    "gamma": 0.005,
                    "probability": True,
                    "class_weight": None
                },
                "fit_params": {}
            }
        }
    }
}


def train_cv():
    for dataset_name in configs:
        results = []
        for i in range(10):
            voting_opt = VotingOptimizer(UkaraDataset(configs[dataset_name]["root_folder"], "", ""))
            result = voting_opt.cv_score(configs[dataset_name])
            results = results + list(result)
            print(f"CV-{i}, F1: {result}")
        mean_result = np.mean(results)
        std_result = np.std(results)
        print(f"{dataset_name}, cv: {mean_result} +- {std_result}")
    return 0


def optimize():
    for dataset_name in configs:
        voting_opt = VotingOptimizer(UkaraDataset(configs[dataset_name]["root_folder"], "", ""))
        results = voting_opt.optimize(
            configs[dataset_name],
            trials=Trials(),
            algo=tpe.suggest,
            max_evals=configs[dataset_name]["stacking_params"]["max_evals"]
        )
        print("best_params: ", str(results[0]))
    return 0


def get_f1s():
    print("Plot F1 for Dataset A")
    voting_a = VotingOptimizer(UkaraDataset(configs["Dataset A"]["root_folder"], "", ""))
    voting_a.get_f1s(configs["Dataset A"], "soft")
    print("Plot F1 for Dataset B")
    voting_b = VotingOptimizer(UkaraDataset(configs["Dataset B"]["root_folder"], "", ""))
    voting_b.get_f1s(configs["Dataset B"], "soft")
    return 0


def pseudo():
    print("Predict for Dataset A")
    voting_a = VotingOptimizer(UkaraDataset(configs["Dataset A"]["root_folder"], "", ""))
    pred_a = voting_a.predict(configs["Dataset A"], "soft", mode="pseudo")
    pred_a.to_csv(os.path.join(configs["Dataset A"]["root_folder"], "dev_pseudo_20pct.csv"), index=False)
    print("Predict for Dataset B")
    voting_b = VotingOptimizer(UkaraDataset(configs["Dataset B"]["root_folder"], "", ""))
    pred_b = voting_b.predict(configs["Dataset B"], "soft", mode="pseudo")
    pred_b.to_csv(os.path.join(configs["Dataset B"]["root_folder"], "dev_pseudo_20pct.csv"), index=False)
    return 0


def dev():
    print("Predict for Dataset A")
    voting_a = VotingOptimizer(UkaraDataset(configs["Dataset A"]["root_folder"], "", ""))
    pred_a = voting_a.predict(configs["Dataset A"], "hard", mode="dev")
    print("Predict for Dataset B")
    voting_b = VotingOptimizer(UkaraDataset(configs["Dataset B"]["root_folder"], "", ""))
    pred_b = voting_b.predict(configs["Dataset B"], "hard", mode="dev")
    all_preds = pd.concat([pred_a, pred_b])
    print(classification_report(all_preds["LABEL"], all_preds["PREDICTION"]))
    with pd.option_context("display.max_colwidth", 140,
                           "display.max_columns", None,
                           "display.max_rows", None):
        print("TRUE LABEL=0, PREDICTION=1")
        print(all_preds[(all_preds["ERROR"] == 1) & (all_preds["LABEL"] == 0)]["RESPONSE"])
        print()
        print("TRUE LABEL=1, PREDICTION=0")
        print(all_preds[(all_preds["ERROR"] == 1) & (all_preds["LABEL"] == 1)]["RESPONSE"])
    return 0


def submission():
    print("Predict for Dataset A")
    voting_a = VotingOptimizer(UkaraDataset(configs["Dataset A"]["root_folder"], "", ""))
    pred_a = voting_a.predict(configs["Dataset A"], "soft", mode="test")
    print("Predict for Dataset B")
    voting_b = VotingOptimizer(UkaraDataset(configs["Dataset B"]["root_folder"], "", ""))
    pred_b = voting_b.predict(configs["Dataset B"], "soft", mode="test")
    all_preds = pd.concat([pred_a, pred_b])
    print(all_preds.head())
    print(all_preds["LABEL"].value_counts())
    all_preds.to_json(
        "../predictions/test_simple_pseudo-20pct_{}.json".format(pd.Timestamp.today().strftime("%Y%m%d")),
        orient="records"
    )
    return 0


if __name__ == "__main__":
    # optimize()
    train_cv()
    # get_f1s()
    # dev()
    # submission()
