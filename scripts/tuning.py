import os
import string
from typing import List, Tuple, Dict, Any
import difflib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from spacy.lang.id import Indonesian
from hyperopt import hp, fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
from hyperopt.pyll.base import scope
import xgboost as xgb


def get_exact_words(input_str: str, vocab_list:List) -> str:
    exact_words = difflib.get_close_matches(input_str, vocab_list, n=1, cutoff=0.7)
    if len(exact_words) > 0:
        return exact_words[0]
    else:
        return input_str


def typo_correction(sentence: str, nlp: Indonesian, vocab_list: List) -> str:
    corrected_sentence = []
    for token in nlp(sentence.lower()):
        correct_word = get_exact_words(token.text, vocab_list)
        corrected_sentence.append(correct_word)
    return " ".join(corrected_sentence)


def slang_correction(sentence: str, nlp: Indonesian, slang_dict: Dict) -> str:
    corrected_sentence = []
    if not isinstance(sentence, str):
        return ""
    for token in nlp(sentence.lower()):
        if token.text in slang_dict:
            corrected_sentence.append(slang_dict[token.text])
        else:
            corrected_sentence.append(token.text)
    return " ".join(corrected_sentence)


class UkaraDataset(object):
    root_folder: str
    x_train: pd.Series
    y_train: pd.Series
    x_dev: pd.Series
    y_dev: pd.Series
    x_train_fixed_slang: pd.Series
    x_dev_fixed_slang: pd.Series
    x_test_fixed_slang: pd.Series
    x_train_fixed_typo: pd.Series
    x_dev_fixed_typo: pd.Series
    x_test_fixed_typo: pd.Series
    cv_indices: List[List[Tuple[int, int]]]
    vocab_list: List
    slang_dict: Dict
    nlp: Indonesian

    def __init__(self, root_folder: str, slang_file: str, vocab_file: str) -> None:
        self.root_folder = root_folder
        self.vocab_file = vocab_file
        self.nlp = Indonesian()
        self.x_train, self.y_train, self.x_dev, self.y_dev, self.x_test = self.load_dataset()
        if slang_file != "":
            self.fix_typo_and_store(slang_file, vocab_file)

    def load_dataset(self) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        # train_df = pd.read_csv(os.path.join(self.root_folder, "train.csv")).set_index("RES_ID")
        train_df = pd.read_csv(os.path.join(self.root_folder, "train.csv")).fillna("").set_index("RES_ID")
        dev_df = pd.read_csv(os.path.join(self.root_folder, "dev_labeled.csv")).fillna("").set_index("RES_ID")
        test_df = pd.read_csv(os.path.join(self.root_folder, "test.csv")).fillna("").set_index("RES_ID")
        return train_df["RESPONSE"], train_df["LABEL"], dev_df["RESPONSE"], dev_df["LABEL"], test_df["RESPONSE"]

    def fix_typo_and_store(self, slang_file: str, vocab_file: str) -> None:
        isc = pd.read_csv(slang_file)
        stof_df = isc[isc["in-dictionary"] == 1][["slang", "formal"]].groupby("slang")["formal"].apply(
            lambda x: list(x)[0]
        )
        self.slang_dict = stof_df.to_dict()
        self.x_train_fixed_slang = self.x_train.apply(
            lambda sentence: slang_correction(sentence, self.nlp, self.slang_dict)
        )
        self.x_dev_fixed_slang = self.x_dev.apply(
            lambda sentence: slang_correction(sentence, self.nlp, self.slang_dict)
        )
        self.x_test_fixed_slang = self.x_test.apply(
            lambda sentence: slang_correction(sentence, self.nlp, self.slang_dict)
        )
        with open(vocab_file, "r") as fvocab:
            self.vocab_list = []
            for word in fvocab.readlines():
                clean_word = word.strip()
                if clean_word not in self.vocab_list:
                    self.vocab_list.append(clean_word)
        self.x_train_fixed_typo = self.x_train_fixed_slang.apply(
            lambda sentence: typo_correction(sentence, self.nlp, self.vocab_list)
        )
        self.x_dev_fixed_typo = self.x_dev_fixed_slang.apply(
            lambda sentence: typo_correction(sentence, self.nlp, self.vocab_list)
        )
        self.x_test_fixed_typo = self.x_test_fixed_slang.apply(
            lambda sentence: typo_correction(sentence, self.nlp, self.vocab_list)
        )


class Preprocessor(object):
    lemmatize: bool
    nlp: Indonesian

    def __init__(self, lemmatize: bool) -> None:
        self.lemmatize = lemmatize
        self.nlp = Indonesian()

    def tokenizer(self, sentence: str) -> List[str]:
        out = []
        if self.lemmatize:
            out = [token.lemma_ for token in self.nlp(sentence)]
        else:
            out = [token.text for token in self.nlp(sentence)]
        return out


class FeatureExtractor(BaseEstimator, TransformerMixin):

    def fit(self, x: pd.Series, y: pd.Series):
        return self

    def check_puncts(self, sentence):
        if len(sentence) == 0: return 0
        count = 0
        for char in sentence:
            if char in string.punctuation:
                count += 1
        return count / len(sentence)

    def check_uppercase(self, sentence):
        if len(sentence) == 0: return 0
        count = 0
        for char in sentence:
            if char.isupper():
                count += 1
        return count / len(sentence)

    def transform(self, x: pd.Series):
        extra_vars = pd.DataFrame()
        extra_vars["sentence_length"] = x.apply(lambda sentence: len(sentence))
        extra_vars["num_words"] = x.apply(lambda sentence: len(sentence.split(string.whitespace)))
        extra_vars["pct_puncts"] = x.apply(lambda sentence: self.check_puncts(sentence))
        extra_vars["pct_uppercase"] = x.apply(lambda sentence: self.check_uppercase(sentence))
        return extra_vars


class ModelOptimizer(object):
    pipe: Pipeline
    prep: Preprocessor

    def __init__(self, dataset: UkaraDataset) -> None:
        self.dataset = dataset

    def cv(self, params: Dict) -> Dict:
        self.prep = Preprocessor(lemmatize=params["preprocessing"]["lemmatize"])
        fix_vocab = params["preprocessing"]["fix_vocab"]
        if fix_vocab == "slang_and_vocab":
            x_train = self.dataset.x_train_fixed_typo
        elif fix_vocab == "slang_only":
            x_train = self.dataset.x_train_fixed_slang
        else:
            x_train = self.dataset.x_train
        y_train = self.dataset.y_train
        vectorizer = params["preprocessing"]["vectorizer"]
        default_proprecessing = Pipeline([
                ("vectorizer", vectorizer(
                    tokenizer=self.prep.tokenizer,
                    **params["preprocessing"]["vectorizer_params"]
                )),
                ("svd", TruncatedSVD(params["preprocessing"]["svd_dim"], random_state=42))
            ])
        if params["preprocessing"]["use_extra_features"]:
            text_preprocessing = default_proprecessing
            extra_vars_preprocessing = Pipeline([
                ("feature_extraction", FeatureExtractor()),
                ("scaler", StandardScaler())
            ])
            preprocessing = FeatureUnion([
                ("text_preprocessing", text_preprocessing),
                ("extra_vars_preprocessing", extra_vars_preprocessing)
            ])
        else: preprocessing = default_proprecessing
        clf = Pipeline([
            ("preprocessing", preprocessing),
            ("classifier", params["class"](**params["init_params"]))
        ])
        cv_results = cross_validate(clf, x_train, y_train, scoring="f1", cv=5,
                                    return_train_score=False, fit_params=params["fit_params"])
        return {"loss": 1 - np.mean(cv_results["test_score"]), "status": STATUS_OK}

    def optimize(self, space, trials, algo, max_evals):
        # try:
        result = fmin(fn=self.cv, space=space, algo=algo, max_evals=max_evals, trials=trials)
        # except Exception as e:
        #     return {'status': STATUS_FAIL,
        #             'exception': str(e)}
        return result, trials


def optimize(dataset: UkaraDataset, tuned_models: List, params: Dict) -> List:
    best_models = []
    for model_name in tuned_models:
        model_opt = ModelOptimizer(dataset)
        results = model_opt.optimize(
            params[model_name],
            trials=Trials(),
            algo=tpe.suggest,
            max_evals=params[model_name]["max_evals"]
        )
        print("best_params: " + str(results[0]))
        best_models.append((params[model_name]["class"], results[0]))
    return best_models


def test_datasets() -> int:
    dataset_a = UkaraDataset(
        root_folder="../input/A",
        slang_file="../lib/isc.v4.csv",
        vocab_file="../lib/vocab_A.txt"
    )
    print(dataset_a.x_train.values[70:90])
    print()
    print(dataset_a.x_train_fixed_slang.values[70:90])
    print()
    print(dataset_a.x_train_fixed_typo.values[70:90])
    dataset_b = UkaraDataset(
        root_folder="../input/B",
        slang_file="../lib/isc.v4.csv",
        vocab_file="../lib/vocab_B.txt"
    )
    print(dataset_b.x_train.values[70:90])
    print()
    print(dataset_b.x_train_fixed_slang.values[70:90])
    print()
    print(dataset_b.x_train_fixed_typo.values[70:90])
    return 0


def main() -> int:
    preprocessing = {
        "vectorizer": hp.choice("vectorizer", [CountVectorizer, TfidfVectorizer]),
        "vectorizer_params" : {
            "ngram_range": (1, scope.int(hp.quniform("ngram_range", 1, 3, 1))),
            "min_df": scope.int(hp.quniform("min_df", 1, 3, 1)),
            "max_df": hp.uniform("max_df", 0.8, 1.0),
            "binary": hp.choice("binary", [False, True])
        },
        "svd_dim": 100,
        "lemmatize": hp.choice("lemmatize", [False, True]),
        "use_extra_features": hp.choice("use_extra_features", [False, True]),
        # "fix_vocab": "slang_and_vocab"
        "fix_vocab": hp.choice("fix_vocab", [None, "slang_only", "slang_and_vocab"])
    }
    print("[Dataset A]")
    optimize(
        dataset=UkaraDataset(
            root_folder="../input/A/",
            # root_folder="../input/A/",
            slang_file="../lib/isc.v4.csv",
            vocab_file="../lib/vocab_A.txt"
        ),
        tuned_models=["logreg", "rf", "xgb", "svm"],
        params={
            "logreg": {
                "class": LogisticRegression,
                "max_evals": 100,
                "preprocessing": preprocessing,
                "init_params": {
                    "solver": "lbfgs",
                    "C": hp.loguniform("C", -6, 7),
                    "max_iter": 1000,
                    "class_weight": hp.choice("class_weight", [None, "balanced"]),
                    "random_state": 42
                },
                "fit_params": {}
            },
            "rf": {
                "class": RandomForestClassifier,
                "max_evals": 100,
                "preprocessing": preprocessing,
                "init_params": {
                    "n_estimators": 200,
                    "max_features": hp.choice("max_features", ["auto", "sqrt"]),
                    "max_depth": scope.int(hp.quniform("max_depth", 5, 15, 1)),
                    "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 10, 1)),
                    "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 1, 6, 1)),
                    "bootstrap": True,
                    "class_weight": hp.choice("class_weight", [None, "balanced", "balanced_subsample"]),
                    "random_state": 42
                },
                "fit_params": {}
            },
            "xgb": {
                "class": xgb.XGBClassifier,
                "max_evals": 100,
                "preprocessing": preprocessing,
                "init_params": {
                    "n_estimators": 200,
                    "learning_rate": hp.uniform("learning_rate", 0.15, 0.5),
                    "max_depth": scope.int(hp.quniform("max_depth", 25, 35, 1)),
                    "gamma": hp.uniform("gamma", 0.4, 0.8),
                    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 0.9),
                    "subsample": hp.uniform("subsample", 0.7, 1.0),
                    "random_state": 42
                },
                "fit_params": {
                    # "classifier__eval_metric": "logloss",
                    # "classifier__early_stopping_rounds": 10
                }
            },
            "svm": {
                "class": SVC,
                "max_evals": 100,
                "preprocessing": preprocessing,
                "init_params": {
                    "C": hp.loguniform("C", -6, 7),
                    "kernel": hp.choice("kernel", ["linear", "rbf"]),
                    "gamma": hp.loguniform("gamma", -7, 0),
                    "class_weight": hp.choice("class_weight", [None, "balanced"]),
                    "probability": True,
                    "random_state": 42
                },
                "fit_params": {}
            }
        }
    )

    print("[Dataset B]")
    best_models_b = optimize(
        dataset=UkaraDataset(
            root_folder="../input/B/",
            # root_folder="../input/B/",
            slang_file="../lib/isc.v4.csv",
            vocab_file="../lib/vocab_B.txt"
        ),
        tuned_models=["logreg", "rf", "xgb", "svm"],
        params={
            "logreg": {
                "class": LogisticRegression,
                "max_evals": 100,
                "preprocessing": preprocessing,
                "init_params": {
                    "solver": "lbfgs",
                    "C": hp.loguniform("C", -6, 7),
                    "max_iter": 1000,
                    "class_weight": hp.choice("class_weight", [None, "balanced"]),
                    "random_state": 42
                },
                "fit_params": {}
            },
            "rf": {
                "class": RandomForestClassifier,
                "max_evals": 100,
                "preprocessing": preprocessing,
                "init_params": {
                    "n_estimators": 200,
                    "max_features": hp.choice("max_features", ["auto", "sqrt"]),
                    "max_depth": scope.int(hp.quniform("max_depth", 5, 15, 1)),
                    "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 10, 1)),
                    "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 2, 6, 1)),
                    "bootstrap": hp.choice("bootstrap", [False, True]),
                    "class_weight": hp.choice("class_weight", [None, "balanced", "balanced_subsample"]),
                    "random_state": 42
                },
                "fit_params": {}
            },
            "xgb": {
                "class": xgb.XGBClassifier,
                "max_evals": 100,
                "preprocessing": preprocessing,
                "init_params": {
                    "n_estimators": 200,
                    "learning_rate": hp.uniform("learning_rate", 0.15, 0.5),
                    "max_depth": scope.int(hp.quniform("max_depth", 25, 35, 1)),
                    "gamma": hp.uniform("gamma", 0.4, 0.8),
                    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 0.9),
                    "subsample": hp.uniform("subsample", 0.7, 1.0),
                    "random_state": 42
                },
                "fit_params": {
                    # "classifier__eval_metric": "logloss",
                    # "classifier__early_stopping_rounds": 10
                }
            },
            "svm": {
                "class": SVC,
                "max_evals": 100,
                "preprocessing": preprocessing,
                "init_params": {
                    "C": hp.loguniform("C", -6, 7),
                    "kernel": hp.choice("kernel", ["linear", "rbf"]),
                    "gamma": hp.loguniform("gamma", -7, 0),
                    "class_weight": hp.choice("class_weight", [None, "balanced"]),
                    "probability": True,
                    "random_state": 42
                },
                "fit_params": {}
            }
        }
    )
    return 0


if __name__ == "__main__":
    main()
