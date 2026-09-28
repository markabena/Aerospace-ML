"""
Project 5: Engine Health Monitor
Aerospace-ML -- Foundations track

Purpose
-------
Train a machine learning model to flag a degraded rocket engine from its
static-fire sensor readings, before the degradation becomes a failure.
This is the first real ML project in the portfolio and the direct
stepping stone to the capstone (predictive maintenance for aircraft
components) -- same problem shape, different hardware.

Input file: Merlin_Engine_Test_Data.csv
    1,000 simulated static-fire test records for a Merlin-class LOX/RP-1
    engine. Nominal values sit in the right ballpark for a Merlin 1D
    (chamber pressure ~97 bar, turbopump ~36,000 rpm), but every number
    is generated from a statistical model -- this is NOT measured engine
    data. Three physically-motivated degradation modes are mixed in:
    turbopump bearing wear, injector fouling, and coolant channel
    blockage. 15% of records are degraded, which is deliberate -- see
    the note on class imbalance below.

What "machine learning" actually means here
--------------------------------------------
Everything up to P4 was a FORMULA: you knew the equation (vis-viva,
Tsiolkovsky) and coded it. Here there is no equation for "is this
engine degrading". Instead you show the model many labelled examples
and it works out the pattern itself. You supply the examples; it
supplies the rule.

New concepts introduced in this build
---------------------------------------
    - Features (X) vs labels (y) : the inputs vs the answer being learned
    - train_test_split           : holding data back to test honestly
    - StandardScaler             : putting features on a comparable scale
    - .fit() / .predict()        : the two-step core of every sklearn model
    - Confusion matrix           : what KIND of mistakes the model makes
    - Precision / recall          : why "accuracy" alone is a trap here
    - Feature importance          : which sensors the model actually relies on
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Paths resolve relative to this script, so it runs from any directory.
HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "Merlin_Engine_Test_Data.csv"
PREDICTIONS_OUTPUT = HERE / "Engine_Health_Predictions.csv"

FEATURE_COLUMNS = [
    "chamber_pressure_bar",
    "turbopump_rpm",
    "fuel_flow_kg_s",
    "lox_flow_kg_s",
    "exhaust_gas_temp_k",
    "vibration_rms_g",
    "burn_duration_s",
    "mixture_ratio",
]
LABEL_COLUMN = "health_status"


def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{filepath} not found -- it should sit in the same folder as this script."
        )
    return df


def describe_dataset(df):
    """
    Always look at the class balance BEFORE training anything.

    Here 85% of tests are nominal. That means a useless model that
    simply answers "nominal" every single time would score 85%
    accuracy -- while catching precisely zero failing engines. This is
    exactly why accuracy alone is a trap, and why the report below
    leans on recall instead.
    """
    counts = df[LABEL_COLUMN].value_counts()
    total = len(df)
    print(f"Loaded {total} engine test records\n")
    print("Class balance:")
    for label, count in counts.items():
        print(f"  {label:.<12} {count:>4}  ({count / total:.1%})")
    baseline = counts.max() / total
    print(f"\n  Always-guess-majority baseline accuracy: {baseline:.1%}")
    print("  Any model must beat this to be worth anything.\n")


def split_features_and_labels(df):
    """
    >>> CONCEPT: X and y

    X = the features: the sensor readings the model is allowed to see.
    y = the label: the answer it is trying to learn ("nominal"/"degraded").

    test_id is deliberately excluded from X. It's an identifier, not a
    measurement -- feeding it in would let the model memorise individual
    tests instead of learning the underlying physics.
    """
    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]
    return X, y


def build_model():
    """
    >>> CONCEPT: Pipeline + StandardScaler

    The features are on wildly different scales -- turbopump_rpm is in
    the tens of thousands, vibration_rms_g is around 2. Distance- and
    gradient-based models treat a big number as a big signal, so rpm
    would drown out vibration purely because of its units.
    StandardScaler re-centres every feature to mean 0, std 1, so each
    is judged on how far it deviates from normal, not on its raw size.

    Wrapping the scaler and the model in a Pipeline means the scaler is
    fitted on training data ONLY, then reapplied to test data. Doing it
    manually across the whole dataset first would leak information from
    the test set into training -- a classic and easy mistake.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",  # counteract the 85/15 imbalance
        )),
    ])


def evaluate(model, X_test, y_test):
    """
    >>> CONCEPT: confusion matrix, precision, recall

    A confusion matrix breaks results into four buckets:
        - true negative  : nominal engine, correctly cleared
        - false positive : nominal engine, wrongly flagged  -> wasted teardown
        - false negative : degraded engine, MISSED          -> engine flies degraded
        - true positive  : degraded engine, correctly caught

    For engine health those two error types are NOT equally bad. A
    false positive costs an unnecessary inspection. A false negative
    puts a degrading engine on a launch vehicle. So RECALL on the
    degraded class -- the share of genuinely bad engines that got
    caught -- matters more than raw accuracy.
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("MODEL EVALUATION (on held-out test data)")
    print("-" * 52)
    print(f"  Overall accuracy: {accuracy:.1%}\n")

    labels = ["nominal", "degraded"]
    cm = confusion_matrix(y_test, predictions, labels=labels)
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    print("  Confusion matrix:")
    print(f"    Nominal correctly cleared (TN) : {tn}")
    print(f"    Nominal wrongly flagged   (FP) : {fp}   -> unnecessary teardown")
    print(f"    Degraded MISSED           (FN) : {fn}   -> the dangerous one")
    print(f"    Degraded caught           (TP) : {tp}\n")

    print(classification_report(y_test, predictions, labels=labels, digits=3))
    return predictions


def check_generalisation(model, X, y):
    """
    >>> CONCEPT: cross-validation

    A single train/test split can get lucky or unlucky depending on
    which rows landed where. Cross-validation repeats the split five
    times over different slices and reports the spread, which is a far
    more honest read on whether the model actually generalises.
    """
    scores = cross_val_score(model, X, y, cv=5, scoring="recall_macro")
    print("5-fold cross-validation (macro recall):")
    print(f"  Scores: {', '.join(f'{s:.3f}' for s in scores)}")
    print(f"  Mean:   {scores.mean():.3f}  (+/- {scores.std():.3f})\n")


def show_feature_importance(model, feature_names):
    """
    A random forest can report how much each feature contributed to its
    decisions. This is the sanity check that matters most: if the model
    is leaning on features that make no physical sense, it has latched
    onto an artefact of the data rather than real engine behaviour.
    """
    importances = model.named_steps["classifier"].feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda pair: pair[1], reverse=True)

    print("FEATURE IMPORTANCE (what the model actually relies on)")
    print("-" * 52)
    for name, importance in ranked:
        bar = "#" * int(importance * 60)
        print(f"  {name:.<26} {importance:.3f}  {bar}")
    print()


def compare_with_simpler_model(X_train, X_test, y_train, y_test):
    """
    Always sanity-check a complex model against a simple one. If
    logistic regression -- essentially a weighted sum of the features --
    performs just as well, the extra complexity of a forest isn't
    buying anything and the simpler model is the better engineering
    choice: faster, and far easier to explain to a reviewer.
    """
    simple = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    simple.fit(X_train, y_train)
    simple_acc = accuracy_score(y_test, simple.predict(X_test))
    print(f"Baseline comparison -- logistic regression accuracy: {simple_acc:.1%}\n")


def main():
    df = load_data(DATA_FILE)
    describe_dataset(df)

    X, y = split_features_and_labels(df)

    # >>> CONCEPT: train_test_split
    # The model is trained on 80% and judged on the 20% it has never
    # seen. Testing on data the model trained on tells you only that it
    # can memorise -- not that it can generalise to a new engine.
    # stratify=y keeps the 85/15 balance intact in BOTH splits, so the
    # test set isn't accidentally short of degraded examples.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training on {len(X_train)} records | testing on {len(X_test)} held-out records\n")

    model = build_model()
    model.fit(X_train, y_train)  # <- this single line is "the learning"

    predictions = evaluate(model, X_test, y_test)
    check_generalisation(model, X, y)
    show_feature_importance(model, FEATURE_COLUMNS)
    compare_with_simpler_model(X_train, X_test, y_train, y_test)

    # Export the test-set predictions alongside the truth, so a failed
    # call can be traced back to the specific engine test that caused it.
    results = df.loc[X_test.index, ["test_id"] + FEATURE_COLUMNS].copy()
    results["actual"] = y_test.values
    results["predicted"] = predictions
    results["correct"] = results["actual"] == results["predicted"]
    results.to_csv(PREDICTIONS_OUTPUT, index=False)
    print(f"Test-set predictions written to: {PREDICTIONS_OUTPUT.name}")
    print("Plain CSV -- MATLAB reads it with readtable(), same bridge as P3.")


if __name__ == "__main__":
    main()
