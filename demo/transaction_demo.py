"""DEMO: transaction prediction from Age, Income, Gender and Occupation.

This is NOT the PRCP-1003 model. The PRCP-1003 bank data is anonymized (var_0 ... var_199) and has no
age, income, gender or occupation, so a model on those fields cannot be trained from it. This demo trains
a small model on GENERATED sample data to show how such an input form and prediction would work.

Usage (from the project folder):
    python demo/transaction_demo.py                      # asks for the four details
    python demo/transaction_demo.py --age 35 --income 800000 --gender Female --occupation Salaried
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEMO_DIR = Path(__file__).resolve().parent
MODEL_PATH = DEMO_DIR / "demo_model.pkl"

GENDERS = ["Male", "Female", "Other"]
OCCUPATIONS = ["Salaried", "Self-employed", "Business owner", "Student", "Retired", "Unemployed", "Other"]
OCCUPATION_EFFECT = {"Salaried": 0.4, "Self-employed": 0.5, "Business owner": 0.8, "Student": -0.6,
                     "Retired": -0.2, "Unemployed": -0.9, "Other": 0.0}
DEMO_NOTE = "DEMO ONLY - trained on generated sample data, not on the PRCP-1003 bank data."


def generate_data(n=20000, seed=42):
    rng = np.random.default_rng(seed)
    occupation = rng.choice(OCCUPATIONS, size=n, p=[0.40, 0.15, 0.08, 0.12, 0.10, 0.08, 0.07])
    age = np.clip(rng.normal(38, 12, n), 18, 80).round()
    age[occupation == "Student"] = rng.integers(18, 26, (occupation == "Student").sum())
    age[occupation == "Retired"] = rng.integers(58, 81, (occupation == "Retired").sum())
    income = np.exp(rng.normal(np.log(600000), 0.6, n))
    income[occupation == "Student"] *= 0.2
    income[occupation == "Unemployed"] *= 0.15
    gender = rng.choice(GENDERS, size=n, p=[0.49, 0.49, 0.02])

    logit = (-2.6
             + 0.9 * np.log(income / 600000)
             - 0.0015 * (age - 40) ** 2
             + np.vectorize(OCCUPATION_EFFECT.get)(occupation))
    target = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame({"Age": age, "Income": income.round(-2), "Gender": gender,
                         "Occupation": occupation, "target": target.astype(int)})


def train():
    data = generate_data()
    X, y = data.drop(columns="target"), data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train = X_train.assign(Income=np.log1p(X_train["Income"]))
    X_test = X_test.assign(Income=np.log1p(X_test["Income"]))

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), ["Age", "Income"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Gender", "Occupation"]),
    ])
    model = Pipeline([("preprocessor", preprocessor),
                      ("model", LogisticRegression(max_iter=1000))])
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    joblib.dump({"model": model, "test_auc": auc, "base_rate": float(y.mean())}, MODEL_PATH)
    return joblib.load(MODEL_PATH)


def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else train()


def ask(prompt, convert, check, error):
    while True:
        raw = input(prompt).strip()
        try:
            value = convert(raw)
            if check(value):
                return value
        except ValueError:
            pass
        print(f"  {error}")


def pick(prompt, options, allow_other=False):
    print(prompt)
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")
    while True:
        raw = input("Choose a number" + (" or type an occupation" if allow_other else "") + ": ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        match = next((o for o in options if o.lower() == raw.lower()), None)
        if match:
            return match
        if allow_other and raw:
            print(f"  '{raw}' is not a known occupation; it will be treated as 'Other'.")
            return "Other"
        print("  Please choose one of the listed options.")


def normalise(value, options):
    return next((o for o in options if o.lower() == value.strip().lower()), None)


def predict(bundle, age, income, gender, occupation):
    customer = pd.DataFrame([{"Age": age, "Income": np.log1p(income), "Gender": gender,
                              "Occupation": occupation}])
    return float(bundle["model"].predict_proba(customer)[0, 1])


def main():
    parser = argparse.ArgumentParser(description="Transaction prediction demo")
    parser.add_argument("--age", type=float)
    parser.add_argument("--income", type=float, help="annual income in rupees")
    parser.add_argument("--gender", choices=GENDERS)
    parser.add_argument("--occupation")
    args = parser.parse_args()

    bundle = load_model()
    print(DEMO_NOTE)
    print()

    if None in (args.age, args.income, args.gender, args.occupation):
        age = ask("Age (18-100): ", float, lambda v: 18 <= v <= 100, "Enter a number between 18 and 100.")
        income = ask("Annual income in rupees: ", lambda s: float(s.replace(",", "")), lambda v: v >= 0,
                     "Enter a number of 0 or more, for example 600000.")
        gender = pick("Gender:", GENDERS)
        occupation = pick("Occupation:", OCCUPATIONS, allow_other=True)
    else:
        if not 18 <= args.age <= 100 or args.income < 0:
            parser.error("age must be 18-100 and income 0 or more")
        age, income, gender = args.age, args.income, args.gender
        occupation = normalise(args.occupation, OCCUPATIONS) or "Other"

    probability = predict(bundle, age, income, gender, occupation)
    print()
    print(f"Customer: age {age:.0f}, income Rs {income:,.0f}, {gender}, {occupation}")
    print(f"Transaction Prediction Probability: {probability * 100:.1f}%")
    print(f"(Average customer in the demo data: {bundle['base_rate'] * 100:.1f}%. Demo model ROC-AUC: {bundle['test_auc']:.3f})")
    print(DEMO_NOTE)


if __name__ == "__main__":
    main()
