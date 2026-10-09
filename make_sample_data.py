"""Generates two small sample datasets into Datasets/ for trying the app."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 600
dept = rng.choice(["IT", "Sales", "HR", "Finance", "Marketing"], n, p=[.3, .25, .1, .15, .2])
base = {"IT": 70000, "Sales": 55000, "HR": 48000, "Finance": 65000, "Marketing": 52000}
age = rng.integers(22, 60, n)
salary = np.array([base[d] for d in dept]) + age * 600 + rng.normal(0, 6000, n)
salary[:5] = salary[:5] * 4  # outliers
emp = pd.DataFrame({
    "Employee_ID": np.arange(1000, 1000 + n), "Department": dept, "Age": age, "Salary": salary.round(2),
    "City": rng.choice(["Chennai", "Mumbai", "Delhi", "Bengaluru"], n),
    "Join_Date": pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.integers(0, 1400, n), unit="D"),
    "Churn": rng.choice(["Yes", "No"], n, p=[.2, .8]),
    "Feedback": rng.choice(["slow response from support team", "billing issue unresolved again", "great product and support",
                            "delivery was late and packaging damaged"], n),
})
emp.loc[rng.choice(n, 25, replace=False), "Salary"] = np.nan
emp = pd.concat([emp, emp.head(7)])
emp.to_csv("Datasets/employees_sample.csv", index=False)

m = 3000
retail = pd.DataFrame({
    "InvoiceNo": rng.integers(500000, 501000, m).astype(str),
    "Description": rng.choice(["WHITE MUG", "RED LAMP", "BLUE VASE", "GREEN CLOCK", "PINK BAG", "TEA SET"], m),
    "Quantity": rng.integers(1, 20, m), "UnitPrice": rng.choice([1.25, 2.5, 4.95, 7.5, 12.0], m),
    "InvoiceDate": pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 700, m), unit="D"),
    "CustomerID": rng.integers(12000, 12400, m).astype(float),
    "Country": rng.choice(["United Kingdom", "France", "Germany", "India", "Spain"], m, p=[.6, .1, .1, .1, .1]),
})
retail.to_excel("Datasets/online_retail_sample.xlsx", index=False)
print("written")
