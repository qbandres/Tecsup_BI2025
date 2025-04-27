import pandas as pd
import numpy as np

data = pd.read_csv("data/Advertising.csv",sep=",")
data.head()

import statsmodels.formula.api as smf
lm = smf.ols(formula="Sales~TV", data = data).fit()
print(lm.params)


print(lm.params)
print(lm.summary())

data["sales_pred"] = 7.032594 + 0.047537*data["TV"]
print(data)

sales_pred = lm.predict(pd.DataFrame(data["TV"]))
print(sales_pred)

print(lm.summary())

lm = smf.ols(formula="Sales~TV+Radio", data = data). fit()
print(lm.params)