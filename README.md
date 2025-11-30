# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
Date:
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("/content/gold.csv", parse_dates=["Date"], index_col="Date")

data["Price"] = data["Price"].str.replace(",", "").astype(float)

resampled_data = data["Price"].resample("Y").mean().to_frame()
resampled_data.index = resampled_data.index.year
resampled_data.reset_index(inplace=True)
resampled_data.rename(columns={"Date": "Year"}, inplace=True)

years = resampled_data["Year"].tolist()
prices = resampled_data["Price"].tolist()

X = [i - years[len(years) // 2] for i in years]
x2 = [i ** 2 for i in X]
xy = [i * j for i, j in zip(X, prices)]

n = len(years)
b = (n * sum(xy) - sum(prices) * sum(X)) / (n * sum(x2) - (sum(X) ** 2))
a = (sum(prices) - b * sum(X)) / n
linear_trend = [a + b * X[i] for i in range(n)]

x3 = [i ** 3 for i in X]
x4 = [i ** 4 for i in X]
x2y = [i * j for i, j in zip(x2, prices)]

coeff = [[len(X), sum(X), sum(x2)],
         [sum(X), sum(x2), sum(x3)],
         [sum(x2), sum(x3), sum(x4)]]

Y = [sum(prices), sum(xy), sum(x2y)]
A = np.array(coeff)
B = np.array(Y)

solution = np.linalg.solve(A, B)
a_poly, b_poly, c_poly = solution
poly_trend = [a_poly + b_poly * X[i] + c_poly * (X[i] ** 2) for i in range(n)]

print(f"Linear Trend: y={a:.2f} + {b:.2f}x")
print(f"Polynomial Trend: y={a_poly:.2f} + {b_poly:.2f}x + {c_poly:.2f}x²")

resampled_data["Linear Trend"] = linear_trend
resampled_data["Polynomial Trend"] = poly_trend
resampled_data.set_index("Year", inplace=True)

# A - LINEAR TREND ESTIMATION

plt.figure(figsize=(8, 5))
resampled_data["Linear Trend"].plot(color="black", linestyle="--", marker="o")
plt.xlabel("Year")
plt.ylabel("Gold Price (Linear Trend)")
plt.title("Gold Price Linear Trend")
plt.show()

# B- POLYNOMIAL TREND ESTIMATION

plt.figure(figsize=(8, 5))
resampled_data["Polynomial Trend"].plot(color="red", marker="o")
plt.xlabel("Year")
plt.ylabel("Gold Price (Polynomial Trend)")
plt.title("Gold Price Polynomial Trend")
plt.show()
```
### OUTPUT
A - LINEAR TREND ESTIMATION
<img width="880" height="581" alt="image" src="https://github.com/user-attachments/assets/946f8429-8393-4db4-b71a-0c817426e63e" />

B- POLYNOMIAL TREND ESTIMATION
<img width="881" height="582" alt="image" src="https://github.com/user-attachments/assets/1c034fc4-985d-4760-aba7-971862d597be" />

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
