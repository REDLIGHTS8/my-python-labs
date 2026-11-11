
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = [
    [879, 2.80, 311, 250, 146, 198, 114, 34.7, 3090, 23.0, 19.0],
    [1533, 3.24, 250, 420, 57, 212, 212, 30.9, 3388, 23.9, 20.0],
    [2526, 3.66, 220, 400, 18.7, 260, 103, 41.4, 7525, 56.9, 46.5],
    [1646, 1.55, 307, 200, 62, 243, 582, 33.8, 3795, 27.4, 23.4],
    [1997, 3.26, 360, 400, 123, 115, 2978, 3.60, 0.40, 22.9, 25.3],
    [6005, 2.60, 275, 420, 21.0, 166, 940, 20.2, 0.00, 18.7, 50.9],
    [1824, 2.96, 343, 120, 19.68, 160, 858, 10.4, 0.00, 44.3, 36.5],
    [2693, 2.53, 280, 420, 9.92, 128, 206, 37.8, 5360, 40.2, 33.1],
    [3416, 3.00, 190, 800, 14.0, 168, 509, 34.2, 6514, 47.5, 40.2],
    [1386, 2.44, 258, 439, 27.0, 93, 194, 24.7, 4995, 37.8, 30.7],
    [2547, 3.00, 330, 335, 39.5, 222, 164, 67.0, 4889, 35.9, 30.1],
    [4217, 2.75, 250, 520, 16.0, 190, 439, 55.8, 7361, 54.7, 45.4],
    [3506, 3.17, 247, 320, 15.0, 195, 409, 40.3, 4878, 36.1, 30.0],
    [2531, 3.63, 280, 380, 29.5, 179, 222, 37.1, 5344, 39.4, 32.8],
    [2229, 3.09, 260, 430, 10.0, 28, 177, 45.8, 5938, 43.4, 36.2],
    [2161, 2.99, 240, 90.0, 27.0, 251, 58, 68.7, 5268, 39.0, 32.4],
    [3169, 2.97, 126, 400, 4.50, 313, 325, 43.8, 8299, 62.3, 51.1],
    [3552, 2.10, 88.0, 475, 29.0, 178, 93, 99.8, 7041, 52.0, 43.1],
    [1332, 2.72, 310, 92.0, 12.0, 208, 117, 57.2, 5155, 40.6, 31.8],
    [3419, 3.18, 145, 250, 5.30, 445, 241, 56.6, 9011, 66.8, 55.5],
    [1194, 3.03, 290, 165, 58.0, 184, 150, 31.9, 4697, 35.8, 28.7],
    [2828, 2.86, 330, 260, 29.0, 257, 219, 48.0, 7008, 51.5, 42.7],
    [2748, 2.17, 502, 180, 17.0, 329, 613, 36.4, 5467, 36.3, 31.5],
    [1196, 3.21, 263, 485, 7.00, 193, 127, 39.6, 3535, 25.7, 21.7],
    [2265, 3.69, 100, 128, 1.00, 177, 11, 105.1, 10644, 76.7, 65.7],
    [1241, 1.80, 720, 320, 43.0, 294, 432, 23.6, 2768, 19.2, 16.7],
    [3217, 1.77, 420, 400, 7.00, 465, 920, 36.3, 5225, 37.0, 32.1],
    [1997, 1.82, 360, 220, 8.00, 311, 643, 31.0, 5167, 38.1, 31.5],
    [1983, 1.66, 230, 474, 1.00, 414, 742, 31.9, 8393, 45.8, 39.1],
    [842, 3.47, 289, 250, 21.0, 165, 175, 23.6, 3810, 28.4, 23.2]
]

columns = [
    "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
    "y1", "y2", "y3"
]

df = pd.DataFrame(data, columns=columns)
df.index = range(1, len(df) + 1)
df.index.name = "Номер шахты"

print("\nИсходные данные:")
print(df.to_string())


X = df[["x1", "x2", "x3", "x4",
        "x5", "x6", "x7", "x8"]]

Y = df[["y1", "y2", "y3"]]

print("\nКоличество шахт:", len(df))
print("Количество признаков X:", X.shape[1])
print("Количество показателей Y:", Y.shape[1])

correlation = df.corr()

print("\nМатрица корреляций:")
print(correlation.round(3))


X_std = (X - X.mean()) / X.std(ddof=1)
Y_std = (Y - Y.mean()) / Y.std(ddof=1)

X_std = X_std.to_numpy()
Y_std = Y_std.to_numpy()

Sxx = np.cov(X_std, rowvar=False)
Syy = np.cov(Y_std, rowvar=False)
Sxy = X_std.T @ Y_std / (len(df) - 1)
Syx = Sxy.T

M = np.linalg.pinv(Sxx) @ Sxy @ np.linalg.pinv(Syy) @ Syx

eigenvalues, eigenvectors = np.linalg.eig(M)

order = np.argsort(eigenvalues.real)[::-1]
eigenvalues = eigenvalues.real[order]
eigenvectors = eigenvectors.real[:, order]

canonical_correlations = np.sqrt(
    np.clip(eigenvalues[:3], 0, 1)
)

A = eigenvectors[:, :3]

for i in range(3):
    variance_u = A[:, i].T @ Sxx @ A[:, i]
    A[:, i] /= np.sqrt(variance_u)

B = np.linalg.pinv(Syy) @ Syx @ A

for i in range(3):
    if canonical_correlations[i] > 1e-10:
        B[:, i] /= canonical_correlations[i]

    variance_v = B[:, i].T @ Syy @ B[:, i]

    if variance_v > 1e-10:
        B[:, i] /= np.sqrt(variance_v)

U = X_std @ A
V = Y_std @ B

print("\nКАНОНИЧЕСКИЕ КОРРЕЛЯЦИИ")
print("=" * 50)

for i, r in enumerate(canonical_correlations, start=1):
    print(f"Каноническая корреляция {i}: {r:.4f}")

print("\nКоэффициенты для X:")
print(pd.DataFrame(
    A,
    index=X.columns,
    columns=["Функция 1", "Функция 2", "Функция 3"]
).round(4))

print("\nКоэффициенты для Y:")
print(pd.DataFrame(
    B,
    index=Y.columns,
    columns=["Функция 1", "Функция 2", "Функция 3"]
).round(4))


plt.figure(figsize=(12, 9))

plt.imshow(
    correlation,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    aspect="auto"
)

plt.colorbar(label="Коэффициент корреляции")

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.yticks(
    range(len(correlation.index)),
    correlation.index
)

for i in range(len(correlation)):
    for j in range(len(correlation.columns)):
        plt.text(
            j, i,
            f"{correlation.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=8
        )

plt.title("Карта теплоты корреляций показателей шахт")
plt.tight_layout()


plt.figure(figsize=(8, 6))

plt.scatter(U[:, 0], V[:, 0], s=60)

for i in range(len(df)):
    plt.annotate(
        str(i + 1),
        (U[i, 0], V[i, 0]),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8
    )

plt.xlabel("Первая каноническая переменная U1")
plt.ylabel("Первая каноническая переменная V1")
plt.title(
    f"Диаграмма рассеивания\n"
    f"Каноническая корреляция = "
    f"{canonical_correlations[0]:.3f}"
)

plt.axhline(0, color="gray", linewidth=0.7)
plt.axvline(0, color="gray", linewidth=0.7)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()
