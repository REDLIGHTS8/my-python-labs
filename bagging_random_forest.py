
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, validation_curve, learning_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


data = load_breast_cancer()

X = data.data
y = data.target

feature_names = [
    "Средний радиус",
    "Средняя текстура",
    "Средний периметр",
    "Средняя площадь",
    "Средняя гладкость",
    "Средняя компактность",
    "Средняя вогнутость",
    "Средние вогнутые точки",
    "Средняя симметрия",
    "Средняя фрактальность",
    "Ошибка радиуса",
    "Ошибка текстуры",
    "Ошибка периметра",
    "Ошибка площади",
    "Ошибка гладкости",
    "Ошибка компактности",
    "Ошибка вогнутости",
    "Ошибка вогнутых точек",
    "Ошибка симметрии",
    "Ошибка фрактальности",
    "Максимальный радиус",
    "Максимальная текстура",
    "Максимальный периметр",
    "Максимальная площадь",
    "Максимальная гладкость",
    "Максимальная компактность",
    "Максимальная вогнутость",
    "Максимум вогнутых точек",
    "Максимальная симметрия",
    "Максимальная фрактальность"
]

class_names = ["Злокачественная", "Доброкачественная"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Количество записей:", len(X))
print("Обучающая выборка:", len(X_train))
print("Тестовая выборка:", len(X_test))



def create_bagging(n_estimators=100):

    base_tree = DecisionTreeClassifier(
        max_depth=None,
        random_state=42
    )

    try:
        model = BaggingClassifier(
            estimator=base_tree,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )

    except TypeError:
        model = BaggingClassifier(
            base_estimator=base_tree,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )

    return model



tree = DecisionTreeClassifier(
    max_depth=None,
    random_state=42
)

bagging = create_bagging(100)
forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

models = {
    "Дерево решений": tree,
    "Бэггинг": bagging,
    "Случайный лес": forest
}


accuracies = {}

for name, model in models.items():

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    accuracies[name] = accuracy

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    print("Точность:", round(accuracy, 4))

    print(
        classification_report(
            y_test,
            predictions,
            target_names=class_names
        )
    )


fig1, axes = plt.subplots(
    2,
    2,
    figsize=(15, 10),
    constrained_layout=True
)

fig1.suptitle(
    "Анализ методов классификации",
    fontsize=16,
    fontweight="bold"
)


ax = axes[0, 0]

for class_id, class_name in enumerate(class_names):

    ax.scatter(
        X[y == class_id, 0],
        X[y == class_id, 1],
        label=class_name,
        alpha=0.65,
        s=25
    )

ax.set_title("Распределение объектов")
ax.set_xlabel("Средний радиус")
ax.set_ylabel("Средняя текстура")
ax.legend()
ax.grid(alpha=0.3)


ax = axes[0, 1]

plot_tree(
    tree,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=6,
    ax=ax
)

ax.set_title("Дерево решений")

ax = axes[1, 0]
model_names = list(accuracies.keys())
accuracy_values = list(accuracies.values())

bars = ax.bar(
    model_names,
    accuracy_values,
    alpha=0.8
)

ax.set_title("Сравнение точности моделей")
ax.set_ylabel("Точность")
ax.set_xlabel("Метод")
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)

for bar, value in zip(bars, accuracy_values):

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.015,
        f"{value:.3f}",
        ha="center",
        fontsize=10
    )


ax = axes[1, 1]

class_counts = [
    np.sum(y == 0),
    np.sum(y == 1)
]

ax.pie(
    class_counts,
    labels=class_names,
    autopct="%1.1f%%",
    startangle=90
)

ax.set_title("Распределение классов")


def draw_validation_curve(
    ax,
    model,
    X_data,
    y_data,
    parameter_name,
    parameter_values,
    title,
    labels=None
):

    train_scores, validation_scores = validation_curve(
        model,
        X_data,
        y_data,
        param_name=parameter_name,
        param_range=parameter_values,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    validation_mean = np.mean(validation_scores, axis=1)
    validation_std = np.std(validation_scores, axis=1)

    x_values = [
        21 if value is None else value
        for value in parameter_values
    ]

    ax.plot(
        x_values,
        train_mean,
        marker="o",
        linewidth=2,
        label="Обучающая выборка"
    )

    ax.fill_between(
        x_values,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15
    )

    ax.plot(
        x_values,
        validation_mean,
        marker="o",
        linewidth=2,
        label="Проверочная выборка"
    )

    ax.fill_between(
        x_values,
        validation_mean - validation_std,
        validation_mean + validation_std,
        alpha=0.15
    )

    ax.set_title(title)
    ax.set_xlabel(parameter_name)
    ax.set_ylabel("Точность")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)

    if labels is not None:
        ax.set_xticks(x_values)
        ax.set_xticklabels(labels)

    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)



def draw_learning_curve(
    ax,
    model,
    X_data,
    y_data,
    title
):

    train_sizes, train_scores, validation_scores = learning_curve(
        model,
        X_data,
        y_data,
        train_sizes=np.linspace(0.05, 1.0, 20),
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    validation_mean = np.mean(validation_scores, axis=1)
    validation_std = np.std(validation_scores, axis=1)
    x_values = np.concatenate(([0], train_sizes))
    train_mean = np.concatenate(([0], train_mean))
    validation_mean = np.concatenate(([0], validation_mean))
    train_std = np.concatenate(([0], train_std))
    validation_std = np.concatenate(([0], validation_std))

    ax.plot(
        x_values,
        train_mean,
        marker="o",
        markersize=3,
        linewidth=2,
        label="Обучающая выборка"
    )

    ax.fill_between(
        x_values,
        np.maximum(0, train_mean - train_std),
        np.minimum(1, train_mean + train_std),
        alpha=0.12
    )

    ax.plot(
        x_values,
        validation_mean,
        marker="o",
        markersize=3,
        linewidth=2,
        label="Проверочная выборка"
    )

    ax.fill_between(
        x_values,
        np.maximum(0, validation_mean - validation_std),
        np.minimum(1, validation_mean + validation_std),
        alpha=0.12
    )

    ax.set_title(title)
    ax.set_xlabel("Количество обучающих объектов")
    ax.set_ylabel("Точность")
    ax.set_xlim(0, max(train_sizes) * 1.05)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)


fig2, axes2 = plt.subplots(
    3,
    2,
    figsize=(14, 11),
    constrained_layout=True
)

fig2.suptitle(
    "Кривые валидации и обучения трёх моделей",
    fontsize=16,
    fontweight="bold"
)


tree_depths = [1, 2, 3, 4, 5, 7, 10, 15, 20, None]
tree_depth_labels = [
    "1", "2", "3", "4", "5",
    "7", "10", "15", "20", "∞"
]

draw_validation_curve(
    axes2[0, 0],
    DecisionTreeClassifier(random_state=42),
    X_train,
    y_train,
    "max_depth",
    tree_depths,
    "Дерево решений: кривая валидации",
    tree_depth_labels
)

draw_learning_curve(
    axes2[0, 1],
    DecisionTreeClassifier(
        max_depth=None,
        random_state=42
    ),
    X_train,
    y_train,
    "Дерево решений: кривая обучения"
)

estimators_values = [1, 5, 10, 25, 50, 75, 100]
draw_validation_curve(
    axes2[1, 0],
    create_bagging(),
    X_train,
    y_train,
    "n_estimators",
    estimators_values,
    "Бэггинг: кривая валидации"
)

draw_learning_curve(
    axes2[1, 1],
    create_bagging(100),
    X_train,
    y_train,
    "Бэггинг: кривая обучения"
)

draw_validation_curve(
    axes2[2, 0],
    RandomForestClassifier(
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ),
    X_train,
    y_train,
    "n_estimators",
    estimators_values,
    "Случайный лес: кривая валидации"
)

draw_learning_curve(
    axes2[2, 1],
    RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ),
    X_train,
    y_train,
    "Случайный лес: кривая обучения"
)



plt.show()
