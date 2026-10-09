# Практическая работа №2: Дискриминативные интеллектуальные алгоритмы
# Задача: бинарная классификация оттока клиентов (Churn), данные Telco Customer Churn из ПР1
# Каждый график сохраняется в файл pr2_figN_*.png рядом со скриптом (для вставки в отчёт).

# Шаг 1: Импорт библиотек и настройки
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              VotingClassifier, StackingClassifier)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
from imblearn.over_sampling import SMOTENC

sns.set_theme(style="whitegrid")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 250)

PATH = 'C:\\A) НЕ УДАЛЯТЬ\\Математика для программирования\\Мои отчёты\\ПР2\\WA_Fn-UseC_-Telco-Customer-Churn.csv'
num = ['tenure', 'MonthlyCharges', 'TotalCharges']


def show(name):
    """Сохранить текущий график в файл и показать на экране."""
    plt.savefig(f"pr2_{name}.png", dpi=150, bbox_inches='tight')
    plt.show()


# Шаг 2: СЫРЫЕ данные (без предобработки)
# Минимум, без которого модели не запустятся: TotalCharges -> число,
# строки с пропусками удаляются, customerID убирается (иначе get_dummies создаст ~7000 столбцов),
# категории кодируются get_dummies. Нет: заполнения пропусков, удаления дубликатов,
# объединения категорий, стандартизации и аугментации.
raw = pd.read_csv(PATH)
raw['TotalCharges'] = pd.to_numeric(raw['TotalCharges'], errors='coerce')
raw = raw.dropna().drop(columns='customerID').reset_index(drop=True)
y_raw = (raw['Churn'] == 'Yes').astype(int).values
X_raw = pd.get_dummies(raw.drop(columns='Churn'), drop_first=True, dtype=int)

Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
    X_raw, y_raw, test_size=.2, stratify=y_raw, random_state=42)
print("СЫРЫЕ данные: всего", raw.shape, "| train", Xtr_r.shape, "test", Xte_r.shape,
      "| классы в train:", np.bincount(ytr_r))


# Шаг 3: ПРЕДОБРАБОТАННЫЕ данные
df = pd.read_csv(PATH)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df = df.drop(columns='customerID').drop_duplicates().reset_index(drop=True)
df = df.replace({'No internet service': 'No', 'No phone service': 'No'})
print("ПРЕДОБРАБОТАННЫЕ данные после очистки:", df.shape)

y = (df['Churn'] == 'Yes').astype(int).values
X = pd.get_dummies(df.drop(columns='Churn'), drop_first=True, dtype=int)
print("Признаков после кодирования:", X.shape[1])

# 3.1 Баланс классов и корреляции (оценка до обучения)
print("Баланс классов:\n", df['Churn'].value_counts(normalize=True).round(3))

plt.figure(figsize=(6, 4.5))
sns.countplot(data=df, x='Churn'); plt.title('Баланс классов')
show("fig1_balance")

corr = X.assign(Churn=y).corr()['Churn'].drop('Churn').sort_values()
plt.figure(figsize=(8, 9)); corr.plot(kind='barh'); plt.title('Корреляция признаков с оттоком')
plt.tight_layout(); show("fig2_corr")
print("Самые сильные корреляции с оттоком:\n", corr.round(2).iloc[[0, 1, 2, -3, -2, -1]])

# 3.2 Разбиение, стандартизация, аугментация
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)

# Стандартизация: параметры считаются только по train
sc = StandardScaler().fit(X_tr[num])
X_tr, X_te = X_tr.copy(), X_te.copy()
X_tr[num] = sc.transform(X_tr[num])
X_te[num] = sc.transform(X_te[num])

# Аугментация SMOTENC только для обучающей выборки; тест остаётся настоящим
cat_idx = [i for i, c in enumerate(X_tr.columns) if c not in num]
X_res, y_res = SMOTENC(categorical_features=cat_idx, random_state=42).fit_resample(X_tr, y_tr)
print("ПРЕДОБРАБОТАННЫЕ данные: train до", X_tr.shape, np.bincount(y_tr),
      "после", X_res.shape, np.bincount(y_res),
      "| test", X_te.shape, "классы в test:", np.bincount(y_te))
print("Доля класса No в тесте (точность тривиального прогноза 'никто не уйдёт'):",
      round((y_te == 0).mean(), 3))


# Шаг 4: Модели разных классов + ансамбли
def make_models():
    lr  = LogisticRegression(max_iter=1000, random_state=42)
    svm = SVC(probability=True, random_state=42)
    rf  = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),   # линейная
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),                            # метрическая
        "SVM (RBF)": SVC(probability=True, random_state=42),                         # ядровая
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),       # дерево
        "Gaussian NB": GaussianNB(),                                                 # вероятностная
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10,
                                                random_state=42, n_jobs=-1),         # бэггинг
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),            # бустинг
        "Voting (soft)": VotingClassifier(                                           # ансамбль из 3 моделей
            estimators=[("lr", lr), ("svm", svm), ("rf", rf)], voting="soft"),
        "Stacking": StackingClassifier(                                              # стекинг
            estimators=[("lr", lr), ("svm", svm), ("rf", rf)],
            final_estimator=LogisticRegression(max_iter=1000), cv=5),
    }


def run_all(Xtr, Xte, ytr, yte, label):
    rows, preds, probas = [], {}, {}
    print(f"\n=== Обучение на {label} данных ===")
    for name, m in make_models().items():
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        proba = m.predict_proba(Xte)[:, 1]
        rows.append({
            "Модель": name,
            "Accuracy": accuracy_score(yte, pred),
            "Precision (Yes)": precision_score(yte, pred, zero_division=0),
            "Recall (Yes)": recall_score(yte, pred),
            "F1 (Yes)": f1_score(yte, pred),
            "F1 macro": f1_score(yte, pred, average='macro'),
            "ROC-AUC": roc_auc_score(yte, proba),
        })
        preds[name], probas[name] = pred, proba
        print(f"{name}: готово")
    return pd.DataFrame(rows).set_index("Модель"), preds, probas


res_raw, pred_raw, proba_raw = run_all(Xtr_r, Xte_r, ytr_r, yte_r, "СЫРЫХ")
res_prep, pred_prep, proba_prep = run_all(X_res, X_te, y_res, y_te, "ПРЕДОБРАБОТАННЫХ")


# Шаг 5: Таблицы результатов
print("\nМЕТРИКИ НА СЫРЫХ ДАННЫХ:")
print(res_raw.round(3).to_string())
print("\nМЕТРИКИ НА ПРЕДОБРАБОТАННЫХ ДАННЫХ:")
print(res_prep.round(3).to_string())

delta = (res_prep - res_raw)[["Accuracy", "Recall (Yes)", "F1 (Yes)", "ROC-AUC"]]
print("\nВКЛАД ПРЕДОБРАБОТКИ (предобработанные минус сырые):")
print(delta.round(3).to_string())

res_raw.round(4).to_csv("pr2_results_raw.csv")
res_prep.round(4).to_csv("pr2_results_prep.csv")

best = res_prep["F1 (Yes)"].idxmax()
print(f"\nЛучшая модель по F1 (Yes) на предобработанных данных: {best}, "
      f"F1 = {res_prep.loc[best, 'F1 (Yes)']:.3f}")


# Шаг 6: Графики
# 6.1 Сырые vs предобработанные данные
cmp = pd.DataFrame({"Сырые": res_raw["F1 (Yes)"], "Предобработанные": res_prep["F1 (Yes)"]})
cmp.plot(kind="bar", figsize=(12, 6), edgecolor="black")
plt.title("F1 (класс Yes): сырые и предобработанные данные")
plt.ylabel("F1"); plt.xticks(rotation=45, ha="right"); plt.ylim(0, 1)
plt.tight_layout(); show("fig3_raw_vs_prep")

# 6.2 Сравнение всех моделей на предобработанных данных
res_prep.sort_values("F1 (Yes)")[["Accuracy", "Recall (Yes)", "F1 (Yes)", "ROC-AUC"]].plot(
    kind="barh", figsize=(10, 8), edgecolor="black")
plt.title("Сравнение моделей на предобработанных данных")
plt.xlabel("Значение метрики"); plt.xlim(0, 1)
plt.tight_layout(); show("fig4_all_models")

# 6.3 ROC-кривые на предобработанных данных
plt.figure(figsize=(8, 7))
for name, p in proba_prep.items():
    fpr, tpr, _ = roc_curve(y_te, p)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {res_prep.loc[name, 'ROC-AUC']:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label="случайный классификатор")
plt.xlabel("Доля ложных срабатываний (FPR)"); plt.ylabel("Полнота (TPR)")
plt.title("ROC-кривые (предобработанные данные)"); plt.legend(loc="lower right", fontsize=8)
plt.tight_layout(); show("fig5_roc")

# 6.4 Матрицы ошибок лучшей модели: сырые и предобработанные данные
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for a, yt, yp, title in [(ax[0], yte_r, pred_raw[best], "Сырые данные"),
                         (ax[1], y_te, pred_prep[best], "Предобработанные данные")]:
    sns.heatmap(confusion_matrix(yt, yp), annot=True, fmt='d', cmap='Blues', ax=a,
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    a.set_title(f"{best}: {title}"); a.set_xlabel("Предсказано"); a.set_ylabel("Истинно")
plt.tight_layout(); show("fig6_confusion")