# Шаг 1: Загрузка и первичный анализ
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTENC

sns.set_theme(style="whitegrid")

df = pd.read_csv('C:\\A) НЕ УДАЛЯТЬ\\Математика для программирования\\Мои отчёты\\ПР1\\WA_Fn-UseC_-Telco-Customer-Churn.csv')  # Загрузка данных из CSV файла
print("Размер:", df.shape)
print(df.head())
print(df.dtypes)
print("Дубликатов:", df.duplicated().sum())
print("Пропуски (NaN):", df.isnull().sum().sum())
print("Баланс классов:\n", df['Churn'].value_counts(normalize=True).round(3))



# Шаг 2: Очистка данных
# 2.1 TotalCharges: текст -> число; пустые строки станут NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print("Пропусков в TotalCharges:", df['TotalCharges'].isna().sum())
print("tenure у этих строк:", df.loc[df['TotalCharges'].isna(), 'tenure'].unique())

# 2.2 Пропуски у клиентов с tenure=0 (ещё ничего не платили) -> логично 0
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# 2.3 Лишний признак и дубликаты
df = df.drop(columns='customerID')
print("Дубликатов после удаления customerID:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

# 2.4 Объединение избыточных категорий
df = df.replace({'No internet service': 'No', 'No phone service': 'No'})

# 2.5 Проверка выбросов по IQR
for c in ['tenure', 'MonthlyCharges', 'TotalCharges']:
    q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
    n_out = ((df[c] < q1 - 1.5*iqr) | (df[c] > q3 + 1.5*iqr)).sum()
    print(f"{c}: выбросов {n_out}")
print("Размер после очистки:", df.shape)



# Шаг 3: Визуализация и корреляции
sns.countplot(data=df, x='Churn'); plt.title('Баланс классов'); plt.show()

num = ['tenure', 'MonthlyCharges', 'TotalCharges']
sns.pairplot(df.sample(1500, random_state=42)[num + ['Churn']], hue='Churn',
             plot_kws={'alpha': .4}); plt.show()

for c in num:
    sns.boxplot(data=df, x='Churn', y=c); plt.title(f'{c} по классам'); plt.show()

# 3.1 Границы усов boxplot внутри каждого класса (точки выше усов на рис. 9 и 11)
for c in ['tenure', 'TotalCharges']:
    for cls, g in df.groupby('Churn')[c]:
        q1, q3 = g.quantile([.25, .75]); iqr = q3 - q1
        print(c, cls, 'верхняя граница усов:', round(q3 + 1.5*iqr, 1),
              'точек выше:', (g > q3 + 1.5*iqr).sum())



# Шаг 4: Кодирование и корреляция с целевой переменной
y = (df['Churn'] == 'Yes').astype(int).values
X = df.drop(columns='Churn')

# бинарные признаки (Yes/No, Male/Female) -> 0/1
for c in [c for c in X.columns if X[c].dtype == object and X[c].nunique() == 2]:
    X[c] = (X[c] == sorted(X[c].unique())[-1]).astype(int)

# остальные категории (Contract, InternetService, PaymentMethod) -> one-hot
X = pd.get_dummies(X, drop_first=True, dtype=int)
print("Признаков после кодирования:", X.shape[1])

corr = X.assign(Churn=y).corr()['Churn'].drop('Churn').sort_values()
plt.figure(figsize=(8, 9)); corr.plot(kind='barh'); plt.title('Корреляция признаков с оттоком')
plt.tight_layout(); plt.show()
print(corr.round(2))



# Шаг 5: Разделение, стандартизация, аугментация
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)

sc = StandardScaler().fit(X_tr[num])
X_tr, X_te = X_tr.copy(), X_te.copy()
X_tr[num] = sc.transform(X_tr[num])
X_te[num] = sc.transform(X_te[num])

cat_idx = [i for i, c in enumerate(X_tr.columns) if c not in num]
X_res, y_res = SMOTENC(categorical_features=cat_idx, random_state=42).fit_resample(X_tr, y_tr)

print("Классы до:", np.bincount(y_tr), "после:", np.bincount(y_res))
print("Добавлено синтетических:", len(X_res) - len(X_tr))
is_synth = np.arange(len(X_res)) >= len(X_tr)



# Шаг 6: Снижение размерности
pca_full = PCA(random_state=42).fit(X_res)
cum = pca_full.explained_variance_ratio_.cumsum()
print("Накопленная дисперсия (первые 10):", cum[:10].round(3))
print("Компонент для 80%:", int(np.argmax(cum >= .80)) + 1)

Z = PCA(n_components=2, random_state=42).fit_transform(X_res)

idx = np.random.default_rng(42).choice(len(X_res), 3000, replace=False)
Z2 = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_res.iloc[idx])

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
for a, Zx, ys, sy, name in [(ax[0], Z, y_res, is_synth, 'PCA'),
                            (ax[1], Z2, y_res[idx], is_synth[idx], 't-SNE (выборка 3000)')]:
    a.scatter(*Zx[~sy].T, c=ys[~sy], cmap='coolwarm', s=10, alpha=.5)
    a.scatter(*Zx[sy].T, c='green', marker='x', s=12, alpha=.6, label='синтетика (SMOTENC)')
    a.set_title(name); a.legend()
plt.show()