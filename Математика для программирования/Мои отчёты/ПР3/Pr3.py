# Практическая работа №3: Генерация табличных данных
# Задача: синтез таблицы Telco Customer Churn (данные и предобработка из ПР1) генеративной моделью GMM
# (Gaussian Mixture Model) и анализ распределения синтетических данных методами ПР1.

# Шаг 1: Импорт библиотек и настройки
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sns.set_theme(style="whitegrid")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 250)

PATH = os.environ.get(
    'TELCO_PATH',
    'C:\\A) НЕ УДАЛЯТЬ\\Математика для программирования\\Мои отчёты\\ПР1\\WA_Fn-UseC_-Telco-Customer-Churn.csv')
SEED = 42
GMM_GRID = [1, 2, 4, 8, 12, 16]     # перебор числа компонент по BIC
num = ['tenure', 'MonthlyCharges', 'TotalCharges']
SHOW = os.environ.get('MPLBACKEND', '') != 'Agg'


def show(name):
    """Показать текущий график на экране (name - номер рисунка для отчёта)."""
    if SHOW:
        plt.show()
    plt.close()


# Шаг 2: Предобработанные данные (как в ПР1/ПР2)
df = pd.read_csv(PATH)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df = df.drop(columns='customerID').drop_duplicates().reset_index(drop=True)
df = df.replace({'No internet service': 'No', 'No phone service': 'No'})
df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)       # 0/1 - это категория
COLS = list(df.columns)
CAT = [c for c in COLS if c not in num]                      # включая целевую Churn
CATS = {c: sorted(df[c].unique()) for c in CAT}
print("Предобработанные данные:", df.shape)
print("Баланс классов (настоящие):\n", df['Churn'].value_counts(normalize=True).round(3))
LO, HI = df[num].min(), df[num].max()

# Шаг 3: Обучение генеративной модели GMM
# Числовые признаки стандартизуются, категориальные кодируются one-hot (включая Churn),
# после генерации категория определяется по максимальному значению внутри своей группы столбцов.
sc = StandardScaler().fit(df[num])


# Кодирование DataFrame в матрицу для GMM
def encode(d):
    parts = [sc.transform(d[num])]
    for c in CAT:
        col = d[c].astype(str).to_numpy(dtype=object)[:, None]
        parts.append((col == np.array(CATS[c], dtype=object)[None, :]).astype(float))
    return np.hstack(parts)


# Декодирование сгенерированных данных обратно в DataFrame
def decode(M):
    out = pd.DataFrame(sc.inverse_transform(M[:, :len(num)]), columns=num)
    j = len(num)
    for c in CAT:
        k = len(CATS[c])
        out[c] = np.array(CATS[c], dtype=object)[M[:, j:j + k].argmax(1)]
        j += k
    return out[COLS]


M = encode(df)
best, bics = None, {}
for k in GMM_GRID:
    g = GaussianMixture(k, covariance_type='full', reg_covar=1e-3, n_init=2,
                        max_iter=300, random_state=SEED).fit(M)
    bics[k] = g.bic(M)
    print(f"GMM k={k}: BIC = {bics[k]:.0f}")
    if best is None or bics[k] < bics[best[0]]:
        best = (k, g)
print("Выбрано компонент по BIC:", best[0])

plt.figure(figsize=(6, 4.5))
plt.plot(list(bics), list(bics.values()), 'o-')
plt.axvline(best[0], color='r', ls='--', label=f'выбрано k = {best[0]}')
plt.xlabel('Число компонент'); plt.ylabel('BIC'); plt.title('GMM: выбор числа компонент по BIC'); plt.legend()
show("fig1_gmm_bic")

raw = decode(best[1].sample(len(df))[0])        # столько же строк, сколько в настоящих данных
raw.to_csv("pr3_synth_raw.csv", index=False)
print("Сгенерировано строк:", len(raw))
print("Пропусков в сгенерированных данных:", int(raw.isna().sum().sum()), "| столбцов:", raw.shape[1],
      "| типы совпадают с настоящими:", all((raw[c].dtype == df[c].dtype) or (c in num) for c in COLS))

# Шаг 4: Шумовые значения и обработка сгенерированных данных
# Экспертные ограничения предметной области (аналог логических правил "A -> B")
ADDONS = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
pos = df[df['tenure'] > 0]
ratio_real = pos['TotalCharges'] / (pos['tenure'] * pos['MonthlyCharges'])
R_LO, R_HI = ratio_real.quantile(0.005), ratio_real.quantile(0.995)
RULES = ["R1: PhoneService=No -> MultipleLines=No",
         "R2: InternetService=No -> все доп. услуги интернета = No",
         "R3: tenure=0 <=> TotalCharges=0",
         "R4: TotalCharges ~ tenure x MonthlyCharges",
         "R5: значения в допустимых диапазонах (tenure - целое 0..72)"]


# Экспертные ограничения предметной области (аналог логических правил "A -> B")
def violations(d):
    t = d['tenure'].round(); tc = d['TotalCharges']
    ratio = tc / (t.where(t > 0) * d['MonthlyCharges'])
    r1 = (d['PhoneService'] == 'No') & (d['MultipleLines'] != 'No')
    r2 = (d['InternetService'] == 'No') & (d[ADDONS] != 'No').any(axis=1)
    r3 = ((t == 0) & (tc > 1)) | ((t > 0) & (tc <= 0))
    r4 = (t > 0) & ((ratio < R_LO * 0.995) | (ratio > R_HI * 1.005))
    r5 = ((d['tenure'] != d['tenure'].round()) | (d['tenure'] < 0) | (d['tenure'] > 72)
          | (d['MonthlyCharges'] < LO['MonthlyCharges']) | (d['MonthlyCharges'] > HI['MonthlyCharges'])
          | (d['TotalCharges'] < 0) | (d['TotalCharges'] > HI['TotalCharges']))
    v = pd.Series([r.mean() for r in (r1, r2, r3, r4, r5)], index=RULES)
    v['Любое нарушение'] = (r1 | r2 | r3 | r4 | r5).mean()
    return v


# Функция очистки сгенерированных данных по экспертным ограничениям
def repair(d):
    d = d.copy()
    d['tenure'] = d['tenure'].round().clip(0, 72)
    d['MonthlyCharges'] = d['MonthlyCharges'].clip(LO['MonthlyCharges'], HI['MonthlyCharges'])
    d.loc[d['PhoneService'] == 'No', 'MultipleLines'] = 'No'
    d.loc[d['InternetService'] == 'No', ADDONS] = 'No'
    base = d['tenure'] * d['MonthlyCharges']
    r = (d['TotalCharges'].clip(0, HI['TotalCharges']) / base.where(base > 0)).clip(R_LO, R_HI)
    d['TotalCharges'] = np.where(d['tenure'] == 0, 0.0, r * base)
    d['TotalCharges'] = d['TotalCharges'].clip(0, HI['TotalCharges']).round(2)
    d['MonthlyCharges'] = d['MonthlyCharges'].round(2)
    return d[COLS].reset_index(drop=True)


syn = repair(raw)
syn.to_csv("pr3_synth_clean.csv", index=False)
viol = pd.DataFrame({'Настоящие': violations(df), 'GMM до очистки': violations(raw),
                     'GMM после очистки': violations(syn)})
print("\nНарушения экспертных ограничений (доля строк):\n", viol.round(4).to_string())
viol.round(4).to_csv("pr3_violations.csv")

# Выбросы по границам усов (как в ПР1): границы считаются по настоящим данным
q1, q3 = df[num].quantile(.25), df[num].quantile(.75)
wlo, whi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
out_real = ((df[num] < wlo) | (df[num] > whi)).mean() * 100
out_syn = ((syn[num] < wlo) | (syn[num] > whi)).mean() * 100
print("\nДоля выбросов по IQR, %:\n", pd.DataFrame({'Настоящие': out_real, 'GMM': out_syn}).round(2).to_string())
print("Дубликатов в синтетике:", syn.duplicated().sum(), f"({syn.duplicated().mean() * 100:.2f}%)")

# Шаг 5: Анализ распределения (методы ПР1)
# 5.1 Средние, медианы, дисперсия числовых признаков
rows = []
for name, d in [('Настоящие', df), ('GMM', syn)]:
    for c in num:
        rows.append({'Набор': name, 'Признак': c, 'mean': d[c].mean(), 'median': d[c].median(),
                     'std': d[c].std(), 'var': d[c].var(), 'min': d[c].min(), 'max': d[c].max(),
                     'skew': d[c].skew()})
stat = pd.DataFrame(rows)
print("\nСтатистики числовых признаков:\n", stat.round(2).to_string(index=False))
stat.round(3).to_csv("pr3_numeric_stats.csv", index=False)
ks = pd.Series({c: ks_2samp(df[c], syn[c]).statistic for c in num})
print("\nСтатистика Колмогорова-Смирнова (0 - распределения совпадают):\n", ks.round(3).to_string())

# 5.2 Баланс классов
p_real, p_syn = (df['Churn'] == 'Yes').mean(), (syn['Churn'] == 'Yes').mean()
print(f"\nДоля оттока: настоящие {p_real:.3f}, GMM {p_syn:.3f}")
plt.figure(figsize=(6, 4.5))
bal = pd.DataFrame({'Настоящие': df['Churn'].value_counts(normalize=True),
                    'GMM': syn['Churn'].value_counts(normalize=True)}).T * 100
bal.plot(kind='bar', stacked=True, ax=plt.gca(), edgecolor='k'); plt.xticks(rotation=0)
plt.ylabel('%'); plt.title('Баланс классов')
show("fig2_balance")

# 5.3 Распределения числовых признаков и ящики с усами
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for a, c in zip(ax, num):
    sns.kdeplot(df[c], ax=a, label='Настоящие', lw=2.5, color='k')
    sns.kdeplot(syn[c], ax=a, label='GMM'); a.set_title(c)
ax[0].legend()
plt.tight_layout(); show("fig3_numeric_dist")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for a, c in zip(ax, num):
    sns.boxplot(data=pd.DataFrame({'Настоящие': df[c], 'GMM': syn[c]}), ax=a); a.set_title(c)
plt.tight_layout(); show("fig4_boxplots")

# 5.4 Частоты категорий
tvd = pd.Series({c: 0.5 * np.abs(df[c].value_counts(normalize=True).reindex(CATS[c]).fillna(0)
                                 - syn[c].value_counts(normalize=True).reindex(CATS[c]).fillna(0)).sum() for c in CAT})
print("\nРасхождение частот категорий (TVD, 0 - идентично):\n", tvd.round(3).to_string())
tvd.sort_values().plot(kind='barh', figsize=(7, 6), edgecolor='k'); plt.xlabel('TVD')
plt.title('Расхождение частот категорий'); plt.tight_layout(); show("fig5_cat_tvd")


# 5.5 Корреляции
def to_matrix(d):
    m = d.copy(); m['SeniorCitizen'] = m['SeniorCitizen'].astype(int)
    for c in CAT:
        if c not in ('SeniorCitizen', 'Churn'):
            m[c] = pd.Categorical(m[c], categories=CATS[c])
    m['Churn'] = (m['Churn'] == 'Yes').astype(int)
    return pd.get_dummies(m, drop_first=True, dtype=int)


# Корреляции между признаками (настоящие vs GMM)
C_real, C_syn = to_matrix(df).corr(), to_matrix(syn).corr().reindex_like(to_matrix(df).corr()).fillna(0)
iu = np.triu_indices(len(C_real), 1)
print(f"\nСреднее |Δcorr| по всем парам: {np.abs(C_real.values - C_syn.values)[iu].mean():.3f}")
cr = C_real['Churn'].drop('Churn'); cs = C_syn['Churn'].drop('Churn')
print("Корреляция с оттоком: настоящие vs GMM")
print(pd.DataFrame({'Настоящие': cr, 'GMM': cs}).reindex(cr.abs().sort_values(ascending=False).index).head(8).round(2).to_string())
fig, ax = plt.subplots(1, 3, figsize=(20, 6.5))
sns.heatmap(C_real, cmap='coolwarm', center=0, ax=ax[0], xticklabels=False, yticklabels=False); ax[0].set_title('Настоящие')
sns.heatmap(C_syn, cmap='coolwarm', center=0, ax=ax[1], xticklabels=False, yticklabels=False); ax[1].set_title('GMM')
sns.heatmap((C_real - C_syn).abs(), cmap='Reds', vmin=0, vmax=.5, ax=ax[2], xticklabels=False, yticklabels=False)
ax[2].set_title('|Δcorr|')
plt.tight_layout(); show("fig6_corr")

# 5.6 Понижение размерности: PCA и t-SNE
Xr = to_matrix(df).drop(columns='Churn'); Xs = to_matrix(syn).drop(columns='Churn')
scx = StandardScaler().fit(Xr)
Zr, Zs = scx.transform(Xr), scx.transform(Xs)
pca = PCA(n_components=2, random_state=SEED).fit(Zr)
Pr, Ps = pca.transform(Zr), pca.transform(Zs)
print(f"\nPCA: две компоненты объясняют {pca.explained_variance_ratio_.sum() * 100:.1f}% дисперсии настоящих данных")
n80 = lambda Z: int(np.searchsorted(np.cumsum(PCA().fit(Z).explained_variance_ratio_), 0.8) + 1)
print("Компонент для 80% дисперсии: настоящие", n80(Zr), "| GMM", n80(StandardScaler().fit_transform(Xs)))
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
ax[0].scatter(Pr[:, 0], Pr[:, 1], s=6, c='lightgray', label='Настоящие')
ax[0].scatter(Ps[:, 0], Ps[:, 1], s=6, c='tab:red', alpha=.4, label='GMM')
ax[0].set_title('PCA'); ax[0].legend(markerscale=3)
rng = np.random.default_rng(SEED)
ir, i_s = rng.choice(len(Zr), 1000, replace=False), rng.choice(len(Zs), 1000, replace=False)
T = TSNE(n_components=2, perplexity=30, init='pca', random_state=SEED).fit_transform(np.vstack([Zr[ir], Zs[i_s]]))
ax[1].scatter(T[:1000, 0], T[:1000, 1], s=8, c='lightgray', label='Настоящие')
ax[1].scatter(T[1000:, 0], T[1000:, 1], s=8, c='tab:red', alpha=.5, label='GMM')
ax[1].set_title('t-SNE (по 1000 точек)'); ax[1].legend(markerscale=3)
plt.tight_layout(); show("fig7_pca_tsne")

print("\nГотово. Таблицы сохранены в pr3_*.csv")