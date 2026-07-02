# analyze_lengths.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from testoBPE import BPE

# Инициализация токенизатора
tok = BPE()

# Конфигурация датасетов: путь, колонка_вход, колонка_выход, тип_задачи
DATASETS = {
    "gazeta": {
        "path": "/home/debservak/Рабочий стол/buffer/stModel/llm_datasets/gazeta/default/train",
        "col_in": "summary",      # anchor: короткая выжимка
        "col_out": "text",        # positive: полная статья
        "task": "Суммаризация (выжимка → статья)"
    },
    "ru-WANLI": {
        "path": "/home/debservak/Рабочий стол/buffer/stModel/llm_datasets/ru-WANLI/data/train.parquet",
        "col_in": "hypothesis",   # anchor: гипотеза
        "col_out": "premise",     # positive: посылка (для entailment)
        "task": "NLI (гипотеза → посылка)",
        "filter": lambda df: df[df["label"] == "entailment"]  # только entailment как +
    },
    "samsum-ru": {
        "path": "/home/debservak/Рабочий стол/buffer/stModel/llm_datasets/samsum-ru/data/train-00000-of-00001-76cc3fe8132d8f4b.parquet",
        "col_in": "summary",
        "col_out": "dialogue",
        "task": "Диалог-суммаризация"
    },
    "xlsum-russian-bbc": {
        "path": "/home/debservak/Рабочий стол/buffer/stModel/llm_datasets/xlsum-russian-bbc/bbcrussian.csv.gz",
        "col_in": "resume",
        "col_out": "news",
        "task": "BBC-суммаризация",
        "read_csv": True
    },
    "RuSentEval": {
        "path": "/home/debservak/Рабочий стол/buffer/stModel/llm_datasets/RuSentEval/data",
        "col_in": "target_word",  # целевое слово (якорь)
        "col_out": "sentence",    # контекст, который притягиваем
        "task": "Ключевое слово → контекст",
        "custom_load": True
    }
}

def load_dataset(name, cfg):
    """Загружает датасет в DataFrame с учётом специфики формата"""
    if cfg.get("custom_load") and name == "RuSentEval":
        # RuSentEval: читаем все .txt, парсим "метка \t слово \t предложение"
        rows = []
        for fname in os.listdir(cfg["path"]):
            if fname.endswith(".txt"):
                with open(os.path.join(cfg["path"], fname), "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) >= 3:
                            rows.append({cfg["col_in"]: parts[1], cfg["col_out"]: parts[2]})
        return pd.DataFrame(rows)
    
    elif cfg.get("read_csv"):
        return pd.read_csv(cfg["path"], compression="gzip", nrows=5000)  # сэмпл для скорости
    
    elif os.path.isdir(cfg["path"]):
        # parquet split: читаем первый файл для анализа
        files = [f for f in os.listdir(cfg["path"]) if f.endswith(".parquet")]
        df = pd.read_parquet(os.path.join(cfg["path"], files[0]))
        if "filter" in cfg:
            df = cfg["filter"](df)
        return df
    else:
        df = pd.read_parquet(cfg["path"])
        if "filter" in cfg:
            df = cfg["filter"](df)
        return df

def get_lengths(df, col_in, col_out, max_samples=20000):
    """Считает длины в токенах для двух колонок, сэмплируя если нужно"""
    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=42)
    
    lens_in, lens_out = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing", leave=False):
        text_in = str(row.get(col_in, ""))
        text_out = str(row.get(col_out, ""))
        if text_in and text_out:
            lens_in.append(len(tok.encode(text_in)))
            lens_out.append(len(tok.encode(text_out)))
    return np.array(lens_in), np.array(lens_out)

def plot_distribution(lengths, ax, title, is_bar=None):
    """Рисует распределение: столбцы если <10 уникальных значений, иначе линия"""
    if is_bar is None:
        is_bar = len(np.unique(lengths)) < 10
    
    # Кап на 99-м квантиле
    cap = np.percentile(lengths, 99)
    lengths_clipped = np.clip(lengths, 0, cap)
    
    if is_bar:
        # Столбчатая диаграмма для дискретных значений
        counts = pd.Series(lengths_clipped.astype(int)).value_counts().sort_index()
        ax.bar(counts.index, counts.values, width=0.8, alpha=0.7)
    else:
        # Линейный график для непрерывных
        hist, bins = np.histogram(lengths_clipped, bins=50)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax.plot(bin_centers, hist, linewidth=1.5)
        ax.fill_between(bin_centers, hist, alpha=0.3)
    
    ax.set_xlabel("Длина в токенах (99% квантиль)")
    ax.set_ylabel("Частота")
    ax.set_title(title)
    ax.grid(alpha=0.3)

# Основной цикл анализа
results = {}
for name, cfg in DATASETS.items():
    print(f"\n🔍 Анализ {name}: {cfg['task']}")
    df = load_dataset(name, cfg)
    print(f"   Загружено строк: {len(df)}")
    
    if len(df) == 0:
        print(f"   ⚠️ Пустой датасет, пропускаем")
        continue
    
    lens_in, lens_out = get_lengths(df, cfg["col_in"], cfg["col_out"])
    results[name] = {"lens_in": lens_in, "lens_out": lens_out, "task": cfg["task"]}
    print(f"   📊 Диапазон длин: вход [{lens_in.min()}, {np.percentile(lens_in, 99):.0f}], "
          f"выход [{lens_out.min()}, {np.percentile(lens_out, 99):.0f}]")

# Построение графика
n_datasets = len(results)
fig, axes = plt.subplots(n_datasets, 2, figsize=(14, 4 * n_datasets), squeeze=False)
fig.suptitle("Распределение длин в токенах (testoBPE) — вход и выход по датасетам", fontsize=16, fontweight="bold")

for idx, (name, data) in enumerate(results.items()):
    # Левый подграфик: вход (anchor)
    plot_distribution(data["lens_in"], axes[idx, 0], 
                     f"{name}\nВход: {data['task'].split('→')[0].strip()}",
                     is_bar=len(np.unique(data["lens_in"])) < 10)
    
    # Правый подграфик: выход (positive)
    plot_distribution(data["lens_out"], axes[idx, 1],
                     f"{name}\nВыход: {data['task'].split('→')[-1].strip()}",
                     is_bar=len(np.unique(data["lens_out"])) < 10)

plt.tight_layout()
plt.savefig("graph.png", dpi=150, bbox_inches="tight")
print(f"\n✅ График сохранён: graph.png ({n_datasets} датасетов × 2 подграфика)")