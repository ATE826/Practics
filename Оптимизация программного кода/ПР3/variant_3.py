import time
import random
from collections import Counter


def word_frequency(texts: list[str]) -> dict[str, int]:
    counter = Counter()
    for text in texts:
        for w in text.lower().split():
            cleaned = "".join(c for c in w if c.isalnum())
            if cleaned:
                counter[cleaned] += 1
    return dict(counter)


def bigram_counts(texts: list[str]) -> dict[tuple[str, str], int]:
    counter = Counter()
    for text in texts:
        words = [
            cleaned
            for cleaned in ("".join(c for c in w if c.isalnum()) for w in text.lower().split())
            if cleaned
        ]
        for i in range(len(words) - 1):
            counter[(words[i], words[i + 1])] += 1
    return dict(counter)


def generate_texts(n: int) -> list[str]:
    words_pool = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "python", "code", "optimization", "performance", "algorithm",
        "data", "structure", "analysis", "profiling", "memory", "cache",
    ]
    texts = []
    for _ in range(n):
        length = random.randint(5, 15)
        texts.append(" ".join(random.choices(words_pool, k=length)))
    return texts


def run_task(n: int):
    texts = generate_texts(n)
    freq = word_frequency(texts)
    bigrams = bigram_counts(texts)
    return len(freq), len(bigrams)


if __name__ == "__main__":
    random.seed(42)
    for n in [300, 600, 1200]:
        start = time.perf_counter()
        res = run_task(n)
        elapsed = time.perf_counter() - start
        print(f"N={n:5d}  result={res}  time={elapsed:.4f}s")