"""benchmark_ann.py — la stessa query: brute-force vs indice (ChromaDB).

Il seguito di benchmark_scala.py di L20: stessi vettori sintetici float32 a
1536 dimensioni, stessa misura (una query, il tempo migliore su più giri) —
ma stavolta la ricerca la fa anche l'indice HNSW di ChromaDB, in memoria.
Sono i numeri del "reveal" in slide: il brute-force cresce lineare, l'indice
resta piatto; in cambio l'indice si PAGA alla scrittura (tempo di
costruzione, anche lui misurato).

Uso (dalla cartella scripts/):

    python scaffolding/benchmark_ann.py

La taglia grossa (50k) impiega qualche decina di secondi a indicizzare:
è il punto, non un difetto. RAM: n × 1536 × 4 byte (50k ≈ 300 MB).
"""
import time

import chromadb
import numpy as np

DIM = 1536            # le dimensioni di text-embedding-3-small (il corpus vero)
RIPETIZIONI = 5       # si tiene il tempo MIGLIORE: misura il costo, non il rumore
TAGLIE = [5_000, 50_000]
LOTTO = 5_000         # gli insert nel DB si fanno a lotti


def brute_ms(vectors, q):
    """ms per UNA query brute-force: la misura di L20, identica."""
    tempi = []
    for _ in range(RIPETIZIONI):
        inizio = time.perf_counter()
        scores = vectors @ q / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q))
        np.argsort(scores)[::-1][:5]
        tempi.append(time.perf_counter() - inizio)
    return min(tempi) * 1000


def chroma_ms(collection, q):
    """ms per la STESSA query, chiesta all'indice."""
    tempi = []
    for _ in range(RIPETIZIONI):
        inizio = time.perf_counter()
        collection.query(query_embeddings=[q], n_results=5)
        tempi.append(time.perf_counter() - inizio)
    return min(tempi) * 1000


if __name__ == "__main__":
    print(f"una query su n vettori, {DIM} dim, float32 (migliore di {RIPETIZIONI} giri)\n")
    print(f"{'n vettori':>10} {'brute ms':>10} {'chroma ms':>10} {'indicizzazione':>15}")
    client = chromadb.EphemeralClient()               # in memoria: niente file in giro
    rng = np.random.default_rng(20)
    for n in TAGLIE:
        vectors = rng.standard_normal((n, DIM), dtype=np.float32)
        q = rng.standard_normal(DIM, dtype=np.float32)

        collection = client.create_collection(
            f"bench_{n}", metadata={"hnsw:space": "cosine"}
        )
        inizio = time.perf_counter()
        for i in range(0, n, LOTTO):
            lotto = vectors[i:i + LOTTO]
            collection.add(
                ids=[str(j) for j in range(i, i + len(lotto))],
                embeddings=lotto,
            )
        costruzione = time.perf_counter() - inizio

        print(f"{n:>10,} {brute_ms(vectors, q):>10.1f} {chroma_ms(collection, q):>10.1f} "
              f"{costruzione:>13.1f} s")

    print("\nIl brute-force cresce dritto con n (L20); l'indice resta piatto —")
    print("ma si paga UNA volta, alla scrittura. Ogni query da lì in poi è gratis.")
