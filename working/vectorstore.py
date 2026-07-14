"""vectorstore.py — il corpus Lumen dentro un vector DB (ChromaDB).

Prima il quick start col docente (la collection su disco, i primi doc, una
query); poi scrivete voi il caricamento completo e le due ricerche, e le
provate nel main confrontandole col vostro brute-force di giovedì:

    python working/vectorstore.py
"""

import json
import sys

from pathlib import Path

import chromadb
import numpy as np

SCRIPTS = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPTS / "scaffolding"))

from aikit.embeddings import embed  # noqa: E402 — il toolkit di L19
from aikit import search as ricerca_l20  # noqa: E402 — il brute-force di L20

DATASET = SCRIPTS / "dataset"
CHROMA_DIR = SCRIPTS / "chroma"  # qui il DB vive su disco


# --------------------------------------------------- TODO 1 · col docente 🎤
def apri_collection():
    """La collection 'lumen' su disco: la crea la prima volta, poi la riapre.

    Client persistente su CHROMA_DIR; la collection usa la metrica cosine
    (la stessa di L19-L20).
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    return client.get_or_create_collection("lumen", metadata={"hnsw:space": "cosine"})


# ----------------------------------------------------------- TODO 2 · voi ⌨️
def carica_corpus_nel_db(collection):
    """Upsert dei 48 doc di L19: id, testo, embedding dall'npz, metadata tipo.

    Gli embedding sono quelli GIÀ CALCOLATI di corpus_embeddings.npz, passati
    espliciti (mai l'embedder di default del DB); l'ordine giusto è in
    corpus_meta.json → "ids". Restituisce il backend del corpus (dal meta),
    che serve per embeddare le query nello stesso spazio.
    """
    # 1. Carichiamo i vettori e i metadati generali
    vectors = np.load(DATASET / "corpus_embeddings.npz")["vectors"]
    meta = json.loads((DATASET / "corpus_meta.json").read_text(encoding="utf-8"))

    # 2. Leggiamo il file JSONL riga per riga per mappare ogni ID al suo testo e tipo
    per_id = {}
    for riga in (
        (DATASET / "corpus_lumen.jsonl").read_text(encoding="utf-8").splitlines()
    ):
        if riga.strip():
            doc = json.loads(riga)
            per_id[doc["id"]] = doc

    # 3. Prepara i dati per l'upsert
    # Recuperiamo la lista ordinata di tutti gli ID del corpus (saranno i nostri 48 id)
    tutti_gli_id = meta["ids"]

    # 4. Costruiamo le liste parallele seguendo rigorosamente l'ordine di tutti_gli_id
    documenti = [per_id[i]["testo"] for i in tutti_gli_id]
    metadati = [{"tipo": per_id[i]["tipo"]} for i in tutti_gli_id]

    # Poiché l'NPZ segue già l'ordine originario del corpus, prendiamo tutti i vettori
    vettori = vectors
    # 5. Inviamo tutto al Vector DB
    collection.upsert(
        ids=tutti_gli_id,
        metadatas=metadati,
        documents=documenti,
        embeddings=vettori,
    )

    # 6. Restituiamo il backend
    return meta["backend"]


# ----------------------------------------------------------- TODO 3 · voi ⌨️
def search(query, k):
    """I k doc più simili alla query, chiesti al DB. Risultati come a L20.

    Ogni risultato è un dict {"id", "tipo", "testo", "score"}: la query va
    embeddata col backend del corpus. Collection e backend qui dentro non
    arrivano dai parametri: potete riaprirli/rileggerli DENTRO la funzione
    (apri_collection() e corpus_meta.json — è quello che fa anche il vostro
    search.py di L20 col corpus). Il DB risponde in DISTANZE (cosine): lo
    score di L20 è 1 - distanza.
    """
    # 1. Riapriamo la connessione al DB
    collection = apri_collection()

    # 2. Leggiamo i metadati per scoprire quale backend usare
    meta = json.loads((DATASET / "corpus_meta.json").read_text(encoding="utf-8"))

    # 3. Trasformiamo la stringa della query in un vettore numerico
    query_vettore = embed([query], backend=meta["backend"])[0]

    # 4. Interroghiamo il DB per ottenere i k risultati più vicini
    risultati_db = collection.query(query_embeddings=[query_vettore], n_results=k)

    # 5. Prepariamo la lista dei risultati nel formato richiesto estraendo i dati dai metadati e dai documenti
    ids = risultati_db["ids"][0]
    documents = risultati_db["documents"][0]
    metadatas = risultati_db["metadatas"][0]
    distances = risultati_db["distances"][0]

    # 6. Ricostruiamo la lista di dizionari nel formato richiesto
    risultati_finali = []
    for i in range(len(ids)):
        res = {
            "id": ids[i],
            "tipo": metadatas[i]["tipo"],
            "testo": documents[i],
            "score": 1 - distances[i],  # Trasformiamo la distanza in score
        }
        risultati_finali.append(res)

    return risultati_finali


# ----------------------------------------------------------- TODO 4 · voi ⌨️
def search_filtrata(query, k, tipo):
    """Come search(), ma solo tra i doc di quel tipo (where sui metadata)."""

    # 1. Recuperiamo il contesto (connessione e metadati)
    collection = apri_collection()
    meta = json.loads((DATASET / "corpus_meta.json").read_text(encoding="utf-8"))

    # 2. Generiamo l'embedding della query
    query_vettore = embed([query], backend=meta["backend"])[0]

    # 3. Interroghiamo il DB con il filtro sui metadati
    risultati_db = collection.query(
        query_embeddings=[query_vettore],
        n_results=k,
        where={"tipo": tipo},  # Filtro sui metadati
    )

    # 4. Prepariamo la lista dei risultati nel formato richiesto
    ids = risultati_db["ids"][0]
    documents = risultati_db["documents"][0]
    metadatas = risultati_db["metadatas"][0]
    distances = risultati_db["distances"][0]

    # 5. Ricostruiamo la lista di dizionari nel formato richiesto
    risultati_finali = []
    for i in range(len(ids)):
        res = {
            "id": ids[i],
            "tipo": metadatas[i]["tipo"],
            "testo": documents[i],
            "score": 1 - distances[i],  # Trasformiamo la distanza in score
        }
        risultati_finali.append(res)

    return risultati_finali


if __name__ == "__main__":
    QUERY_FACILE = "Quanto costa la spedizione?"
    QUERY_GENERICA = "informazioni sulla scrivania"
    QUERY_DOPPIONI = "problemi con il motore della scrivania"
    K = 3

    # Quick start (🎤 col docente): apri_collection(), upsert dei primi 3 doc
    # del corpus con gli embedding dell'npz, una query di prova. Poi rilanciate
    # lo script: i documenti sono ancora lì.
    collection = apri_collection()
    vectors = np.load(DATASET / "corpus_embeddings.npz")["vectors"]
    meta = json.loads((DATASET / "corpus_meta.json").read_text(encoding="utf-8"))
    per_id = {}
    for riga in (
        (DATASET / "corpus_lumen.jsonl").read_text(encoding="utf-8").splitlines()
    ):
        if riga.strip():
            doc = json.loads(riga)
            per_id[doc["id"]] = doc

    primi = meta["ids"][:5]
    collection.upsert(
        ids=primi,
        metadatas=[{"tipo": per_id[i]["tipo"]} for i in primi],
        documents=[per_id[i]["testo"] for i in primi],
        embeddings=vectors[:5],
    )
    print(f"quickstart: {collection.count()} righe.")

    q = embed(["Quanto dura la garanzia?"], backend=meta["backend"])[0]
    prova = collection.query(query_embeddings=[q], n_results=2)
    for i in range(len(prova["ids"][0])):
        print(f"    {1-prova['distances'][0][i]}     {prova["ids"][0]} ")
        print(f"    {prova['documents'][0][i]} ")

    # =====================================================================
    # VERIFICA LAB 1 & LAB 2
    # =====================================================================
    print("\n" + "=" * 40)
    print(" VERIFICA FUNZIONI LAB 1 & LAB 2")
    print("=" * 40)

    # 1. Carichiamo tutto il corpus nel DB (TODO 2)
    print("\n[1/3] Caricamento del corpus completo...")
    carica_corpus_nel_db(collection)
    print(
        f"-> Fatto! Documenti totali nel DB: {collection.count()} (Dovrebbero essere 48)"
    )

    # 2. Test della tua funzione search() (TODO 3)
    print(f"\n[2/3] Test search() su QUERY_FACILE: '{QUERY_FACILE}'")
    risultati_standard = search(QUERY_FACILE, k=K)
    for r in risultati_standard:
        print(f"  - [{r['id']}] Score: {r['score']:.4f} | Tipo: {r['tipo']}")
        print(f"    Testo: {r['testo'][:80]}...")

    # 3. Test della tua funzione search_filtrata() (TODO 4)
    # Proviamo a forzare la ricerca solo sui ticket ("tic")
    FILTRO_TIPO = "tic"
    print(
        f"\n[3/3] Test search_filtrata() [Solo tipo: '{FILTRO_TIPO}'] su QUERY_FACILE"
    )
    risultati_filtrati = search_filtrata(QUERY_FACILE, k=K, tipo=FILTRO_TIPO)
    for r in risultati_filtrati:
        print(f"  - [{r['id']}] Score: {r['score']:.4f} | Tipo: {r['tipo']}")
        print(f"    Testo: {r['testo'][:80]}...")

    # =====================================================================
    # TODO 5: PROVE CRUD (Cancellazione e Aggiornamento Coordinato)
    # =====================================================================
    print("\n" + "=" * 40)
    print(" TODO 5: PROVE CRUD SUL VECTOR DB")
    print("=" * 40)

    # --- 1. OPERAZIONE DELETE (Cancellazione di un doc in top-3) ---
    ID_DA_CANCELLARE = "faq-01"
    print(f"\n[1/2] Eliminazione del documento: {ID_DA_CANCELLARE}...")

    # Rimuoviamo il documento dall'indice vettoriale
    collection.delete(ids=[ID_DA_CANCELLARE])

    print("-> Eseguo nuovamente search() per verificare che sia sparito:")
    risultati_post_delete = search("Quanto dura la garanzia?", k=2)
    for r in risultati_post_delete:
        print(f"  - [{r['id']}] Score: {r['score']:.4f} | Testo: {r['testo'][:75]}...")

    # --- 2. OPERAZIONE UPDATE (Aggiornamento coordinato di Testo + Vettore) ---
    ID_DA_AGGIORNARE = "faq-02"
    NUOVO_TESTO_FAQ = "È possibile effettuare il ritiro in negozio? Sì, seleziona l'opzione al checkout per ritirare gratuitamente in sede."

    print(f"\n[2/2] Aggiornamento (ri-embed + upsert) per l'ID: {ID_DA_AGGIORNARE}...")

    # Ricalcoliamo il vettore sul nuovo testo (evitiamo il bug del disallineamento)
    nuovo_vettore_faq = embed([NUOVO_TESTO_FAQ], backend=meta["backend"])[0]

    # Aggiorniamo il record mantenendo lo stesso ID
    collection.upsert(
        ids=[ID_DA_AGGIORNARE],
        metadatas=[{"tipo": "faq"}],
        documents=[NUOVO_TESTO_FAQ],
        embeddings=[nuovo_vettore_faq],
    )

    print(
        "-> Test di verifica dell'aggiornamento sulla query 'posso ritirare in negozio?':"
    )
    risultati_post_update = search("posso ritirare in negozio?", k=1)
    for r in risultati_post_update:
        print(f"  - [{r['id']}] Score: {r['score']:.4f} | Testo: {r['testo'][:75]}...")

    # Lab 1 (⌨️ da soli): il corpus completo nel DB, poi le tre query qui
    # sopra con la vostra search() fianco a fianco col brute-force di
    # giovedì (ricerca_l20.search): stessi documenti in testa?

    # Lab 2 (⌨️ da soli): le stesse query con search_filtrata() variando il
    # tipo, poi le prove CRUD: delete di un doc in top-3, update di un testo
    # (ri-embed + upsert con lo stesso id). Da qui in giù scrivete voi.

    # Esercizio extra (SOLO se il docente lo lancia): il 49° documento —
    # scrivete VOI una voce nuova del corpus (una FAQ che manca, un ticket
    # realistico), embed + upsert con un id nuovo e il tipo giusto; poi
    # dimostrate che la ricerca la trova e che il filtro la rispetta.

    # =====================================================================
    # ESERCIZIO EXTRA: IL 49° DOCUMENTO
    # =====================================================================
    print("\n" + "=" * 40)
    print(" ESERCIZIO EXTRA: INSERIMENTO 49° DOCUMENTO")
    print("=" * 40)

    # 1. Definiamo i dati del nuovo documento inventato
    NUOVO_ID = "faq-49"
    NUOVO_TESTO = "Per richiedere il rimborso della scrivania LumaDesk entro i 30 giorni, invia una mail a rimborsi@lumadesk.com con la ricevuta d'acquisto."
    NUOVO_TIPO = "faq"

    print(f"\n[1/3] Generazione embedding e inserimento di {NUOVO_ID}...")

    # Calcoliamo l'embedding al volo usando il backend corretto del corpus
    nuovo_vettore = embed([NUOVO_TESTO], backend=meta["backend"])[0]

    # Eseguiamo l'upsert del singolo documento
    collection.upsert(
        ids=[NUOVO_ID],
        metadatas=[{"tipo": NUOVO_TIPO}],
        documents=[NUOVO_TESTO],
        embeddings=[nuovo_vettore],
    )
    print(f"-> Fatto! Nuovo conteggio documenti nel DB: {collection.count()}")

    # 2. Dimostriamo che la ricerca generica lo trova
    QUERY_EXTRA = "Come posso chiedere il rimborso?"
    print(f"\n[2/3] Test search() su query specifica: '{QUERY_EXTRA}'")
    risultati_extra = search(QUERY_EXTRA, k=2)
    for r in risultati_extra:
        print(f"  - [{r['id']}] Score: {r['score']:.4f} | Tipo: {r['tipo']}")
        print(f"    Testo: {r['testo'][:80]}...")

    # 3. Dimostriamo che il filtro la rispetta (Se cerchiamo per 'tic' non deve trovarla)
    print(f"\n[3/3] Test search_filtrata() [Solo tipo: 'tic'] sulla stessa query")
    risultati_filtro_extra = search_filtrata(QUERY_EXTRA, k=2, tipo="tic")
    for r in risultati_filtro_extra:
        print(f"  - [{r['id']}] Score: {r['score']:.4f} | Tipo: {r['tipo']}")
        print(f"    Testo: {r['testo'][:80]}...")
