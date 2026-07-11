# L21 — Vector Databases · consegna

Stamattina la ricerca di giovedì trasloca: il corpus Lumen entra in un
**vector DB** (ChromaDB) che vive su disco, e la vostra `search()` viene
riscritta sopra il database — stessi risultati, ma indicizzati, persistenti
e filtrabili. Tutto si lancia dalla cartella `scripts/`.

## Setup (una volta sola, nei primi minuti del brief)

Ogni lezione ha la sua cartella `scripts/` col suo venv: quello di stamattina
è **nuovo** (non si riusa quello di L20). Aprite il terminale **nella cartella
della lezione** (quella scaricata da Drive, che contiene `scripts/`), poi:

```bash
cd scripts
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # dentro: la STESSA chiave OpenAI di L13-L20
```

> Chiave persa? **Alzate la mano**: il docente ha una chiave di riserva.

> `sentence-transformers` è **commentato** in `requirements.txt`: si
> scommenta solo se il vostro corpus di L19 è col backend `local`
> (controllate `corpus_meta.json` → `"backend"`; su Linux prima torch CPU,
> istruzioni nel `requirements.txt`).

## Il corpus: il VOSTRO, di L19-L20

Come a L20: se avete i vostri `corpus_embeddings.npz` + `corpus_meta.json`,
copiateli in `dataset/` sopra la riserva:

```bash
cp ../../lezione-20-semantic-search/scripts/dataset/corpus_* dataset/
```

(adattate il percorso a dove avete la cartella di L20). Se il comando dà
`No such file or directory`, **lasciate perdere e usate la riserva già in
`dataset/`**: è identica, non serve altro.

## Quick start col docente (🎤 si replica) — `working/vectorstore.py`

| TODO | Cosa | Fatto quando |
|---|---|---|
| 1 | `apri_collection()`: client persistente + collection `lumen` (metrica cosine), poi nel main l'upsert dei primi 3 doc (embedding dall'npz, espliciti) e una query di prova | rilanciate lo script e i documenti **sono ancora lì** — nessun npz da ricaricare: il DB vive in `scripts/chroma/` |

## Lab 1 (⌨️ da soli) — il corpus nel DB

Scrivete le due funzioni. **Prescritti solo i nomi, le variabili del main e
lo scopo**: il come è vostro — soluzioni diverse sono benvenute.

| TODO | Cosa | Fatto quando |
|---|---|---|
| 2 | `carica_corpus_nel_db(collection)`: upsert dei 48 doc — id, testo, embedding dall'npz, metadata `tipo` | `collection.count()` dice 48, e rilanciando lo script RESTA 48 (upsert, non insert) |
| 3 | `search(query, k)`: i k doc più simili chiesti al DB, risultati come a L20 (score, non distanze). Collection e backend non sono parametri: riapriteli/rileggeteli dentro la funzione, come fa il vostro `search.py` di L20 col corpus | sulle tre query del main, il DB e il vostro brute-force di giovedì (`ricerca_l20.search`, già importato) mettono in testa **gli stessi documenti**, fianco a fianco |

## Lab 2 (⌨️ da soli) — filtri e CRUD

| TODO | Cosa | Fatto quando |
|---|---|---|
| 4 | `search_filtrata(query, k, tipo)`: la stessa ricerca ma solo tra i doc di un tipo (`where` sui metadata) | la stessa query cambia faccia: senza filtro escono ticket e recensioni, con `tipo="faq"` escono solo FAQ |
| 5 | nel main, le prove CRUD: `delete` di un doc che sta in top-3 (di una delle query del main, a vostra scelta), poi l'**update fatto bene** — nuovo testo, ri-embed con `embed()`, upsert con lo **stesso id** | il doc cancellato sparisce dai risultati; il doc aggiornato non esce più sulla query di prima e compare su quella nuova |

Il punto del TODO 5: il DB **non si accorge** dei testi cambiati — testo ed
embedding li tenete sincronizzati VOI (è la regola backend/modello di L20,
vista dal lato scrittura).

## Esercizio extra (SOLO se il docente lo lancia) — il 49° documento

| TODO | Cosa | Fatto quando |
|---|---|---|
| 6 | il corpus ha 48 voci: scrivetene **una nuova voi** (una FAQ che manca, un ticket realistico) — embed con `embed()`, upsert con **id nuovo** e il **tipo giusto** | `count()` dice **49**; la vostra query la trova **in top-1** (provate con parole diverse dal testo); col filtro del tipo giusto esce, con quello sbagliato no |

```bash
python working/vectorstore.py      # le vostre prove (qualche query via API, ~$0.000005 a giro completo)
```

## Cosa portate a casa

`vectorstore.py` — il corpus in una collection **persistente** con la ricerca
sopra il DB — e la cartella `scripts/chroma/` che sopravvive al riavvio.
Martedì (L22) arrivano i documenti **lunghi**: si spezzano, si embeddano e
finiscono in questa stessa collection.
