# L21 — Vector Databases · guida agli script

Lab del sabato (⌨️ + 🎤 + 📊, 4h): la ricerca di L20 **trasloca in un vector
DB**. Col docente si apre la collection ChromaDB su disco e si fa il quick
start (`apri_collection`, i primi 3 doc, una query); poi gli studenti
caricano DA SOLI il corpus completo (`carica_corpus_nel_db`), riscrivono la
ricerca sopra il DB (`search`) e la confrontano fianco a fianco col proprio
brute-force di giovedì; nel Lab 2 arrivano `search_filtrata` (metadata) e le
prove CRUD (delete + update fatto bene). Chiude la demo pgvector del docente
(materiale fuori da `scripts/`) e l'ultimo blocco flessibile: criteri di
scelta, poi Q&A o l'esercizio extra "il 49° documento" (a scelta del
docente).
Niente notebook, solo script `.py`.

Il `check_lab21.py` è **strumento del docente**, fuori dal percorso studente
(il "fatto quando" degli esercizi è l'output osservabile): serve per la
preparazione/regressione dei materiali e per la diagnosi rapida di uno
studente bloccato. Verifica i risultati su un DB usa-e-getta con query prese
dai vettori GIÀ calcolati del corpus (istantaneo, zero costi, zero rete);
il giro reale via API sta dietro `--live`.

## Ruolo di ogni file

Legenda: 🧰 ambiente/toolkit riusabile · 🎤 codice di riferimento che il docente
live-coda · ⌨️ da completare dagli studenti · 📄 documenti/dati di esempio.

| File | Ruolo | Cosa fa |
|---|---|---|
| `requirements.txt` | 🧰 | Dipendenze (`chromadb`, `openai`, `numpy`; `sentence-transformers` è **commentato** — si scommenta solo se il corpus di L19 è col backend `local`). |
| `.env.example` | 🧰 | Template per `OPENAI_API_KEY` — la **stessa chiave di L13-L20**, nessuna registrazione nuova. |
| `CONSEGNA.md` | 🧰 | Promemoria operativo della giornata per gli studenti. |
| `scaffolding/aikit/embeddings.py` | 🧰 | Il modulo di L19: `embed()` con backend selezionabile e `cosine_similarity()`. **Non si tocca**: si importa. |
| `scaffolding/aikit/search.py` | 🧰 | Il brute-force **scritto dagli studenti a L20**, promosso nel toolkit: stamattina è il termine di confronto per la ricerca sul DB. **Non si tocca**: si importa. |
| `dataset/corpus_lumen.jsonl` | 📄 | I 48 testi brevi di Lumen S.r.l. (12 FAQ, 12 prodotti, 12 ticket, 12 recensioni): il campo `tipo` diventa il metadata su cui si filtra. |
| `dataset/corpus_embeddings.npz` + `corpus_meta.json` | 📄 | Il corpus vettorizzato: **copia di riserva** identica a quella di L19-L20 — chi ha i propri file li copia qui sopra. |
| `working/vectorstore.py` | 🎤⌨️ | L'unico file della giornata. **TODO 1 col docente** (`apri_collection` + quick start nel main); **TODO 2-4 da soli** (`carica_corpus_nel_db`, `search`, `search_filtrata` — prescritti solo nomi, variabili del main e scopo); **TODO 5 nel main** (prove CRUD: delete + update); **TODO 6 extra** (il 49° documento, solo se il docente lo lancia). Riferimento in `solutions/`. |
| `scaffolding/check_lab21.py` | 🧰 | **Strumento del docente** (preparazione, regressione, diagnosi): `[OK]/[FAIL]` sui risultati, su un DB usa-e-getta — istantaneo, zero costi, zero rete; `--live` fa 2 query vere (~$0.000001); `--solutions` verifica le soluzioni. Non compare nel percorso studente. |
| `scaffolding/benchmark_ann.py` | 🧰 | Brute-force vs indice ChromaDB sugli stessi vettori sintetici di L20: i numeri del "reveal" citati in slide. |
| `solutions/` | 🎤 | Le versioni **complete dei soli file di `working/`** (stesso nome, stessi import). |

La cartella `chroma/` (il DB su disco) nasce al primo lancio e NON si
versiona (`.gitignore`); la demo pgvector del docente vive **fuori da
`scripts/`**: `demo-pgvector-docente.md` + `demo-pgvector-carica.py` nella
cartella della lezione.

## Setup

```bash
cd scripts
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # dentro: la stessa chiave OpenAI di L13-L20
```

> **Nota per chi distribuisce (docente):** agli studenti la cartella arriva
> **senza** `.env`, `.venv` e `chroma/` (git non li versiona). Chi ha il
> proprio corpus di L19 lo copia in `dataset/` sopra la riserva; per tutti
> gli altri la riserva funziona identica.

## Come si prova

Tutto si lancia dalla cartella `scripts/`:

```bash
python working/vectorstore.py                 # le prove della giornata (query via API, ~$0.000005 a giro completo)
python scaffolding/check_lab21.py             # verifica working/: istantanea, zero costi
python scaffolding/check_lab21.py --live      # 2 query vere via OpenAI (~$0.000001)
python scaffolding/check_lab21.py --solutions # per il docente: verifica solutions/
python scaffolding/benchmark_ann.py           # brute-force vs indice, misurato (~40 s)
```

## 🚀 Sviluppo Lab: Vector Database con ChromaDB

Questo modulo implementa l'indicizzazione e la ricerca semantica (vettoriale) del corpus di documenti **Lumen**, integrando un database vettoriale persistente (`ChromaDB`) con logiche di filtraggio avanzate e confrontando i risultati con l'approccio brute-force.

---

### 📋 Architettura e Funzionalità

L'applicazione è suddivisa in quattro blocchi logici principali:

1. **TODO 1: Inizializzazione della Collection (`apri_collection`)**
   * Configurazione e apertura di un client persistente ChromaDB su disco.
   * Creazione/Riuso della collection denominata `"lumen"`.
   * Configurazione della metrica di similarità impostata su **Distanza di Coseno** (`cosine`).

2. **TODO 2: Caricamento Massivo del Dataset (`carica_corpus_nel_db`)**
   * *Bulk-load* dei 48 documenti originali estratti dal file `.jsonl`.
   * Associazione atomica di ogni documento al rispettivo vettore numerico pre-calcolato (file `.npz`).
   * Inserimento strutturato dei metadati associati (`tipo` di documento: *FAQ* o *Ticket*).
   * Gestione nativa dei duplicati tramite operazioni di `upsert`.

3. **TODO 3: Ricerca Vettoriale Semantica (`search`)**
   * Conversione *on-the-fly* della query testuale in un embedding vettoriale tramite modello `OpenAI`.
   * Interrogazione della collection per recuperare i $k$ documenti semanticamente più vicini.
   * Normalizzazione dei risultati: conversione delle distanze grezze del DB in score di similitudine espliciti:
     $$\text{score} = 1 - \text{distanza}$$

4. **TODO 4: Ricerca Semantica Filtrata (`search_filtrata`)**
   * Stessa logica di recupero semantico della ricerca standard.
   * Applicazione di un filtro di segmentazione deterministico a livello database tramite parametro `where={"tipo": tipo}`.

---

### 🧪 Esercizio Extra: Indicizzazione Dinamica

Il sistema supporta l'inserimento e l'indicizzazione in tempo reale di nuove informazioni (es. un ipotetico **49° documento**) senza la necessità di rigenerare l'intero dataset statico:
* Generazione dell'embedding della nuova stringa a runtime.
* Aggiornamento incrementale dell'indice vettoriale persistente tramite `.upsert()`.
* Validazione immediata della ricercabilità e del rispetto dei vincoli di filtraggio sul metadato.

---

### 💻 Modalità di Esecuzione

Assicurati che l'ambiente virtuale (`.venv`) sia attivo e di trovarti nel percorso corretto prima di lanciare lo script:

```bash
# Spostarsi nella cartella di lavoro
cd scripts/working

# Eseguire lo script e visualizzare i test a terminale
python vectorstore.py
