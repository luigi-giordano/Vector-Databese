"""check_lab21.py — [OK]/[FAIL] sui RISULTATI, non sul come.

Da lanciare dalla cartella scripts/:

    python scaffolding/check_lab21.py               # verifica working/ (istantanea, zero costi)
    python scaffolding/check_lab21.py --live        # in più: 2 query VERE via OpenAI (~$0.000001)
    python scaffolding/check_lab21.py --solutions   # verifica solutions/ (per il docente)

Il check controlla che il DB FUNZIONI: come lo riempite e interrogate è
affar vostro. La verifica base non chiama nessun servizio: usa un DB
usa-e-getta in una cartella temporanea e, come query, il vettore GIÀ
CALCOLATO di un documento del corpus (embed viene rimpiazzata al volo) —
quel documento deve uscire primo. Il giro reale via API sta dietro --live.
"""
import json
import os
import shutil
import sys
import tempfile

from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).parent.parent
DATASET = SCRIPTS / "dataset"
TARGET = SCRIPTS / ("solutions" if "--solutions" in sys.argv else "working")

sys.path.insert(0, str(SCRIPTS / "scaffolding"))
sys.path.insert(0, str(TARGET))

FALLIMENTI = 0


def esito(ok, msg):
    global FALLIMENTI
    print(("[OK]  " if ok else "[FAIL]") + " " + msg)
    if not ok:
        FALLIMENTI += 1


def check_caricamento():
    print("\n— TODO 1-2 · la collection si apre e il corpus ci sta tutto")
    import vectorstore as v
    try:
        collection = v.apri_collection()
        backend = v.carica_corpus_nel_db(collection)
    except NotImplementedError:
        esito(False, "apri_collection / carica_corpus_nel_db ancora da fare")
        return False
    except Exception as e:
        esito(False, f"il caricamento solleva {type(e).__name__}: {e}")
        return False
    esito(collection.count() == 48 and bool(backend),
          f"48 documenti nel DB e il backend (trovati {collection.count()})")
    riaperta = v.apri_collection()
    esito(riaperta.count() == 48,
          "riaprendo la collection i documenti sono ancora lì (persistenza)")
    return True


def check_search():
    print("\n— TODO 3 · search trova quello che deve (query = vettore già noto, zero rete)")
    import vectorstore as v
    vectors = np.load(DATASET / "corpus_embeddings.npz")["vectors"]
    meta = json.loads((DATASET / "corpus_meta.json").read_text(encoding="utf-8"))
    doc_id = meta["ids"][0]                                # faq-01
    v.embed = lambda texts, backend="finto": np.array([vectors[0]])
    try:
        ris = v.search("una query qualunque", 3)
    except NotImplementedError:
        esito(False, "search ancora da fare")
        return
    except Exception as e:
        esito(False, f"search solleva {type(e).__name__}: {e}")
        return
    esito(bool(ris) and ris[0].get("id") == doc_id
          and abs(ris[0].get("score", 0) - 1.0) < 5e-3,
          f"query = il vettore di {doc_id} → quel doc esce primo, score ≈ 1")
    scores = [r.get("score") for r in ris]
    esito(len(ris) == 3 and scores == sorted(scores, reverse=True),
          "k rispettato e risultati dal più simile in giù (score, non distanze)")


def check_search_filtrata():
    print("\n— TODO 4 · search_filtrata resta nel recinto dei metadata")
    import vectorstore as v
    vectors = np.load(DATASET / "corpus_embeddings.npz")["vectors"]
    v.embed = lambda texts, backend="finto": np.array([vectors[0]])   # ancora faq-01
    try:
        ris = v.search_filtrata("una query qualunque", 3, "ticket")
    except NotImplementedError:
        esito(False, "search_filtrata ancora da fare")
        return
    except Exception as e:
        esito(False, f"search_filtrata solleva {type(e).__name__}: {e}")
        return
    esito(len(ris) == 3 and all(r.get("tipo") == "ticket" for r in ris),
          "filtro tipo='ticket' → escono solo ticket (la faq più simile resta fuori)")


def check_live():
    print("\n— LIVE · 2 query vere via OpenAI (~$0.000001)")
    if not os.getenv("OPENAI_API_KEY", "").strip():
        esito(False, "serve OPENAI_API_KEY nel .env per il check --live")
        return
    import vectorstore as v
    try:
        buona = v.search("Quanto dura la garanzia?", 3)
        fuori = v.search("Qual è la ricetta della carbonara?", 3)
    except NotImplementedError:
        esito(False, "search ancora da fare")
        return
    esito(any(r["id"] == "faq-01" for r in buona),
          "la FAQ sulla garanzia esce nei top-3 della sua query")
    esito(bool(buona) and bool(fuori) and fuori[0]["score"] < buona[0]["score"],
          f"fuori dominio ({fuori[0]['score']:.3f}) sotto la query sensata "
          f"({buona[0]['score']:.3f})")


if __name__ == "__main__":
    print(f"verifico: {TARGET.name}/")

    # DB usa-e-getta: il check non tocca la collection vera in scripts/chroma/
    import vectorstore as _v
    _tmp = Path(tempfile.mkdtemp(prefix="chroma_check_"))
    _v.CHROMA_DIR = _tmp
    _embed_vero = getattr(_v, "embed", None)

    try:
        if check_caricamento():
            check_search()
            check_search_filtrata()
            if "--live" in sys.argv:
                _v.embed = _embed_vero
                check_live()
    finally:
        shutil.rmtree(_tmp, ignore_errors=True)

    print("\n" + ("Tutto a posto ✔" if FALLIMENTI == 0
                  else f"{FALLIMENTI} requisito/i ancora da sistemare."))
    sys.exit(0 if FALLIMENTI == 0 else 1)
