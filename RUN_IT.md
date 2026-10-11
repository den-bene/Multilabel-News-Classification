# Esecuzione del progetto

## Google Colab

1. Apri `news_classification.ipynb` in Google Colab.
2. Scegli un runtime Python con GPU per velocizzare la LSTM.
3. Dal pannello File carica `input_data.pkl` nella directory di lavoro. Il file pesa circa 883 MB: il caricamento può richiedere alcuni minuti.
4. Esegui tutte le celle in ordine. Se mancano librerie, usa una cella iniziale con `%pip install -r requirements.txt` dopo aver caricato anche quel file.
5. Salva il notebook eseguito e scarica `news_results.zip` dal pannello File. Il runtime è temporaneo: scarica i risultati prima di chiuderlo.

Il notebook usa il dataset locale, senza dipendere dalla disponibilità del sito di download.

## VS Code

1. Apri questa cartella in VS Code e installa le estensioni Microsoft **Python** e **Jupyter**.
2. Usa Python 3.11 o 3.12. Dal terminale della cartella crea un ambiente:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Su Windows, attiva l'ambiente con `.venv\Scripts\Activate.ps1`.

3. Copia il dataset del progetto in `data/input_data.pkl`.
4. Apri il notebook, premi **Select Kernel** e scegli Python nell'ambiente `.venv`.
5. Premi **Run All**. L'esecuzione confronta quattro reti con al massimo dieci epoche ciascuna; su CPU può richiedere diversi minuti. Le tabelle e i grafici vengono scritti in `reports/`.

Se compare un errore sulle librerie, verifica che terminale e kernel usino entrambi `.venv`. Il file pickle va caricato soltanto dalla fonte originale fidata.

## GitHub e CV

Repository pubblico: https://github.com/den-bene/multilabel-news-classification .

La repo contiene notebook con output, README, questa guida, `requirements.txt`, `.gitignore` e `DATASET.md`. Il dataset, i report generati, il modello binario e gli archivi restano locali. Tabelle e grafici sono già visibili nel notebook; eseguendolo vengono rigenerati i report.

Per il CV, collega il titolo **Multi-label News Classification** all'URL del repository.
