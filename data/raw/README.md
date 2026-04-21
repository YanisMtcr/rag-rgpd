# Source PDFs

This folder holds the PDFs that are parsed into the ChromaDB store by `notebooks/01_ingestion.ipynb`. All the documents come from public official sources.

The ingestion parser in `src/ingestion.py` dispatches on the filename prefix:

- `rgpd_*.pdf` is split article by article (regex on `Article N`)
- `cnil_fiche_*.pdf` is chunked with a standard recursive splitter, tagged `source_type=fiche_cnil`
- `cnil_sanction_<entity>_<year>.pdf` is split on the roman numeral sections (I., II., III.), tagged `source_type=sanction`, and the entity/year are extracted from the filename

Filenames must respect those prefixes, otherwise the file is ignored.

## Files used in this project

GDPR text:
- `rgpd_texte.pdf` - consolidated GDPR, Official Journal of the EU

CNIL practical guides (from cnil.fr):
- `cnil_fiche_aipd.pdf`
- `cnil_fiche_consentement.pdf`
- `cnil_fiche_cookies.pdf`
- `cnil_fiche_dpo.pdf`
- `cnil_fiche_droits.pdf`
- `cnil_fiche_dureeconservation.pdf`
- `cnil_fiche_fuite.pdf`
- `cnil_fiche_rh.pdf`
- `cnil_fiche_securite.pdf`
- `cnil_fiche_transferts.pdf`

CNIL sanction rulings:
- `cnil_sanction_google_2019.pdf`
- `cnil_sanction_google_2025.pdf`
- `cnil_sanction_anon_2020.pdf`
- `cnil_sanction_francetravail_2026.pdf`
