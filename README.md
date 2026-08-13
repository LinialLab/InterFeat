# InterFeat

InterFeat is a pipeline for finding **interesting scientific features** by combining statistical/model-based utility filtering, literature mining, biomedical knowledge graphs, and LLM-based annotation.

**Paper:** Ofer D, Linial M, Shahaf D. *InterFeat: a pipeline for finding interesting scientific features.* Scientific Reports 16, 13980 (2026).  
https://doi.org/10.1038/s41598-026-43169-5

## Repository contents

- `code/` — main InterFeat pipeline and analysis code.
- `code/Outputs/` — derived feature-level results, annotations, analyses, and figures used in the paper.
- `SemMed/` — SemMedDB/UMLS knowledge-graph processing.
- `ukbb-hack/` and `Premunge_ukbb_aux_tables.ipynb` — UK Biobank preprocessing code.

Some notebooks require external resources or local path configuration, including UK Biobank data, SemMedDB/UMLS, and MedRAG.

UK Biobank data are not redistributed in this repository and must be obtained directly through UK Biobank under an approved application.

## API credentials

Where required, provide credentials through environment variables (see `.env.example`), rather than placing them in notebooks or source files.


## Citation

If you use us, please cite us!

```bibtex
@article{Ofer2026InterFeat,
  author  = {Ofer, Dan and Linial, Michal and Shahaf, Dafna},
  title   = {InterFeat: a pipeline for finding interesting scientific features},
  journal = {Scientific Reports},
  volume  = {16},
  pages   = {13980},
  year    = {2026},
  doi     = {10.1038/s41598-026-43169-5}
}
```
