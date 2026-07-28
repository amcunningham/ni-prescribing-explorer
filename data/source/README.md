# data/source — inputs for the STAR-PU denominators

These two files feed `build_starpu_ni_weights.py`, which produces
`data/starpu_denominators_practice.parquet` and `data/starpu_denominators_lcg.parquet`.
They used to live outside the repo in a sibling `../prescribing/` folder; that
folder went missing, so both have been reconstructed from their published
sources and moved in here. Nothing in the pipeline now depends on a path
outside the repository.

## practice_demographics.parquet

Registered patients by practice, gender and age group, 2014–2026.

- Source: BSO Family Practitioner Services, *Registered Patients by Practice,
  Age and Sex* —
  https://bso.hscni.net/directorates/operations/family-practitioner-services/directorates-operations-family-practitioner-services-information-unit/1776-2/
- The workbook holds one sheet per year (`1.1a` = 2014 … `1.1m` = 2026), each
  already using the four BSO age bands, so no age regrouping is needed.
- Columns: `year, practice, sex, age_band, population`.

Two filters are applied when reading it, and both matter:

- Rows with `Gender = Unknown` (about 2,200 across all years) and rows with a
  blank gender (about 300) are **dropped**. STAR-PU weights only exist for male
  and female, so these patients cannot be weighted. This means
  `total_population` in the denominators is the male + female count, not the
  full practice list size. That matches how the original denominators were
  built — verified exactly, to the patient.
- Only the four standard age bands are kept.

To refresh: download the current year's workbook from the BSO page above and
re-run the extraction. The file is republished each June, so the 2027 edition
should appear around June 2027.

## ni_starpu_2023_collapsed_bso_bands.csv

NI STAR-PU weights by age band, sex and BNF chapter — 80 rows (4 bands ×
2 sexes × 10 chapters).

- Ultimate source: Department of Health NI, *PFR2024_02: STAR-PU*, part of the
  Prescribing Formula Review —
  https://www.health-ni.gov.uk/publications/northern-ireland-general-practice-prescribing-formula-detailed-papers-associated-data-files
- The published table (Table 1.1 of the PFR2024_02 datafile) uses nine age
  bands: 0-4, 5-15, 16-24, 25-44, 45-59, 60-64, 65-74, 75-84, 85+. These do not
  nest inside the four BSO bands — 16-24 straddles the boundary at 18 — so the
  published weights cannot be collapsed without population data at a finer age
  resolution than the BSO file provides.

Rather than guess at that collapse, the weights in this CSV were **recovered
from the existing denominators by least squares**: with the practice
populations known, `starpu = Σ(population × weight)` over the eight
band × sex cells is an overdetermined linear system with roughly 300 practices
per equation set. The fit is exact — residuals at machine precision (~1e-14),
and the recovered weights are identical across all twelve years, as they must
be if the model is right. Rebuilding the 2014–2025 denominators from these
weights reproduces all 39,420 original rows, with a maximum difference of
4e-10 on STAR-PU and exactly zero on population.

So these are the original weights, not an approximation of them. If you ever
need to re-derive them from the published table instead, you would need NI
population by single year of age (NISRA mid-year estimates) to split the 16-24
band at 18.

Columns: `bso_age_band, sex, chapter, ni_starpu_weight`, where `chapter` is
`BNF1`, `BNF2`, … matching `CHAPTER_MAP` in `build_starpu_ni_weights.py`
(chapters 1–7, 9, 10, 13 — the ten with published weights).
