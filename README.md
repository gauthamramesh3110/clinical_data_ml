# Care Plan Prediction from FHIR Data — Transformer vs. LSTM

Given everything in a patient's record, which care plan comes next — and does the
architecture reading that record actually matter?

Two models answer the same question from the same Synthea FHIR data: a
**time-aware Transformer** over a unified event timeline, and a **multi-layer
LSTM** over one-hot encounter vectors plus demographics.

## Results

| Metric                       | Transformer | LSTM  |
| ---------------------------- | ----------- | ----- |
| Macro precision              | **0.75**    | 0.10  |
| Macro recall                 | **0.70**    | 0.20  |
| Macro F1                     | **0.66**    | 0.11  |
| Top-1 / multi-label accuracy | **70%**     | 88% † |

The Transformer also reaches **91% top-3** and **97% top-5**.

† **The LSTM's 0.88 is not what it looks like.** The targets are 20 binary
labels that are overwhelmingly negative, so predicting all-zeros scores about the
same. Per-class confusion matrices confirm it: for most care plan types the LSTM
gets zero true positives and predicts only the trivial negative class. The
per-class ROC curves look respectable for a few common plans (Care Plan 0.99,
Cancer 0.98, Dementia management 0.95) and collapse for the rest.

This is the reason macro metrics are reported first here. Accuracy on a sparse
multi-label target measures the label distribution, not the model.

## Data

[Synthea](https://github.com/synthetichealth/synthea) synthetic patient records,
April 2020 CSV release, ~1,000 patients with complete histories across allergies,
care plans, conditions, encounters, immunizations, medications, observations and
procedures — coded in SNOMED-CT, RxNorm, CVX and LOINC.

Observations dominate the record volume (median ≈ 164 per patient) while care
plans are sparse (median ≈ 3), and encounter counts vary widely between patients.

Synthetic data means no privacy constraints, but also no real-world messiness —
see [Limitations](#limitations).

## The two pipelines

### Transformer — one timeline, one vocabulary

Six clinical tables are loaded and prefixed by domain (`COND_`, `MED_`, `OBS_`, …)
so every event type shares a single vocabulary of ~5,000 tokens. Encounter
timestamps are merged and `DAYS_SINCE_START` computed relative to the patient's
first event. A custom `MultimodalTokenizer` assigns integer indices including
`<PAD>` and `<UNK>`.

The target is the **latest** care plan code; the inputs are all events preceding
it, which is what prevents leakage.

| Component      | Choice                                                       |
| -------------- | ------------------------------------------------------------ |
| Token embedding| `nn.Embedding` → d_model = 64, scaled by √d_model             |
| Time embedding | log(1 + Δt), linearly projected to d_model, summed with token |
| Encoder        | single-layer `TransformerEncoder`, 2 heads, key padding mask  |
| Head           | output at last non-padded position → linear over C classes    |
| Training       | cross-entropy, Adam (lr 1e-4, wd 1e-4), grad clip 1.0, dropout 0.3, 100 epochs, best-model checkpointing on validation loss |

### LSTM — encounter-level, multi-label

Data is organised per encounter: each clinical domain one-hot encoded into a wide
feature vector, with demographics (age, marital status, race, ethnicity, gender)
encoded separately as static features. The target is a multi-label binary vector
over the 20 most frequent care plan types, with care plan features shifted one
step to prevent leakage. Sequences are padded to a maximum length of 500.

| Component  | Choice                                                          |
| ---------- | --------------------------------------------------------------- |
| Static path| linear projection of demographics → 128                          |
| Temporal   | 6-layer LSTM, h = 128, packed sequences, last valid time step    |
| Head       | concat (256) → 2 FC layers, dropout 0.1 → 20 logits              |
| Training   | `BCEWithLogitsLoss` with class-specific positive weights, Adam (lr 1e-3, wd 1e-3), `ReduceLROnPlateau` (patience 5), early stopping (patience 10), PyTorch Lightning |

## Why the gap

1. **Self-attention keeps the whole timeline.** It computes direct pairwise
   scores between all events, while the LSTM's hidden state progressively
   overwrites older clinical signal — most damaging for patients with more than
   200 encounters.
2. **Time is a feature, not an ordering.** log(1 + Δt) makes irregular visit
   intervals explicit; the LSTM only gets the implicit signal of encounter order.
3. **Capacity is not the story.** The Transformer wins with *fewer* parameters
   (1 layer, 2 heads against 6 LSTM layers). Architectural inductive bias matters
   more than size on this task.

## Interpretability

Predictions are attributed to individual clinical events with
[Integrated Gradients](https://arxiv.org/abs/1703.01365) via
[Captum](https://captum.ai/). A custom `EmbeddingWrapper` enables gradient-based
attribution on token embeddings; attributions are computed from a zero-tensor
baseline to the actual embeddings and summed across the embedding dimension for
per-token importance.

On a sample patient, the strongest positive contributor to a predicted **Head
Injury Rehabilitation** plan was `COND_62106007` (concussion with no loss of
consciousness) at Day 1980, attribution ≈ 7.9 — well above the observation scores
(QOLS, DALY) that follow it. Earlier routine observations contributed negatively.
The model prioritised the acute event over years of routine data, which is what a
clinician would do.

## Files

| File                                   | Contents                          |
| -------------------------------------- | --------------------------------- |
| `Care_Plan_Suggestor_Transformer.ipynb`| Transformer pipeline, training, evaluation, Integrated Gradients |
| `CarePlanSuggestorLSTM.ipynb`          | LSTM pipeline, training, evaluation, per-class ROC |

Also runnable on Colab:
[Transformer](https://colab.research.google.com/drive/1ytutp_cLkgfuVJ0NuspkxBZxn02ZljYO?usp=sharing) ·
[LSTM](https://colab.research.google.com/drive/1wrC9cOFCD5MWrGzBh1btrS4uAr-zValJ?usp=sharing)

## Limitations

- **Synthetic data only.** Synthea records are realistic in structure but not in
  the noise, missingness and coding inconsistency of real EHR data.
- **~1,000 patients.** Small enough that rare care plans have very little support,
  which is visible in the per-class results.
- **The two models solve differently-shaped targets.** The Transformer predicts a
  single latest care plan code; the LSTM predicts a 20-way multi-label vector.
  Macro precision/recall/F1 are comparable across the two, and the head-to-head
  table uses them for that reason — but the accuracy columns are not measuring
  the same thing, which is exactly why the LSTM's 0.88 needs its footnote.
- **Class imbalance is handled but not solved.** Care plan frequencies are
  long-tailed; positive class weighting helps the LSTM's loss without producing
  true positives on rare plans.

## Next

- Ensemble the Transformer's precision with the LSTM's multi-label capability
- Pre-train via masked event prediction on a larger EHR corpus
- Validate on de-identified real-world records
- Multi-modal inputs: clinical notes, imaging, genomic data
- A clinician-facing dashboard presenting recommendations with attributions

## Authors

Gautham Ramesh Babu and Farhan Patel, Department of Computer Science,
Western University.
