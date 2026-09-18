# NIF and oracle-frame annotation

NIF records the number of distinct evidence frames retained for a question under the annotation procedure below. The [sampled records](../data/sampled_records/) contain the video IDs, available question identifiers or text, recorded NIF values and evidence timestamps. The [caption download](reproduction-data.md) uses the same four 90-video lists.

## Annotation procedure

Three annotators participate, one author and two volunteers.

1. The first annotator identifies frames needed to answer the question and records their timestamps. RAG-Adapter can accelerate candidate search. The annotator also searches elsewhere in the video for missing evidence, so the final evidence set is determined manually.
2. The two other annotators independently answer the question using the proposed evidence frames.
3. If either answer is incorrect, the question is recorded for revision and the first annotator supplements the evidence. This repeats until both annotators answer correctly.
4. During answering, the two annotators flag redundant or irrelevant frames. These are removed from the final evidence set.
5. Adjacent frames containing the same answer-relevant information contribute one representative frame. Distinct events or states needed for counting or ordering are retained separately. Record the resulting timestamps and NIF count.

This protocol describes evidence sufficiency under the given candidate and review procedure. It does not establish global minimality over all possible video-frame subsets or measure reasoning difficulty. Original individual answer sheets, disagreement logs and independent difficulty labels are not reconstructed by this release.

## Oracle-frame evaluation

Use all manually identified evidence frames for each question. When fewer than the automatic methods' budget K are annotated, do not pad the oracle input with additional frames. The oracle frame count varies with the question's NIF. Sort the frames chronologically and use the same answering and option-parsing procedure as the corresponding model evaluation.

The released CSV files preserve the recorded counts and timestamps. Empty fields remain empty. If a historical record's NIF count and listed timestamps disagree, that record requires reconciliation before regenerating oracle inputs. Neither value should silently overwrite the other.

## Sampling and aggregation

The sampling guide distinguishes the diagnostic samples from the expanded main evaluation sets. Dataset-level mean NIF is calculated over the annotated questions in the stated sample. Retain the question-level records so the weighting can be inspected. ASS is a separate retrieval diagnostic tied to its specified encoders, captions, candidate lists and scoring configuration. Its archived calculation is provided in the research notebook.
