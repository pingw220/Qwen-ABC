# annotations/

Empty on purpose: **no human annotations have been collected** (status PENDING).

Put each annotator's export from `materials/annotate.html` here as `annotations_<annotator>.json`
(a JSON list of records conforming to `../schema.json`), then run
`python -m paper_eval.annotation_rescore --annotations <this dir> --system name=<generations dir> ...`.

Annotation records contain no lyrics (only section indices, labels, counts and free notes) and may be
committed. Do not commit anything from `../materials/`.
