# Dataset

The experiments use the **ACL Citation Sentiment Corpus** (Athar, 2011) — a set
of citation sentences from the ACL Anthology, each labelled with the sentiment
the citing paper expresses toward the cited paper.

The code expects a single tab-separated file here:

```
data/citation_sentiment_corpus.txt
```

> The corpus is **not committed** to this repository (see `.gitignore`). Place
> your copy at the path above, or point any run script at it with
> `--data-path /path/to/file.txt`.

## Expected format

Four tab-separated columns, **no header**:

| column | name             | description                                   |
|--------|------------------|-----------------------------------------------|
| 1      | `Source_PaperID` | citing paper id (e.g. an ACL Anthology id)    |
| 2      | `Target_PaperID` | cited paper id                                |
| 3      | `Sentiment`      | one of `p` (positive), `o` (objective), `n` (negative) |
| 4      | `Citation_text`  | the citation sentence                         |

Example row:

```
A92-1018	J90-1003	o	The system described in ( ... ) uses a similar approach.
```

The loader (`src/common.py`) maps the sentiment labels to integers:
`n → 0`, `o → 1`, `p → 2`.

## Reference

> Awais Athar. *Sentiment Analysis of Citations using Sentence Structure-Based
> Features.* Proceedings of the ACL 2011 Student Session, 2011.
