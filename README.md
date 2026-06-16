# CitationIntentClassifier

A unified framework for Citation Intent Classification across multiple taxonomies using BERT and SciBERT.

## Supported Taxonomies

### SciCite
- Background
- Method
- Result

### ACL-ARC
- Background
- Uses
- CompareOrContrast
- Motivation
- Extends
- Future

### SOFT
**Citation Function**
- Contextualize
- SignalGap
- HighlightLimitation
- JustifyDesignChoice
- Use
- Modify
- EvaluateAgainst

**Citation Content**
- Performed Work
- Discovery
- Produced Resource

---

## Models

- BERT (`bert-base-uncased`)
- SciBERT (`allenai/scibert_scivocab_uncased`)

---

## Project Structure

```text
configs/       # Experiment configurations
data/          # Datasets
experiments/   # Run scripts
scripts/       # Dataset builders
src/           # Core pipeline
outputs/       # Metrics and predictions
```  qa

---

## Running Experiments

# Configuration

Before running an experiment, update the corresponding configuration file in `current configs/`.
Available configurations:

- `scicite_scibert_config.py`
- `scicite_bert_config.py`
- `acl_arc_scibert_config.py`
- `acl_arc_bert_config.py`
- `soft_scibert_config.py`
- `soft_bert_config.py`

### SciCite

```bash
python -m experiments.run_scicite
```

### ACL-ARC

```bash
python -m experiments.run_acl_arc
```

### SOFT

```bash
python -m experiments.run_soft
```

---

## Outputs

Each experiment saves:

```text
outputs/<experiment_name>/

├── metrics/
│   ├── metrics.json
│   ├── classification_report.txt
│   └── confusion_matrix.csv
│
└── predictions/
    └── test_predictions.csv
```

## Future Work

- LLM Prompting (Zero-Shot / Few-Shot)
- DSPy-based Prompt Optimization
- Qwen Models
- Triply Annotated Benchmark Evaluation
- Cross-Taxonomy Comparison

---

## References

- SciCite (Cohan et al., 2019)
- ACL-ARC (Jurgens et al., 2018)
- SOFT Citation Classification Framework
