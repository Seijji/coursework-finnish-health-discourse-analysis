# Finnish Health Discourse Analysis - Coursework

Coursework for **Advanced Text Analysis Techniques in Health Sciences**

Analysis of Finnish online forum discourse using Word2Vec semantic analysis and LDA topic modeling. Focus on health-related vocabulary in Finnish Language Bank concordance data.

## Project Structure

```
coursework-finnish-health-discourse-analysis/
├── lemmatization/        # Finnish text lemmatization
│   └── lemmaus.py        # Stanza-based lemmatizer
└── analysis/             # Discourse analysis pipeline
    ├── analyze.py        # Main analysis script
    ├── report_builder.py # HTML report generator
    ├── stopwords.txt     # Finnish stop words
    └── output/           # Output files (generated)
```

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv/bin/activate` on some systems

# Install dependencies
cd lemmatization
pip install stanza

cd ../analysis
pip install gensim pandas numpy matplotlib seaborn scikit-learn
```

### Usage

**1. Lemmatize concordance data:**
```bash
cd lemmatization
python lemmaus.py -i concordance.txt -o lemmatized.txt -d
```

**2. Run discourse analysis:**
```bash
cd ../analysis
python analyze.py -i lemmatized.txt -t rokotevastaisuus -o output
```

**3. View results:**
Open `analysis/output/analysis_report.html` in browser

## Command-Line Arguments

### Lemmatization (`lemmaus.py`)

**Required:**
- `-i, --input`: Input concordance file with raw Finnish text (default: concordance.txt)
  - One sentence per line, UTF-8 encoding

**Optional:**
- `-o, --output`: Output file for lemmatized text (default: lemmatized.txt)
- `-d, --dedup`: Remove duplicate lines before processing
- `--download`: Download Finnish Stanza model (~500MB, required on first use)

### Discourse Analysis (`analyze.py`)

**Required:**
- `-i, --input`: Input file with lemmatized text (one sentence per line, space-separated tokens)
- `-t, --target`: Target word to analyze (must exist in vocabulary)

**Optional:**
- `-s, --stopwords`: Stop words file to filter (default: stopwords.txt)
- `-o, --output`: Output directory for all results (default: output)
- `--topics`: Number of LDA topics (default: 5, recommended: 2-20)
- `--min-count`: Minimum word frequency for Word2Vec vocabulary (default: 10, higher = smaller vocab)
- `--window`: Context window size for Word2Vec and collocations (default: 5, typical: 3-10)
- `--top-words`: Number of words to visualize in t-SNE plot (default: 200)

## Features

### Lemmatization (`lemmatization/`)
- Finnish language lemmatization using Stanza NLP
- Duplicate removal
- Preserves sentence structure

### Discourse Analysis (`analysis/`)
- **Word2Vec semantic analysis** - Skip-gram embeddings with t-SNE visualization
- **LDA topic modeling** - Thematic clustering with configurable topics
- **Collocation analysis** - Context window co-occurrence patterns
- **Keyness analysis** - Log-likelihood distinctive vocabulary per topic
- **Interactive HTML reports** - All visualizations and results

## Requirements

- Python 3.9+
- Stanza (Finnish model)
- gensim, pandas, numpy, matplotlib, seaborn, scikit-learn

Tested on Arch Linux using venv.

## Course Context

Part of coursework for Advanced Text Analysis Techniques in Health Sciences. Analyzes health-related discourse patterns in Finnish online forums using corpus linguistics and NLP methods.

**Note:** This is a hypothetical research project for educational purposes. The analysis methodology is a proof of concept and has not been evaluated for scientific validity or rigour. Not intended for actual research conclusions.

## Acknowledgments

Code scaffolding and debugging assisted by Claude (Anthropic) and OpenCode. Analysis methodology, implementation, and interpretation are original coursework.

Concordance data provided by the Finnish Language Bank (Kielipankki). Data not included in this repository.

## Citations

Honnibal, M., Montani, I., Van Landeghem, S., & Boyd, A. (2020). spaCy: Industrial-strength Natural Language Processing in Python. https://doi.org/10.5281/zenodo.1212303

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, *12*, 2825–2830.

Qi, P., Zhang, Y., Zhang, Y., Bolton, J., & Manning, C. D. (2020). Stanza: A Python Natural Language Processing Toolkit for Many Human Languages. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations*.

Rehurek, R., & Sojka, P. (2011). Gensim–python framework for vector space modelling. *NLP Centre, Faculty of Informatics, Masaryk University, Brno, Czech Republic*, *3*(2).

## License

Public repository for course duration. Academic use only.
