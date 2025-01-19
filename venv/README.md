# Project Description

# Text Summarization Model Evaluation using TOPSIS

This project evaluates different pre-trained models for text summarization using the TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) method. We analyze models across various domains (Sports, Politics, Finance, Science) from the XSUM dataset.

## Overview

The project follows these key steps:

1. Domain Classification
2. Parameter Evaluation
3. Impact Analysis
4. Metric Calculation
5. TOPSIS Ranking

## List of Domains of xsum dataset

1. Politics
2. Science
3. Sports
4. Finance

## Models Evaluated

- BART (facebook/bart-base)
- PEGASUS (google/pegasus-xsum)
- T5 (t5-small)
- LongT5 (google/long-t5-tglobal-base)
- LED (allenai/led-base-16384)

## Evaluation Parameters

The following parameters are used to evaluate the summarization models:

1. ROUGE Scores (Maximize)

   - Measures overlap between generated and reference summaries
   - Includes ROUGE-1, ROUGE-2, and ROUGE-L

2. BLEU Score (Maximize)

   - Evaluates the quality of machine-generated text

3. METEOR Score (Maximize)

   - Measures translation quality by considering synonyms

4. BERTScore (Maximize)

   - Computes similarity using contextual embeddings

5. Length Difference (Minimize)

   - Absolute difference between generated and reference summary lengths

6. Compression Ratio (Minimize)
   - Ratio of summary length to original text length

## Setup Instructions

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:

```bash
pip install transformers datasets evaluate bert-score torch nltk rouge_score absl-py
```

3. Download NLTK data:

```python
import nltk
nltk.download('punkt')
```

## Project Structure

```
.
├── txt_summarization.py    # Main script for model evaluation create the csv files of every domain
├── topsis_evaluation       # Evaluate the topsis ranking of every model for respective domain and tell which model is best among them
└── README.md               # Project documentation
```

## Usage

1. Run the txt_summarization script:

```bash
python txt_summarization.py
```

The script will:

- Load the XSUM dataset
- Download and initialize models
- Generate summaries
- Calculate evaluation metrics
- Create a CSV file with results

3. Run the topsis_evaluation script:

```bash
python topsis_evaluation.py
```

The script will:

- Evaluate the topsis ranking of every model for respective domain
- Print which model is best among them over all the domains

## Expected Output

### Expected Output from txt_summarization -

The script generates CSV files (`politics.csv`, `science.csv`, `sports.csv`, `finance.csv`) containing:

- Model alternatives (M1-M5)
- Evaluation criteria scores
- Final TOPSIS rankings

### Expected Output from topsis_evaluation -

The script generates result :

```bash
The best model for xsum dataset is : {most_frequent_best_model}
```

most_frequent_best_model are :

- BART (facebook/bart-base)
- PEGASUS (google/pegasus-xsum)
- T5 (t5-small)
- LongT5 (google/long-t5-tglobal-base)
- LED (allenai/led-base-16384)

## Requirements

- Python 3.8+

## License

This project is licensed under the MIT License - see the LICENSE file for details.
