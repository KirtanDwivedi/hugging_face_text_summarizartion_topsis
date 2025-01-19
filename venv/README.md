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
├── txt_summarization.py    # Main script for model evaluation
├── requirements.txt        # Package dependencies
└── README.md              # Project documentation
```

## Usage

1. Run the main script:

```bash
python txt_summarization.py
```

2. The script will:
   - Load the XSUM dataset
   - Download and initialize models
   - Generate summaries
   - Calculate evaluation metrics
   - Create a CSV file with results

## Expected Output

The script generates a CSV file (`politics.csv`) containing:

- Model alternatives (M1-M5)
- Evaluation criteria scores
- Final TOPSIS rankings

## Requirements

- Python 3.8+

## License

This project is licensed under the MIT License - see the LICENSE file for details.
