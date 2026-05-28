Citation Intent Classification using SciBERT

This project implements a modular NLP pipeline for **Citation Intent Classification**

Baseline Model
Dataset: SciCite
Encoder: SciBERT (allenai/scibert_scivocab_uncased)
Task: Citation Intent Classification
Classes:
background
method
result

# Running the Project on Google Colab

To run this project on Google Colab, first open Google Colab and create a new notebook. Before running the project, enable GPU support by going to **Runtime → Change runtime type → GPU**. This ensures faster training using Colab’s free GPU resources. Next, clone the GitHub repository into the Colab environment using:
!git clone https://github.com/akanshaj107/CitationIntentClassifier.git
install all required dependencies using the requirements.txt file: !pip install -r requirements.txt
Finally, run the complete citation intent classification pipeline using: !python -m experiments.run_scicite
