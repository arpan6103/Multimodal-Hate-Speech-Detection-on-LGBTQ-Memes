# Multimodal-Hate-Speech-Detection-on-LGBTQ-Memes
This project focuses on detecting hate speech targeting the LGBTQ+ community in internet memes using a multimodal machine learning pipeline. It combines OCR-extracted text and image features, encoded using CLIP, and classifies the content using a custom-built MLP neural network.

**Model Overview**
  OCR Preprocessing:
    - Text extracted from memes using OCR (pre-annotated CSV)
    - Spell corrected using TextBlob and pyspellchecker
  Feature Extraction:
    - Text Embeddings: CLIP Text Encoder
    - Image Embeddings: CLIP Image Encoder (ViT or ResNet variants)
  Embedding Fusion:
    - Text and image embeddings concatenated for multimodal representation
  Classifier:
    - Multi-Layer Perceptron (MLP)
    - Tuned using different optimizers, dropout, and activation functions

**Results**
Best Model performed: ViT-V14(336px):
  Accuracy: 93.94%
  Optimizer: AdamW
  Learning Rate: 0.01
  Weight Decay=0
  Activation fn: Leaky ReLu
  Loss Functon: FocalLoss
  Layers: [512,256,128]
