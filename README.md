# Leveraging Few-shot with Federated Learning and Ensembling for Text Classification
[Upcoming paper]
### Supervised Project for MSc NLP @ Université de Lorraine

## Abstract
In the Natural Language Processing (NLP) domain, zero-shot learning has become a mainstream paradigm to exploit large pre-trained models, such as GPT-3, T5 and BART in various tasks without any supervised label. However, fine-tuning these models on a few samples in a few-shot setting is still useful to adapt to a specific user or task. We propose to leverage few-shot learning of state-of-the-art NLP models with federated learning with the objective of pooling together several such specific fine-tuned models into a single central model that is better than any individual model, without having access to private local data. Our experiments are carried out using both an Induction Network and a pretrained BART model for sentiment analysis in English on the Amazon Review benchmark.

## Repo Structure

    supervised-project-nlp
        ├── Bibliography
        ├── Paper
        └── Realisation
                ├── DiverseFewShot_Amazon
                ├── bart model
                ├── federated transfer model
                ├── induction network
                └── models
