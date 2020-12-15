# Dense Passage Retrieval in Conversational Search

The code is splitted between 2 different plateforms, which are Google Collab and Compute Canada. It also contains 2 other repos that are used in the implenetation of this project, DPR repo for "Dense Passage Retrieval", CAsT Dataset, MSMARCo Dataset. Another repo used for evaluation of the CAsT.


## Features
1. BASH files to submit Jobs on different clusters on CoputeCanada.
2. NoteBooks to used on Google Collabs. 
3. Dense retriever model is based on bi-encoder architecture.
4. Dense Passage Retrieval  inspired by [this](https://arxiv.org/abs/2004.04906) paper.
5. Related data pre- and post- processing tools.
6. Dense retriever component for inference time logic is based on FAISS index for the DPR paper.

## 


## Installation

Installation from the source. Python's virtual or Conda environments are recommended.

```bash
git clone https://github.com/AhmedHussKhalifa/Dense_Passage_Retrieval_in_Conversational_Search
cd Dense_Passage_Retrieval_in_Conversational_Search
```

This project is tested on Python 3.6+, PyTorch 1.2.0+ and Transformers 3.5.

