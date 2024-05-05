# Speaker Identification

This repository contains data and code for the project of speaker identification. End-to-end speaker identification is done in two stages:
- Person name recognition: given a meeting transcript, find spans for person names for each sentence in the transcript.
- Speaker identification: given a meeting transcript and person names, find names for speaker IDs.

The following documentation provides instructions to run the code for models of each of the two subtasks.

## Preparation

Before following the instructions below, make sure that you create a conda environment as below:
```
conda create -n speaker-id python=3.8
conda activate speaker-id

pip install transformers==3.5.1
pip install torch==1.7.1
pip install trankit==1.1.1
pip install nltk==3.6.7
pip install spacy==3.3.0
pip install thefuzz==0.19.0
pip install six==1.16.0
pip install protobuf==3.20.0 
pip install xlsxwriter==3.0.3
```

## Person Name Recognition
We developed a model for detecting person names in text. The model follows a standard NER model architecture: a BERT-based encoder for producing contextualized representations, a Conditional Random Field layer for tagging start and end words of person names based on the BERT representations. The implementation of the model can be found in the class *TranscriptNER* defined in the file `./individual_model.py`.

### Training
To train the custom NER model, please run the command:
```
python train.py --dataset CoNLL03-English-synthetic
```
Here, we use *CoNLL03-English-synthetic*, which is a special version containing both the original version of the well-known [CoNLL03-English](https://huggingface.co/datasets/conll2003) dataset and its lower-cased version, to train the model. This helps improve the robustness of the custom NER model in predicting person names in transcript data, where names could be incorrectly lowercased by an ASR system. The trained model will be stored at `logs/CoNLL03-English-synthetic/best-model.mdl`. A trained custom NER model can be found at: [Google Drive link](https://drive.google.com/file/d/1qHtFnjENHR6cOZJX1-wDzvLOTIM580dY/view?usp=sharing).

## Speaker Identification
For speaker identification, we developed two main models:
- Individual Model: given an inidividual person name, the model tries to decide if the name belongs to any speaker around.
- Joint Model: given a sentence containing multiple names, the model tries to decide if each of the given names belongs to any speaker around.

### SpeakerID Data Generation
We use the trained NER model to make predictions on [Mediasum Corpus](https://github.com/zcgzcgzcg1/MediaSum) to obtain person names on transcripts. Afterwards, text matching is performed to obtain training data for speaker identification. To run this process, use the following command:

```
python util.py
```
The preprocessed dataset will be stored at `datasets/mediasum`. The preprocessed dataset can be downloaded at: [Google Drive link](https://drive.google.com/file/d/1yfbFL2NtKcVFgslGGgGzh8jjRUbl5Afz/view?usp=sharing).

### Training
To train the models, please use the following command:
```
python train.py --dataset mediasum --model_type [joint|individual] 
```
In this command:
    - [REQUIRED] `--model_type [joint|individual]` specifies what type of models that we use for training. `joint` means the Joint Model while `individual` means the Individual Model.

Trained models can be found at: [Google Drive link](https://drive.google.com/file/d/1qHtFnjENHR6cOZJX1-wDzvLOTIM580dY/view?usp=sharing).

## License

The code and model are licensed under the [Adobe Research License](./LICENSE.md). The license prohibits commercial use and allows non-commercial research use. 

