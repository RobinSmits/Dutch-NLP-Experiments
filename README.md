# Dutch language experiments with Multi-lingual Transformer models

## Introduction

While the majority of current Transformer models is focused on English (or a small set of languages) a few Transformer models are truly multi-lingual as for example multi-lingual BERT (I'll further call this MBERT ;-) ) or XLM-RoBERTa. Recently some more Transformer models became available that support a large number of languages (mT5 and ByT5).

As my own native language is Dutch and not English I already had the idea for a while to try out some of these models on a Dutch language dataset. I would like to see what they are capable of doing .. or not. Also being able to make a comparison between them and see what performance they would be capable of.

Recently I came accross a Dutch news articles dataset which contained for each article labels for whether or not the article was partisan. I also noticed that to my best knowledge no existing work with that dataset and Transformer models was available...so it was time to start experimenting.

All code was created based on Tensorflow and the Huggingface Transformers Library. All models were trained on Google Colab Pro TPUv2 runtimes.

Ideas completed so far:
- Basic Exploratory Data Analysis
- Train and evaluate the performance of MBERT and XLM-RoBERTa as classifiers of partisan news articles.

Ideas that I'am currently researching/coding/experimenting:
- Pre-training MBERT and XLM-RoBERTa Masked LM models. After that again fine-tuning as classifiers. What is the difference with just fine-tuning the models as classifiers?
- Use MBERT and XLM-RoBERTa as feature extractors and perform classification with SVM.
- Train and evaluate mT5 and ByT5 as classifiers.

The remaining ideas should be completed in the next few weeks (July 2021)

Note! If anyone is aware of more multi-lingual models that support Dutch and you would like me to add those...put in a request through an Issue.

## Dataset

The dataset used in my experiment is the [DpgMedia2019: A Dutch News Dataset for Partisanship Detection](https://github.com/dpgmedia/partisan-news2019) dataset.
It contains various parts but the main part I use is a set of about 104K news articles. For each article there is a label 'partisan' stating whether the article is partisan or not (True/False). The amount of partisan / non-partisan articles is roughly balanced.

The dataset was created by the authors with the aim to contribute to for example create a partisan news detector. In the python code used in the experiments the specific dataset files are downloaded automatically. Checkout the github and paper for more information about the dataset and how it whas constructed. See the References for the information.

## Multi-Lingual BERT and XLM-RoBERTa

The python file 'train_mbert_xlmroberta.py' contains all the code to download and process the data. The training is performed based on 3 different rounds with each round containing a full 5 fold stratified Cross Validation training run. The average validation score is determined by taking the mean of the validation accuracy across all 15 trained models.

Training was performed on Google Colab Pro TPUv2 hardware. With a batch_size of 64, learning rate of 0.00002, 3 epochs and the maximum token input length of 512 the training process for all 3 rounds with 5 fold CV's could be completed within about 6 to 7 hours.

For both MBERT and XLM-RoBERTa I performed this process on 2 different model setup's:
- For each model we used the default Huggingface Transformers SequenceClassification Models
- For each model we used the the BaseModel (TFBertModel / TFRobertaModel) and added a custom classification head.

The achieved performance can be seen in below table. XLM-RoBERTa scores slightly higher than MBERT...but with scores around 95-96% they are both excellent classifiers.

| Transformer Model Type and Architecture | Average Validation Accuracy (%) Score |
|:---------------|----------------:|
| MBERT Standard Sequence Classification Model | 95.33 |
| MBERT Custom Sequence Classification Model | 95.20 |
| XLM-RoBERTa Standard Sequence Classification Model | 95.87 |
| XLM-RoBERTa Custom Sequence Classification Model | 96.20  |

## Exploratory Data Analysis

<< To Be Documented Soon ... >>

## References

```
@misc{1908.02322,
  Author = {Chia-Lun Yeh and Babak Loni and Mariëlle Hendriks and Henrike Reinhardt and Anne Schuth},
  Title = {DpgMedia2019: A Dutch News Dataset for Partisanship Detection},
  Year = {2019},
  Eprint = {arXiv:1908.02322},
}
```