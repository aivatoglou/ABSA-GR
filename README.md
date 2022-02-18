### Aspect-Based Sentiment Analysis in Greek Data

In this repo you can find the thesis for my MSc degree in Artificial Intelligenece. The methodology tackles the problem of aspect-based sentiment analysis methodology in low resource languages and especially in Greek. The dataset was derived from Twitter and the aspects, labels were carefully chosen manually by human annotators. 

#### * Abstract *
Due to the exponential growth of data the recent years, many businesses and organizations seek innovative techniques in order to collect and understand data. Since they are most interested in the public opinion for their products and services they implement methodologies for sentiment classification. These techniques are part of Sentiment Analysis and Natural Language Processing with the goal to automatically understand the sentiment given a corpus. Although the proliferation of text documents caused problems in the last decade that were mostly related to computational resources, nowadays we are able to analyze data more efficiently due to the Deep Learning architectures. However, many languages are characterized as low-resources languages due to the limited data that are available online for analysis. For this reason in this thesis an aspect-based Sentiment Analysis methodology is presented, targeting to classify the aspects of a sentence into pre-defined sentiment categories. Specifically, the dataset of this study is written in the Greek language and was collected from social networks like Twitter. Moreover, the labels, both aspects and sentiments, were manually chosen from expert annotators. Furthermore, various language models and deep learning architectures were developed and tested. Finally, the results of our best architecture, which is a combination of neural machine translation and a language-model ensemble methodology clearly proved the necessity of machine translation and the superiority of the Transformer-based architectures achieving ambitious results for the problem of aspect-based Sentiment Analysis in low-resource languages.

#### * Data *
The format of the data is the following:

| text | target | sentiment |
| :---: | :---: | :---: |
| str | str | str |

The text column contains the tweets, the target column contains the aspect extracted from the tweets, and the sentiment column contains the polarity of the tweet regarding the aspect and has one of the following five values: -2 (very negative), -1 (negative), 0 (neutral), 1 (positive), 2 (very positive).

#### * Methodology *
The methodology takes advantage of neural machine translation techniques in order to deal with any imbalance problems.

#### * Requirements *
The environment of the experiments was implemented using Anaconcda. In general, python and pytorch are required and some additional packages that can be found inside the environment.yml file.

#### * How to run *

"cd" to the appropriate folder and run:
```console
foo@bar:~$ python3 main.py
```

#### * Results *

<table>
<tr><th> Ensemble methodology </th><th> Hybrid methodology </th></tr>
<tr><td>
  
| Class  | F1-score |
| ------------- | ------------- |
| negative | 0.69  |
| neutral  | 0.85  |
| positive  | 0.54  |

</td><td>

| Class  | F1-score |
| ------------- | ------------- |
| negative | 0.62  |
| neutral  | 0.82  |
| positive  | 0.54  |

</td></tr> </table>
