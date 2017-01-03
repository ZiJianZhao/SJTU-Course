This is the code of the Sentiment Analysis project in SJTU Cource NLP.

Environment setup: Linux, sklearn, nltk, numpy, MXNet with python interface.

Code infrastructure:
1. baseline.py: implements the lr(Logistic Regression), rf(Random Forest) and nb(Naive Bayes)
classifiers with w2c(Word2vec) and bow(Bag of Words) features. You can run this file:

```Python
python baseline.py --train_file filepath --valid_file filepath --w2v_file filepath|None --classifier lr|rf|nb --feature bow|w2c
```

2. baseline_text.py: support preprocess functions for baseline.py
3. visual.py: visualization functions

4. train.py: define the training function of the CNN model, you can run this file:

```Python
python train.py --train_file filepath --valid_file filepath --test_file filepath --mode rand|static|non_static --w2v_file filepath|None --machine gpu|cpu
```
Note: train_file and valid_file both have labels but test_file doesn't have labels.

5. cnn.py: implements the CNN model.
6. text_io.py: support preprocess functions for train.py
7. data_iter.py: data iterator function for train.py
