# Visual Caption Evaluation

Improved evaluation codes for common visual captioning metrics (base on [coco-caption](https://github.com/tylin/coco-caption)).

1. bleu, meteor, rouge, cider, spice
2. supporting python3 
3. faster cider with cache for reference captions

In order to use spice metric, you should download [stanford-corenlp-3.6.0-models.jar](http://stanfordnlp.github.io/CoreNLP/index.html) into the spice/lib directory.

```
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2015-12-09.zip
unzip stanford-corenlp-full-2015-12-09
cd stanford-corenlp-full-2015-12-09
mv stanford-corenlp-full-2015-12-09/stanford-corenlp-3.6.0-models.jar spice/lib
mv stanford-corenlp-full-2015-12-09/stanford-corenlp-3.6.0.jar spice/lib
```
