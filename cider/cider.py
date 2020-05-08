from __future__ import print_function
from __future__ import division

import copy
import numpy as np
import math

def precook(s, n=4, out=False):
  """
  Takes a string as input and returns an object that can be given to
  either cook_refs or cook_test. This is optional: cook_refs and cook_test
  can take string arguments as well.
  :param s: string : sentence to be converted into ngrams
  :param n: int    : number of ngrams for which representation is calculated
  :return: term frequency vector for occuring ngrams
  """
  words = s.split()
  counts = {}
  for k in range(1, n+1):
    for i in range(len(words)-k+1):
      ngram = tuple(words[i: i+k])
      counts.setdefault(ngram, 0)
      counts[ngram] += 1
  return counts

def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
  '''Takes a list of reference sentences for a single segment
  and returns an object that encapsulates everything that BLEU
  needs to know about them.
  :param refs: list of string : reference sentences for some image
  :param n: int : number of ngrams for which (ngram) representation is calculated
  :return: result (list of dict)
  '''
  return [precook(ref, n) for ref in refs]

def cook_test(test, n=4):
  '''Takes a test sentence and returns an object that
  encapsulates everything that BLEU needs to know about it.
  :param test: list of string : hypothesis sentence for some image
  :param n: int : number of ngrams for which (ngram) representation is calculated
  :return: result (dict)
  '''
  return precook(test, n, True)

class Cider(object):
  """CIDEr scorer.
  """
  def __init__(self, n=4, sigma=6.0):
    ''' singular instance '''
    self.n = n
    self.sigma = sigma
    self.idf = None
    self.log_ndocs = None


    # penalize for the mismatch length of the pred and refs (max diff length set as 100)
    self.delta_smooth_tab = {}
    for i in range(0, 100):
      self.delta_smooth_tab[i] = np.e**(-(float(i)**2)/(2*self.sigma**2))

    # ref representation: only need to calculate once
    self.sent2rep = {}
    self.is_init_refs = False

  def init_refs(self, vid2refs):
    self.is_init_refs = True
    self.sent2rep = {}

    vid2ngrams = {}
    for vid, refs in vid2refs.items():
      ref_ngrams = cook_refs(refs)
      vid2ngrams[vid] = ref_ngrams

    document_frequency, ndocs = self.compute_doc_freq(vid2ngrams.values())
    self.idf, self.log_ndocs = self.compute_idf(document_frequency, ndocs)

    for vid, sents in vid2refs.items():
      for sent, ngrams in zip(sents, vid2ngrams[vid]):
        if sent not in self.sent2rep:
          vec, norm, length = self._counts2vec(ngrams)
          self.sent2rep[sent] = (vec, norm, length)

  def compute_doc_freq(self, crefs):
    '''
    Compute term frequency for reference data.
    This will be used to compute idf (inverse document frequency later)
    The term frequency is stored in the object
    :return: None
    '''
    document_frequency = {}
    for refs in crefs:
      # refs, k ref captions of one image
      for ngram in set([ngram for ref in refs for (ngram,count) in ref.items()]):
        document_frequency.setdefault(ngram, 0)
        document_frequency[ngram] += 1
    ndocs = len(crefs)
    return document_frequency, ndocs

  def compute_idf(self, document_frequency, ndocs):
    log_ndocs = np.log(ndocs)
    idf = {}
    for ngram, freq in document_frequency.items():
      idf[ngram] = log_ndocs - np.log(freq)
    return idf, log_ndocs

  def _counts2vec(self, cnts):
    """
    Function maps counts of ngram to vector of tfidf weights.
    The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
    The n-th entry of array denotes length of n-grams.
    :param cnts:
    :return: vec (array of dict), norm (array of float), length (int)
    """
    vec = [{} for _ in range(self.n)]
    length = 0
    norm = [0.0 for _ in range(self.n)]
    for (ngram, term_freq) in cnts.items():
      # ngram index
      n = len(ngram)-1
      # tf (term_freq) * idf (precomputed idf) for n-grams
      vec[n][ngram] = float(term_freq) * self.idf.get(ngram, self.log_ndocs)
      # compute norm for the vector.  the norm will be used for computing similarity
      norm[n] += pow(vec[n][ngram], 2)

      if n == 1:
        length += term_freq
    norm = [np.sqrt(n) for n in norm]
    return vec, norm, length

  def _sim(self, vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
    '''
    Compute the cosine similarity of two vectors.
    :param vec_hyp: array of dictionary for vector corresponding to hypothesis
    :param vec_ref: array of dictionary for vector corresponding to reference
    :param norm_hyp: array of float for vector corresponding to hypothesis
    :param norm_ref: array of float for vector corresponding to reference
    :param length_hyp: int containing length of hypothesis
    :param length_ref: int containing length of reference
    :return: array of score for each n-grams cosine similarity
    '''
    delta = abs(length_hyp - length_ref)
    # measure consine similarity
    val = np.array([0.0 for _ in range(self.n)])
    for n in range(self.n):
      # ngram
      for (ngram, count) in vec_hyp[n].items():
        # vrama91 : added clipping
        val[n] += min(vec_hyp[n][ngram], vec_ref[n].get(ngram, 0)) * vec_ref[n].get(ngram, 0)
        
      if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
        val[n] /= (norm_hyp[n]*norm_ref[n])

      assert(not math.isnan(val[n]))
      # vrama91: added a length based gaussian penalty
      val[n] *= self.delta_smooth_tab.get(delta, np.e**(-(delta**2)/(2*self.sigma**2)))
    return val

  def compute_cider(self, vid2refs, vid2tsts, vid_order, cache=False):
    scores = []
    for key in vid_order:
      pred_sents = vid2tsts[key]
      assert len(pred_sents) == 1
      pred_sent = pred_sents[0]
      if pred_sent not in self.sent2rep:
        tst_ngram = cook_test(pred_sent)
        vec, norm, length = self._counts2vec(tst_ngram)
        if cache:
          self.sent2rep[pred_sent] = (vec, norm, length)
      else:
        vec, norm, length = self.sent2rep[pred_sent]

      score = np.array([0.0 for _ in range(self.n)])
      for ref_sent in vid2refs[key]:
        vec_ref, norm_ref, length_ref = self.sent2rep[ref_sent]
        score += self._sim(vec, vec_ref, norm, norm_ref, length, length_ref)
      score_avg = np.mean(score)
      score_avg /= len(vid2refs[key])
      score_avg *= 10.0
      scores.append(score_avg)
    return np.mean(np.array(scores)), np.array(scores)

  def compute_score(self, vid2refs, vid2tsts, vid_order=None, option=None, verbose=0):
    if not self.is_init_refs:
      self.init_refs(vid2refs)
    if vid_order is None:
      vid_order = vid2refs.keys()
    score, scores = self.compute_cider(vid2refs, vid2tsts, vid_order)
    return score, scores
