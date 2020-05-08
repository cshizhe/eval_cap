# Filename: cider.py
#
# Description: Describes the class to compute the CIDEr (Consensus-Based Image Description Evaluation) Metric 
#               by Vedantam, Zitnick, and Parikh (http://arxiv.org/abs/1411.5726)
#
# Creation Date: Sun Feb  8 14:16:54 2015
#
# Authors: Ramakrishna Vedantam <vrama91@vt.edu> and Tsung-Yi Lin <tl483@cornell.edu>

from cider_scorer_fast import CiderScorer
import pdb
import numpy as np

class Cider:
  """
  Main Class to compute the CIDEr metric 

  """
  def __init__(self, test=None, refs=None, n=4, sigma=6.0, 
    idf_path=None):
    # set cider to sum over 1 to 4-grams
    self._n = n
    # set the standard deviation parameter for gaussian penalty
    self._sigma = sigma
    # set the init document_frequency (defaultdict) and ref_len (int)
    if idf_path is not None:
      data = np.load(idf_path)
      self.document_frequency = data['document_frequency'].item()
      self.ndocs = data['ndocs']
    else:
      self.document_frequency = self.ndocs = None

  def precompute_document_frequency(self, gts, idf_path):
    from cider_scorer import cook_refs
    import numpy as np

    cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)
    crefs = []
    for refs in gts.itervalues():
      # Sanity check.
      assert(type(refs) is list)
      assert(len(refs) > 0)
      crefs.append(cook_refs(refs))
    document_frequency, ndocs = cider_scorer.compute_doc_freq(crefs)
    
    print(len(document_frequency), ndocs)
    with open(idf_path, 'wb') as f:
      np.savez(f, document_frequency=document_frequency, ndocs=ndocs)


  def compute_score(self, gts, res):
    """
    Main function to compute CIDEr score
    :param  hypo_for_image (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
        ref_for_image (dict)  : dictionary with key <image> and value <tokenized reference sentence>
    :return: cider (float) : computed CIDEr score for the corpus 
    """

    # assert(gts.keys() == res.keys())
    imgIds = gts.keys()

    cider_scorer = CiderScorer(n=self._n, sigma=self._sigma,
      document_frequency=self.document_frequency,
      ndocs=self.ndocs)

    for id in imgIds:
      hypo = res[id]
      ref = gts[id]

      # Sanity check.
      assert(type(hypo) is list)
      assert(len(hypo) == 1)
      assert(type(ref) is list)
      assert(len(ref) > 0)

    (score, scores) = cider_scorer.compute_score(gts, res)

    return score, scores

  def method(self):
    return "CIDEr"


if __name__ == '__main__':
  import os
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--human_caption_file')
  parser.add_argument('--split_dir')
  parser.add_argument('--idf_path')
  opts = parser.parse_args()

  captions = np.load(opts.human_caption_file)
  trnids = np.load(os.path.join(opts.split_dir, 'trn.npy'))
  trngts = {}
  for id in trnids:
    trngts[id] = captions[id]
  cider_scorer = Cider()
  cider_scorer.precompute_document_frequency(trngts, opts.idf_path)
