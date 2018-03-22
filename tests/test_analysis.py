# -*- coding: utf-8 -*-

from .context import jargon_distance
from text_data import fnames, groups

from jargon_distance import JargonDistance
from jargon_distance.analysis import JargonDistanceAnalysis

from nltk import word_tokenize, ngrams
from collections import Counter
import codecs

import unittest


class AnalysisTests(unittest.TestCase):
    """Test cases for analysis of jargon distance output."""

    def setUp(self):
        term_counts = []
        for fname in fnames:
            with codecs.open(fname, 'r', encoding='utf8') as f:
                txt = f.read()
            txt = txt.lower()
            tokens = word_tokenize(txt)
            ng = ngrams(tokens, 1)
            term_counts.append(Counter(ng))
        term_count_dict = {}
        group_map = {}
        for i, fname in enumerate(fnames):
            term_count_dict[fname] = term_counts[i]
            group_map[fname] = groups[i]
        self.j = JargonDistance(term_counts=term_count_dict, group_map=group_map)
        self.j.calculate_jargon_distance()
        
        self.ja = JargonDistanceAnalysis().from_object(self.j)
        self.ja.symmetrize_graph()

    def test_number_of_nodes(self):
        self.assertEqual(len(self.ja.G_sym.nodes()), len(fnames))


if __name__ == '__main__':
    unittest.main()
