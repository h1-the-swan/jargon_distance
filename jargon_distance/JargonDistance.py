from collections import defaultdict, Counter
from six import iterkeys, itervalues, iteritems
from numpy import log2
from timeit import default_timer as timer
import logging
logging.basicConfig(format='%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s',
        datefmt="%H:%M:%S",
        level=logging.INFO)
# logger = logging.getLogger(__name__)
logger = logging.getLogger('__main__').getChild(__name__)

# teleportation probability. see "Data and methods" section of Vilhena et al.
ALPHA = 0.01

class JargonDistance(object):

    """Calculate Jargon Distance"""

    def __init__(self, term_counts=None, group_map=None, alpha=ALPHA):
        """

        :term_counts: a dictionary mapping documents to term counts
        :group_map: a dictionary mapping documents to groups (default: put each document in its own group)

        Initialize a JargonDistance instance with a term_counts dict and (optionally) a group_map dict:

        j = JargonDistance(term_counts)

        Then, calculate the jargon distances:

        j.calculate_jargon_distance()
        j.write_to_file('jargon_distance.csv')

        """
        self.term_counts = term_counts  # {document -> {term: Counter}}
        if not isinstance (self.term_counts, dict):
            raise RuntimeError("JargonDistance instance must be initialized with a dictionary of document -> term counter")

        self.group_map = group_map  # {document -> group}
        if self.group_map is None:
            # default behavior: put each document in its own group
            self.group_map = {doc: doc for doc in iterkeys(self.term_counts)}

        self.groups = set(group for group in itervalues(self.group_map))
        self.group_term_prob = defaultdict(dict)  # {group -> {term: prob}}
        self.global_prob = dict()  # {term -> prob}
        self.alpha = alpha  # teleport probability
        self.entropy = dict()  # {group -> entropy}
        self.cross_entropy = defaultdict(dict)  # {group1 -> {group2: cross_entropy}}
        self.jargon_distance = defaultdict(dict)  # {group1 -> {group2: jargon_distance}}

    def count_terms_by_groups(self):
        """From terms by document, count terms by group
        :returns: TODO

        """
        group_term_counts = defaultdict(Counter)  # {group -> {term: count}}
        global_term_counts = Counter()
        for doc, terms in iteritems(self.term_counts):
            this_group = self.group_map[doc]
            for term, term_count in iteritems(terms):
                global_term_counts[term] += term_count
                group_term_counts[this_group][term] += term_count
        return group_term_counts, global_term_counts


    def calculate_probs(self):
        """Get sums and probabilities
        :returns: TODO

        """
        group_term_counts, global_term_counts = self.count_terms_by_groups()
        group_sums = {group: sum(term_counts.values()) for group, term_counts in iteritems(group_term_counts)}
        for group, term_counts in iteritems(group_term_counts):
            for term, term_count in iteritems(term_counts):
                self.group_term_prob[group][term] = float(term_count) / group_sums[group]
        global_count = sum(group_sums.values())
        for term, term_count in iteritems(global_term_counts):
            self.global_prob[term] = float(term_count) / global_count

    def _merge_codebooks(self, p, term, alpha=None):
        """Merge global codebook as explained in Vilhena et al.

        """
        if alpha is None:
            alpha = self.alpha
        return ((1 - alpha)) * p + (alpha * self.global_prob[term])

    def calculate_entropy(self, alpha=None):
        """TODO: Docstring for calculate_entropy.
        :returns: TODO

        """
        if alpha is None:
            alpha = self.alpha
        for group in self.groups:
            if group not in self.entropy:
                self.entropy[group] = 0
            for term, term_prob in iteritems(self.group_term_prob[group]):
                p = self._merge_codebooks(term_prob, term, alpha=alpha)
                self.entropy[group] += p * log2(p)
            # make negative
            self.entropy[group] = -self.entropy[group]

        return self.entropy

    def calculate_cross_entropy(self, alpha=None):
        """TODO: Docstring for calculate_cross_entropy.

        :alpha: TODO
        :returns: TODO

        """
        if alpha is None:
            alpha = self.alpha
        for group1 in self.groups:
            for group2 in self.groups:
                if group2 not in self.cross_entropy[group1]:
                    self.cross_entropy[group1][group2] = 0
                if group1 == group2:
                    self.cross_entropy[group1][group2] = self.entropy[group1]
                else:
                    for term, term_prob_i in iteritems(self.group_term_prob[group1]):
                        p_i = self._merge_codebooks(term_prob_i, term, alpha=alpha)
                        if term in self.group_term_prob[group2]:
                            p_j = self._merge_codebooks(self.group_term_prob[group2][term], term, alpha=alpha)
                        else:
                            p_j = self.global_prob[term]
                            # TODO: IS THIS RIGHT?? I think previous code used alpha times the global prob (but why?)
                        self.cross_entropy[group1][group2] += p_i * log2(p_j)
                    # make negative
                    self.cross_entropy[group1][group2] = - self.cross_entropy[group1][group2]

        return self.cross_entropy

    def _calc_jargon_dist(self, groups, entropy, cross_entropy, alpha=None):
        """TODO: Docstring for _calc_jargon_dist.

        :returns: jargon distance

        """
        if alpha is None:
            alpha = self.alpha

        jargon_distance = defaultdict(dict)

        for group1 in groups:
            for group2 in groups:
                try:
                    jargon_distance[group1][group2] = 1 - ( entropy[group1] / cross_entropy[group1][group2] )
                except ZeroDivisionError:
                    # TODO: is this the right way to handle zero cross-entropy?
                    jargon_distance[group1][group2] = 1
        return jargon_distance

    def calculate_jargon_distance(self, alpha=None):
        """TODO: Docstring for calculate_jargon_distance.
        :returns: TODO

        """

        start = timer()
        logger.debug("Calculating jargon distance...")
        if alpha is None:
            alpha = self.alpha
        if not ( self.group_term_prob and self.global_prob ):
            self.calculate_probs()
        if not self.entropy:
            self.calculate_entropy(alpha=alpha)
        if not self.cross_entropy:
            self.calculate_cross_entropy(alpha=alpha)


        self.jargon_distance = self._calc_jargon_dist(self.groups, self.entropy, self.cross_entropy, alpha=alpha)
        logger.debug("done. Took {:.2f} seconds".format(timer()-start))
        # return self.jargon_distance
        return self

    def write_to_file(self, fname, jargon_distance=None, sep=','):
        """write the jargon distances to a (csv) file
        overwrite if exists

        :fname: filename to write to
        :sep: separator (delimiter). default ','

        """
        logger.debug("Writing to file: {}".format(fname))
        if jargon_distance is None:
            jargon_distance = self.jargon_distance

        with open(fname, 'w') as outf:
            for group1, v in iteritems(jargon_distance):
                for group2, val in iteritems(v):
                    line = sep.join([str(item) for item in [group1, group2, val]])
                    outf.write(line)
                    outf.write('\n')
        
