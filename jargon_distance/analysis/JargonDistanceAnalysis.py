import sys, os, time
from six import iterkeys, itervalues, iteritems
from datetime import datetime
from timeit import default_timer as timer
try:
    from humanfriendly import format_timespan
except ImportError:
    def format_timespan(seconds):
        return "{:.2f} seconds".format(seconds)

import logging
logging.basicConfig(format='%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s',
        datefmt="%H:%M:%S",
        level=logging.INFO)
# logger = logging.getLogger(__name__)
logger = logging.getLogger('__main__').getChild(__name__)

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # use this backend to save figures via a script
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
sns.set_context('paper')

from scipy import version
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform, is_valid_y

# from .xml_method_utils import get_group_map

# CATEGORIES_FILE = 'data_v2/categories.csv'  # maps documents to categories
# LABELMAP_FILE = 'arxiv-categories.csv'  # maps abbreviated category names (e.g., "nuc-th") to long names

class JargonDistanceAnalysis(object):

    """Analysis and dendrogram creation for jargon distance"""

    def __init__(self, 
                    jargondistance_fname=None,
                    group_map=None,
                    labelmap=None,
                    labelmap_file=None,
                    threshold=None):
        """TODO: to be defined1.

        :jargondistance_fname: TODO

        """
        self.jargondistance_fname = jargondistance_fname
        self.group_map = group_map
        self.labelmap = labelmap
        if not self.labelmap:
            self.labelmap_file = labelmap_file
        self.threshold = threshold

        self.G = None
        self.G_sym = None
        self.Z = None

        if self.jargondistance_fname:
            self.load_jargondistance_file(self.jargondistance_fname)


    def process_jargondistance_file(self, fname, sep=',', header=False):
        with open(fname, 'r') as f:
            for i, line in enumerate(f):
                if header is True and i == 0:
                    continue
                # line = line.strip().split(sep)
                # line = " ".join(line)  # this is what networkx likes
                yield line

    def get_networkx_graph_from_jargondistance(self, edge_list, sep=','):
        """Get an unprocessed, non-symmetrized networkx DiGraph (directed graph)
        from a jargon distance adjacency (typically csv) file.
        Each row in the input file represents the jargon distance from
        the first category to the second (cat1, cat2, distance).

        :jargondistance_fname: filename (typically csv)
        :sep: delimiter for the input. default: comma
        :returns: networkx DiGraph

        """
        G = nx.DiGraph()
        G = nx.parse_edgelist(edge_list, delimiter=sep, create_using=G, nodetype=str, data=(('weight',float),))
        self.G = G
        return G

    def load_jargondistance_file(self, fname, sep=','):
        """load jargondistance file

        :fname: csv filename

        """
        self.jargondistance_fname = fname
        edge_list = [line for line in self.process_jargondistance_file(fname, sep=sep)]
        G = self.get_networkx_graph_from_jargondistance(edge_list, sep=sep)

        if self.threshold:
            G = self.remove_small_categories(G, thresh=self.threshold)
        return G

    @classmethod
    def from_file(cls, fname, sep=','):
        ja = cls()
        G = ja.load_jargondistance_file(fname, sep=sep)
        return ja

    @classmethod
    def from_object(cls, j):
        """load jargon distances from a JargonDistance object
        (instead of loading from a file)

        :j: JargonDistance object that has had the jargondistances calculated

        """
        G = nx.DiGraph()
        for group1, v in iteritems(j.jargon_distance):
            for group2, val in iteritems(v):
                G.add_edge(group1, group2, weight=val)
        ja = cls()
        ja.G = G
        ja.group_map = j.group_map
        return ja


    def remove_small_categories(self, G=None, 
                                categories_file=None, 
                                thresh=None, 
                                return_remove=False):
        """Given a networkx graphs with nodes representing categories, 
        remove the categories with below <thresh> number of papers

        :G: networkx graph
        :categories_file: file that maps papers to categories
        :thresh: will remove the category if the number of papers <= this threshold
        :return_remove: if True, also return the list of removed categories
        :returns: networkx graph with nodes removed

        """
        if G is None:
            G = self.G

        if thresh is None:
            if self.threshold:
                thresh = self.threshold
            else:
                thresh = 100
        self.threshold = thresh

        # group_map = get_group_map(categories_file)
        # cats_df = pd.DataFrame.from_dict(group_map, orient='index').rename(columns={0: 'category'})
        if categories_file:
            cats_df = pd.read_csv(categories_file, sep='\t')
            vc = cats_df.category.value_counts().sort_values()
            vc = vc[vc<=thresh]
            remove = vc.index.tolist()
            G.remove_nodes_from(remove)
            if return_remove:
                return G, remove
            return G
        elif self.group_map:
            vc = pd.Series(self.group_map).value_counts()
            vc = vc[vc<=thresh]
            remove = vc.index.tolist()
            G.remove_nodes_from(remove)
            if return_remove:
                return G, remove
            return G
        else:
            raise RuntimeError("need to specify either a categories_file or a JargonDistanceAnalysis.group_map")


    def symmetrize_graph(self, G=None):
        """Create a symmetrized version of the networkx graph G
        by taking the average of the edge weights.

        :G: networkx graph
        :returns: G_sym: symmetrized graph (undirected graph)

        """
        if G is None:
            G = self.G
        G_sym = nx.Graph()
        # for u, v in G.edges_iter():
        for u, v in G.edges():
            if not G_sym.has_edge(u, v):
                weight_uv = G[u][v]['weight']
                weight_vu = G[v][u]['weight']
                weight_sym = ( weight_uv + weight_vu ) / 2
                # G_sym.add_edge(u, v, {'weight': weight_sym})
                G_sym.add_edge(u, v, weight=weight_sym)
        self.G_sym = G_sym

    def get_labelmap(self, labelmap_fname=None, sep=',', 
                        header=True, reverse=False,
                        get_broad=True):
        """Get a dictionary mapping abbreviated category names to the full name

        :labelmap_fname: csv file containing the mapping
        :reverse: if True, return the reverse mapping---long category name to abbreviated name
        :get_broad: if True, also return a mapping of short labels to broad labels (e.g. 'math.LO' -> 'Mathematics')
        :returns: dictionary

        """
        if not labelmap_fname:
            labelmap_fname = self.labelmap_file

        labelmap = {}
        broad_labelmap = {}
        with open(labelmap_fname, 'r') as f:
            for i, line in enumerate(f):
                if header is True and i == 0:
                    continue
                line = line.strip().split(sep)
                cat_short = line[0]
                cat_long = line[1]
                if reverse is True:
                    labelmap[cat_long] = cat_short
                else:
                    labelmap[cat_short] = cat_long

                if get_broad:
                    cat_broad = line[2]
                    broad_labelmap[cat_short] = cat_broad

        self.labelmap = labelmap
        self.broad_labelmap = broad_labelmap

        if get_broad:
            return labelmap, broad_labelmap
        return labelmap

    def get_long_labels(self, short_labels=None, labelmap=None):
        G = self.G_sym or self.G
        if labelmap in ['broad', 'broad_labels', 'use_broad', 'use_broad_labels']:
            labelmap = self.broad_labelmap
        if not labelmap:
            labelmap = self.labelmap
            if not labelmap:
                # labelmap, broad_labelmap = self.get_labelmap()
                labelmap = self.get_labelmap(sep='\t', get_broad=False)
        if not short_labels:
            short_labels = G.nodes()
        labels = []
        for lab in short_labels:
            try:
                this_label = labelmap[lab]
            except KeyError:
                this_label = lab
            labels.append(this_label)
        return labels

    def get_linkage(self, d=None, method='average', skip_condensed=False):
        if d is None:
            d = self.G_sym or self.G

        # convert to numpy matrix if it isn't already
        if isinstance(d, nx.Graph):
            d = nx.to_numpy_matrix(d)
        if (skip_condensed is False) and (not is_valid_y(d)):
            d = squareform(d)
        Z = linkage(d, method=method)
        self.Z = Z
        return Z

    def make_dendrogram(self,
                        Z=None,
                        labels=None,
                        figsize=(25,20),
                        rotation=45.,
                        x_font_size=16.,
                        y_lab='Jargon Distance',
                        show_plot=False,
                        save=None,
                        save_dpi=300,
                        nonzero_diagonal=False):
        """Make a dendrogram from a scipy linkage

        :Z: scipy linkage (should be a condensed distance matrix)
          if Z is None, the linkage matrix will be calculated from the graph
        :labels: list of labels (for the x-axis)
        :rotation: degree of rotation for the x-axis labels
        :x_font_size: font size for the x-axis labels
        :show_plot: if true, call plt.show()
        :save: save to this filename (.png)
        :nonzero_diagonal: if True and Z is None, calculate and use a non-condensed distance matrix
        :returns: figure, dendrogram object

        """
        # plt.figure(figsize=(25, 20))
        # plt.ylabel(y_lab, fontsize=20)
        # plt.yticks(fontsize=16)
        Z = Z if Z is not None else self.Z
        if Z is None:
            G = self.G_sym or self.G
            if nonzero_diagonal is True:
                Z = self.get_linkage(G, skip_condensed=True)
            else:
                Z = self.get_linkage(G)
        if labels is None:
            labels = self.get_long_labels()
        fig, ax = plt.subplots(figsize=figsize)
        den = dendrogram(
            Z,
    #         p=4,
    #         truncate_mode='lastp',
            leaf_rotation=rotation,  # rotates the x axis labels
            leaf_font_size=x_font_size,  # font size for the x axis labels
            labels=labels,
            ax=ax
        )
        ax.set_ylabel(y_lab, fontsize=20)
        ax.tick_params(axis='y', labelsize=16)
        if save:
            plt.gcf().tight_layout()
            plt.savefig(save, dpi=save_dpi)
        # cf = plt.gcf()
        if show_plot is True:
            plt.show()
        else:
            return fig, den

    def make_sns_clustermap(self, distance_matrix=None, labels=None, Z=None, figsize=None, metric_label="Distance", show_plot=False, save=None, save_dpi=300, nonzero_diagonal=False):
        """make a clustermap (heatmap with dendrograms) using seaborn

        :distance_matrix: dataframe
        :labels: TODO
        :Z: TODO
        :save: TODO
        :returns: TODO

        """
        G = self.G_sym or self.G
        Z = Z if Z is not None else self.Z
        if Z is None:
            if nonzero_diagonal is True:
                Z = self.get_linkage(G, skip_condensed=True)
            else:
                Z = self.get_linkage(G)
        if labels is None:
            labels = self.get_long_labels()
        if distance_matrix is None:
            arr = nx.to_numpy_matrix(G)
            distance_matrix = pd.DataFrame(arr, index=labels, columns=labels)
        # fig, ax = plt.subplots(figsize=figsize)
        # cm = sns.clustermap(distance_matrix, row_linkage=Z, col_linkage=Z, cbar_kws={'label': metric_label}, ax=ax)
        cm = sns.clustermap(distance_matrix, row_linkage=Z, col_linkage=Z, figsize=figsize, cbar_kws={'label': metric_label})
        for item in cm.ax_heatmap.get_yticklabels():
            item.set_rotation(0)
        if save:
            # plt.gcf().tight_layout()
            plt.gcf().subplots_adjust(bottom=.3, right=.7)
            plt.savefig(save, dpi=save_dpi)
        # cf = plt.gcf()
        if show_plot is True:
            plt.show()
        # return cm

    def prepare_clustergrammer_data(self, outfname='clustergrammer_data.json', G=None):
        """for a distance matrix, output a clustergrammer JSON file
        that clustergrammer-js can use

        for now it loads the clustergrammer-py module from local dev files
        TODO: once changes are pulled into clustergrammer-py, we can use the actual module (pip)

        :outfname: filename for the output json
        :G: networkx graph (use self.G_sym by default)

        """
        G = self.G_sym or self.G
        # if Z is None:
        #     G = self.G_sym or self.G
        #     Z = self.get_linkage(G)
        clustergrammer_py_dev_dir = '../clustergrammer/clustergrammer-py/'
        sys.path.insert(0, clustergrammer_py_dev_dir)
        from clustergrammer import Network as ClustergrammerNetwork
        start = timer()
        d = nx.to_numpy_matrix(G)
        df = pd.DataFrame(d, index=G.nodes(), columns=G.nodes())
        net = ClustergrammerNetwork()
        # net.load_file(infname)
        # net.load_file(mat)
        net.load_df(df)
        net.cluster(dist_type='precalculated')
        logger.debug("done loading and clustering. took {}".format(format_timespan(timer()-start)))

        logger.debug("writing to {}".format(outfname))
        start = timer()
        net.write_json_to_file('viz', outfname)
        logger.debug("done writing file {}. took {}".format(outfname, format_timespan(timer()-start)))
        
def test():
    fname = 'jargondistance_20171122052755.csv'
    logger.debug("loading {}".format(fname))
    ja = JargonDistanceAnalysis()
    G = ja.load_jargondistance_file(fname)
    # ja = JargonDistanceAnalysis(fname)
    # G = ja.G
    logger.debug("num nodes: {}".format(G.number_of_nodes()))
    
    thresh=200
    logger.debug("remove categories with <= {}".format(thresh))
    G, remove = ja.remove_small_categories(thresh=thresh, return_remove=True)
    logger.debug("{} categories removed".format(len(remove)))
    logger.debug("num nodes: {}".format(G.number_of_nodes()))

    logger.debug("creating symmetrized version")
    # G_sym = ja.symmetrize_graph()
    # logger.debug("num nodes: {}".format(G_sym.number_of_nodes()))
    ja.symmetrize_graph()
    logger.debug("num nodes: {}".format(ja.G_sym.number_of_nodes()))

    # labels = ja.get_long_labels()
    # Z = ja.get_linkage()
    outfname = 'test_den.png'
    logger.debug('saving dendrogram to {}'.format(outfname))
    # ja.make_dendrogram(Z, labels=labels, rotation=90., save=outfname)
    ja.make_dendrogram(rotation=90., save=outfname)

    clustergrammer_outfname = 'test_clustergrammer_data.json'
    logger.debug('saving clustergrammer json to {}'.format(clustergrammer_outfname))
    ja.prepare_clustergrammer_data(outfname=clustergrammer_outfname)

def main(args):
    test()

if __name__ == "__main__":
    total_start = timer()
    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description="JargonDistanceAnalysis object. should not normally be run as a main script")
    parser.add_argument("--debug", action='store_true', help="output debugging info")
    global args
    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug('debug mode is on')
    else:
        logger.setLevel(logging.INFO)
    main(args)
    total_end = timer()
    logger.info('all finished. total time: {}'.format(format_timespan(total_end-total_start)))
