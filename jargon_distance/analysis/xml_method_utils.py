import sys, os, time, re
from datetime import datetime
from timeit import default_timer as timer

try:
    import cPickle as pickle
except ImportError:
    import pickle

from glob import glob
from collections import Counter

# BeautifulSoup for parsing XML
from bs4 import BeautifulSoup

import logging
logging.basicConfig(format='%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s',
        datefmt="%H:%M:%S",
        level=logging.INFO)
# logger = logging.getLogger(__name__)
logger = logging.getLogger('__main__').getChild(__name__)

XML_DIRNAME = '/data/jason/equations/xml_conversion_20170128/'

PID_PATTERN = re.compile(r"(.*?\D)(\d+)")
URL_ID_PATTERN = re.compile(r"abs/(.+\d?)v")

XML_PATHS_FNAME = 'paths_to_xml_files.txt'

LABELMAP_FILE = 'arxiv-categories.csv'  # maps abbreviated category names (e.g., "nuc-th") to long names
# the third column in the labelmap file is the broad_category name

def get_pid_from_fname(fname, p=PID_PATTERN):
    """get arXiv paper id from a filename

    :fname: filename or file path
    :returns: pid

    """
    tail = os.path.split(fname)[1]
    m = p.search(tail)
    pid_parts = m.groups()
    joinchar = "" if pid_parts[0][-1] == "." else "/"
    return joinchar.join(pid_parts)

def get_pid_from_full_path(fpath, id_loc=6):
    path_split = fpath.split('/')
    pid = path_split[id_loc]
    return get_pid_from_fname(pid)

def get_pid_from_url(url_id, p=URL_ID_PATTERN):
    """get arXiv paper id 

    :url_id: TODO
    :returns: TODO

    """
    m = p.search(url_id)
    return m.group(1)


def get_maths(xml_fname):
    """use BeautifulSoup to get 'Maths' tags from XML file

    :returns: maths: BeautifulSoup object

    """
    with open(xml_fname, 'r') as f:
        soup = BeautifulSoup(f.read(), 'xml')
    maths = soup.find_all('Math')
    return maths

def get_toksdicts(maths):
    """from the BeautifulSoup Math (see get_maths()), get
    a list of dictionaries for each 'XMTok'
    with the attributes and the text

    :maths: list of Math objects from BeautifulSoup
    :returns: list of dictionaries corresponding to the XMTok attrs, 
            plus the key value pair: "text" -> text of the XMTok

    """
    toksdicts = []
    for m in maths:
        for xmtok in m.find_all('XMTok'):
            d = xmtok.attrs
            d['text'] = xmtok.text
            toksdicts.append(d)
    return toksdicts

def get_math_tok_counter(toksdicts):
    """get a counter of tokens from the toksdicts

    :toksdicts: from get_toksdicts()
    :returns: Counter

    """
    c = Counter()
    for t in toksdicts:
        c[t.get('text', '')] += 1
    return c

def get_counter_from_xml(xml_fname):
    """get a counter of math tokens from an xml file

    :xml_fname: TODO
    :returns: Counter

    """
    maths = get_maths(xml_fname)
    toksdicts = get_toksdicts(maths)
    c = get_math_tok_counter(toksdicts)
    return c

def deal_with_multiple_abstracts(abstracts):
    abstracts_nonempty = []
    for a in abstracts:
        if a.text.strip():
            abstracts_nonempty.append(a)
    num_abstracts = len(abstracts_nonempty)
    if num_abstracts == 0:
        return None
    elif num_abstracts == 1:
        return abstracts_nonempty[0].text
    else:
        # there are multiple abstracts
        # I have seen cases in which there are multiple abstracts with different languages
        # we can try to identify the english one
        try:
            import langdetect
            from langdetect.lang_detect_exception import LangDetectException
        except ImportError:
            logger.debug("failed to import langdetect")
            return None
        abstracts_english = []
        for a in abstracts_nonempty:
            try:
                if langdetect.detect(a.text) == 'en':
                    abstracts_english.append(a)
            except LangDetectException as e:
                logger.debug("LangDetectException: {}".format(e))
                return None
        if len(abstracts_english) == 1:
            return abstracts_english[0].text
    return None

def get_abstract_from_xml(xml_fname):
    with open(xml_fname, 'r') as f:
        soup = BeautifulSoup(f.read(), 'xml')
    abstracts = soup.find_all('abstract')
    num_abstracts = len(abstracts)
    if num_abstracts == 0:
        return None
    elif num_abstracts > 1:
        extracted_abstract = deal_with_multiple_abstracts(abstracts)
    elif num_abstracts == 1:
        extracted_abstract = abstracts[0].text
    return extracted_abstract

def preprocess_abstract(abstract):
    import regex as re
    punc_pattern = r"\p{P}+"
    if sys.version_info[0] < 3:
        # punc_pattern = ur"\p{P}+"
        punc_pattern = unicode(punc_pattern)
        # NOTE: I'M NOT TOTALLY SURE THIS WILL WORK. BE CAREFUL. see stackoverflow link below
    abstract = abstract.strip().lower()
    # remove punctuation
    # http://stackoverflow.com/questions/11066400/remove-punctuation-from-unicode-formatted-strings/11066687#11066687
    abstract = re.sub(punc_pattern, "", abstract)
    return abstract

def get_ngrams_counter_from_abstract(abstract, n=1):
    if not abstract:
        return None
    from nltk import word_tokenize, ngrams
    abstract = preprocess_abstract(abstract)
    tokens = word_tokenize(abstract)
    ng = ngrams(tokens, n)
    return Counter(ng)

def get_ngrams_counter_from_xml(xml_fname, n=1):
    abstract = get_abstract_from_xml(xml_fname)
    return get_ngrams_counter_from_abstract(abstract, n=n)

def load_all_pickles(dirname='data', ext='pickle'):
    """load all of the pickle files (with math token counters)

    :dirname: name of directory with the pickle files
    :ext: extension of pickle files (default: 'pickle')
    :returns: big dictionary of {'fname': Counter}

    """
    g = glob(os.path.join(dirname, "*.{}".format(ext)))
    d = {}
    for fname in g:
        with open(fname, 'rb') as f:
            d.update(pickle.load(f))
    return d

def load_xml_paths(xml_paths_fname=XML_PATHS_FNAME):
    # get a list of all the paths to xml files from a cached text file
    xml_paths = []
    with open(xml_paths_fname, 'r') as f:
        for line in f:
            line = line.strip()
            xml_paths.append(line)
    return xml_paths

def get_group_map(fname='data_v2/categories.csv', sep=',', header=False):
    """get a dictionary of {paper id -> category name}

    :returns: dict

    """
    group_map = {}
    with open(fname, 'r') as f:
        for i, line in enumerate(f):
            if i == 0 and header is True:
                continue
            line = line.strip().split(sep)
            pid = line[0]
            if pid.startswith('http'):
                pid = get_pid_from_url(pid)
            group = line[1]
            group_map[pid] = group
    return group_map

def get_broad_labels(labelmap_file=LABELMAP_FILE, header=True, sep=','):
    broad_labels = {}
    with open(labelmap_file, 'r') as f:
        for i, line in enumerate(f):
            if ( header ) and ( i == 0 ):
                continue
            line = line.strip().split(sep)
            if line:
                short_label = line[0]
                broad_label = line[2]
                broad_labels[short_label] = broad_label
    return broad_labels

def get_group_map_broad(group_map, broad_labels):
    group_map_broad = {}
    for pid, group in group_map.iteritems():
        try:
            group_map_broad[pid] = broad_labels[group]
        except KeyError:
            group_map_broad[pid] = group
    return group_map_broad
    

def main(args):
    pass

if __name__ == "__main__":
    total_start = timer()
    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    parser = argparse.ArgumentParser(description="")
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
    logger.info('all finished. total time: {:.2f} seconds'.format(total_end-total_start))
