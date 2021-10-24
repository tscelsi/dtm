import spacy
from spacy.matcher import PhraseMatcher
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc
import numpy as np
import pandas as pd
import pickle
import os
import sys
from collections import Counter, defaultdict
import gensim
import gensim.downloader
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
from auto_labelling import AutoLabel

EUROVOC_PATH = os.path.join(os.environ['ROADMAP_SCRAPER'], "eurovoc_export_en.csv")
EUROVOC_LABEL_LOGODDS_PATH = os.path.join(os.environ['ROADMAP_SCRAPER'], "eurovoc_label_logodds.txt")
EUROVOC_ALL_EMBEDDINGS_DIR = os.path.join(os.environ['ROADMAP_SCRAPER'], "eurovoc_embeddings")
WHITELIST_EUROVOC_LABELS_PATH = os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "dtm", "eurovoc_labels_merged.txt")

class Eurovoc:

    eurovoc_label_correction_map = {
        "6626 soft energy": "6626 renewable energy",
        "6616 oil industry": "6616 oil and gas industry"
    }

    # default remapping, can be overridden in __init__
    eurovoc_label_remapping = {
        "1621 economic structure": "1611 economic conditions",
        "2006 trade policy": "2016 business operations and trade",
        "2421 free movement of capital": "2016 business operations and trade",
        "2016 trade": "2016 business operations and trade",
        "4006 business organisation": "2016 business operations and trade",
        "4016 legal form of organisations" : "2016 business operations and trade",
        "2426 financing and investment": "2016 business operations and trade",
        "2026 consumption": "2016 business operations and trade",
    }

    def __init__(self, eurovoc_path=None, eurovoc_whitelist=False, whitelist_eurovoc_labels=None):
        self.nlp = spacy.load("en_core_web_sm")
        print("Initialising EuroVoc...")
        def preproc(label):
            lowered_label = label.lower()
            if lowered_label in self.eurovoc_label_remapping:
                lowered_label = self.eurovoc_label_remapping[lowered_label]
            if lowered_label in self.eurovoc_label_correction_map:
                lowered_label = self.eurovoc_label_correction_map[lowered_label]
            return lowered_label
        self.eurovoc = pd.read_csv(EUROVOC_PATH) if not isinstance(eurovoc_path, str) else pd.read_csv(eurovoc_path)
        self.eurovoc['MT'] = self.eurovoc['MT'].apply(preproc)
        if eurovoc_whitelist:
            self.whitelist_eurovoc_labels = whitelist_eurovoc_labels if whitelist_eurovoc_labels != None else [x.strip().lower() for x in open(WHITELIST_EUROVOC_LABELS_PATH, "r").readlines()]
            m = self.eurovoc.apply(lambda x: x.MT.lower() in self.whitelist_eurovoc_labels, axis=1)
            self.eurovoc = self.eurovoc[m]
            assert len(self.eurovoc['MT'].drop_duplicates()) == len(self.whitelist_eurovoc_labels)
        self.eurovoc['index'] = [i for i in range(len(self.eurovoc))]
        self.eurovoc = self.eurovoc.set_index('index')
        # self._create_eurovoc_label_term_map()
    
    def _init_embeddings(self):
        """This function is really only used for merging eurovoc labels together that are similar.
        see the merge_labels function below.

        We use the auto labelling embedding framework to save on reproducing the same code.
        """
        al = AutoLabel(self.eurovoc, phrase_col="TERMS (PT-NPT)", label_col="MT")
        al._init_embeddings(False, False)
        self.phrase_embeddings = al.phrase_embeddings
        self.eurovoc_topics = al.sorted_labels


def find_sim_pairs(mat, eurovoc_labels, threshold):
    """This function finds eurovoc label embeddings that are 
    similar to each other.

    Args:
        mat (np.array): pairwise similarity between all eurovoc labels
        eurovoc_labels (np.array): array containing eurovoc label names
        threshold (float): similarity threshold over which two labels are considered similar

    Returns:
        [tuple]: list of tuples returned of similar pairs of eurovoc labels
    """
    assert mat.shape[0] == len(eurovoc_labels)
    sim_pairs = []
    for i in range(mat.shape[0]):
        for j in range(mat.shape[0]):
            if j > i:
                sim = mat[i][j]
                if sim >= threshold:
                    sim_pairs.append((eurovoc_labels[i],eurovoc_labels[j]))
    return sim_pairs

def create_merge_plan(pairs):
    # count number of times topic is in pairs
    freqs = []
    lookup = defaultdict(lambda: None)
    merges = []
    pairs = sorted(pairs, key=lambda x: x[0])
    for l1, l2 in pairs:
        freqs.extend([l1, l2])
        if lookup[l1] == None and lookup[l2] == None:
            # neither of these labels have been merged anywhere yet
            merges.append(set([l1, l2]))
            lookup[l1] = len(merges) - 1
            lookup[l2] = len(merges) - 1
        elif (lookup[l1] == None and lookup[l2] != None) or (lookup[l1] != None and lookup[l2] == None):
            # one of the labels has been merged with something else
            index = lookup[l1] if lookup[l1] != None else lookup[l2]
            merges[index].add(l1)
            merges[index].add(l2)
            lookup[l1] = index
            lookup[l2] = index
        else:
            # both of the labels exist elsewhere, we need to merge two sets
            l1_index = lookup[l1]
            l2_index = lookup[l2]
            l1_set = merges[l1_index]
            l2_set = merges[l2_index]
            merged_set = l1_set.union(l2_set)
            # we'll append this new merged set to end of list and update lookups
            merges.append(merged_set)
            for l in merged_set:
                lookup[l] = len(merges) - 1
    c = Counter(freqs)
    merge_groups = {}
    merge_mapping = {}
    merge_plans_to_take = set(lookup.values())
    merges = [merges[i] for i in merge_plans_to_take]
    for merge_action in merges:
        most_common = sorted([(label, c[label]) for label in merge_action], key=lambda x: x[1], reverse=True)[0][0]
        merge_groups[most_common] = list(merge_action)
        for label in merge_action:
            merge_mapping[label] = most_common
    return merge_mapping, merge_groups

def create_label_vectors(embeddings):
    label_vecs = []
    for i, label in enumerate(embeddings):
        # pairwise cosine sim between top words and topic term vectors
        label_mat = np.array(label)
        label_vec = label_mat.mean(axis=0)
        label_vecs.append(label_vec)
    return label_vecs

def merge_labels(max_iterations, threshold=0.95):
    e = Eurovoc(eurovoc_whitelist=True)
    e._init_embeddings()
    steps = []
    for i in range(max_iterations):
        label_vecs = create_label_vectors(e.phrase_embeddings)
        sim_matrix = cosine_similarity(label_vecs, label_vecs)
        pairs = find_sim_pairs(sim_matrix, e.eurovoc_topics, threshold)
        merge_mapping, merge_groups = create_merge_plan(pairs)
        if merge_groups == {}:
            break
        steps.append(merge_groups)
        e.eurovoc['MT'] = e.eurovoc['MT'].apply(lambda x: merge_mapping[x] if x in merge_mapping else x)
        e._init_embeddings()
        # e.eurovoc_topics = sorted(np.array(e.eurovoc['MT'].drop_duplicates()))
    return steps, e.eurovoc, e.eurovoc_topics

def merge_eurovoc_df_with_steps(steps):
    """Given a set of iterative steps, perform the merges to align the labels
    """
    df = pd.read_csv("eurovoc_export_en.csv")
    for step in steps:
        for group_label in step.keys():
            mini_map = {x: group_label for x in step[group_label]}
            df['MT'] = df['MT'].apply(lambda x: mini_map[x] if x in mini_map else x)
    return df

def create_steps_by_label_category():
    """Merge all label categories except 66
    """
    label_category_names = [
        "0406 political framework",
        "0821 defence",
        "1011 European Union law",
        "1221 justice",
        "1606 economic policy",
        "2016 trade",
        "2426 financing and investment",
        "2841 health",
        "3206 education",
        "3606 natural and applied sciences",
        "4006 business organisation",
        "4406 employment",
        "4811 organisation of transport",
        "5206 environmental policy",
        "5606 agricultural policy",
        "6026 foodstuff",
        "6406 production",
        "6606 energy policy",
        "6621 electrical and nuclear industries",
        "6616 oil industry",
        "6626 soft energy",
        "6611 coal and mining industries",
        "6806 industrial structures and policy",
        "7231 economic geography",
        "7621 world organisations"
    ]
    
    df = pd.read_csv("eurovoc_export_en.csv")
    df = df.drop_duplicates(subset=['MT'])
    df['label_category'] = df['MT'].apply(lambda x: int(re.match(r'(\d{2}).*', x).group(1)))
    labels = df.loc[:, ['MT', 'label_category']]
    label_list = labels.groupby('label_category')['MT'].apply(list).tolist()
    steps = {}
    for i,label_group in enumerate(label_list):
        if label_group[0].startswith("66"):
            continue
        for label in label_category_names:
            if label in label_group:
                steps[label] = label_group
    return [steps]




if __name__ == "__main__":
    # steps = create_steps_by_label_category()
    # df = merge_eurovoc_df_with_steps(steps)
    # df.to_csv("eurovoc_export_en_merged.csv")
    # breakpoint()
    steps, eurovoc_df, eurovoc_topics = merge_labels(4, threshold=0.95)
    json.dump(steps, open(os.path.join(os.environ['ROADMAP_SCRAPER'], "eurovoc_auto_labelling", "steps_stage3.json"), "w+"))
    json.dump(eurovoc_topics, open(os.path.join(os.environ['ROADMAP_SCRAPER'], "eurovoc_auto_labelling", "labels_stage3.json"), "w+"))
    # breakpoint()
    print("hey")
