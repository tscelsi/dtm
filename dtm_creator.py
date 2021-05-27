#! env/bin/python3
# Thomas Scelsi
# Full set of dynamic topic modelling tools for a general dataset

import pandas as pd
import os
import spacy
from numpy import random
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.matcher import PhraseMatcher
from collections import defaultdict, Counter
import json
from multiprocessing import Pool

data_path = os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "energy_policy_applied_energy__coal_journals.csv")
bigram_path = os.path.join(os.environ['ROADMAP_SCRAPER'], "BIGRAMS.txt")

class DTMCreator:
    def __init__(self, model_root, csv_path, text_col_name='section_txt', date_col_name='date', bigram=True, seed=42, limit=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df.dropna(subset=[text_col_name])
        self.paragraphs = self.df[text_col_name].tolist()
        self.dates = self._extract_dates(date_col_name)
        self.bigram = bigram
        # create directory structure
        if not os.path.isdir(model_root):
            os.mkdir(model_root)
            # os.mkdir(os.path.join(model_root, "model_run"))
        self.model_root = model_root
        # spacy load
        self.nlp = spacy.load("en_core_web_sm")
        # self.nlp.remove_pipe("tagger")
        self.nlp.remove_pipe("parser")
        self.nlp.remove_pipe("ner")
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
        self.rdocs =[]
        self.rdates = []
        if limit:
            for idx in random.RandomState(seed).permutation(len(self.paragraphs))[:limit]:
                try:
                    self.rdocs.append(self.nlp(self.paragraphs[idx]))
                    self.rdates.append(self.dates[idx])
                except:
                    print(idx)
        else:
            for idx in random.RandomState(seed).permutation(len(self.paragraphs)):
                try:
                    self.rdocs.append(self.nlp(self.paragraphs[idx]))
                    self.rdates.append(self.dates[idx])
                except:
                    print(idx)
        return

    def _extract_dates(self, date_col_name):
        """Here we extract the dates for each document by taking a particular column of the dataframe containing the corpus
        and extracting the year from a date stamp. This will need to be overidden depending on the format of your date and the column name. 
        In this case, the df has a column labelled 'date' which contains a date of form 2020-10-09. 
        We extract the year (2020) for each doc in the df.

        MAKE GENERAL
        """
        return [int(d.split("-")[0]) for d in self.df[date_col_name].tolist()]

    def _add_bigrams(self):
        bigrams = [x.strip("\n") for x in open(bigram_path, "r").readlines()]
        bigram_matcher = PhraseMatcher(self.nlp.vocab, attr="LEMMA")
        for bigram in self.nlp.pipe(bigrams, n_process=11, batch_size=256):
            bigram_matcher.add(bigram.text,[bigram])
        new_paras = []
        for para in self.rdocs:
            matches = bigram_matcher(para)
            if matches:
                para_text = para.text
                for _id, start, end in matches:
                    match_string = para[start:end]
                    match_replace_with = self.nlp.vocab.strings[_id].replace(" ", "_")
                    para_text = para_text.replace(match_string.text, match_replace_with)
                new_paras.append(para_text)
            else:
                new_paras.append(para.text)
        self.rdocs = self.nlp.pipe(new_paras, n_process=11, batch_size=256)

    def preprocess_paras(self, write_vocab=False):
        if self.bigram:
            print("adding bigrams...")
            self._add_bigrams()
        self.paras_processed = []
        wids = {}
        wids_rev = {}
        wcounts = defaultdict(lambda:0)
        for doc in self.rdocs:
            sents = []
            for s in doc.sents:
                words = []
                for w in s:
                    # PREPROCESS: lemmatize
                    # PREPROCESS: remove * puncuation
                    #                    * words that are / contain numbers
                    #                    * URLs
                    #                    * stopwords
                    #                    * words of length==1
                    if not w.is_punct \
                        and not w.is_space \
                        and not w.like_num \
                        and not any(i.isdigit() for i in w.lemma_) \
                        and not w.like_url \
                        and not w.text.lower() in STOP_WORDS \
                        and len(w.lemma_) > 1:
                        words.append(w.lemma_.lower())
                sents.append(words)
            self.paras_processed.append(sents)
            
        # count words
        for d in self.paras_processed:
            for s in d:
                for w in s:
                    wcounts[w]+=1           

        # PREPROCESS: keep types that occur at least 150 times
        wcounts = {k:v for k,v in wcounts.items() if v>150} 

        # collect word IDs
        for d in self.paras_processed:
            for s in d:
                for w in s:
                    if w in wcounts and w not in wids:
                        wids_rev[len(wids)]=w
                        wids[w]=len(wids)

        # print vocab file
        if write_vocab:
            with open(os.path.join(self.model_root, "vocab.txt"), 'w+') as of:
                for i in range(len(wids_rev)):
                    assert wids[wids_rev[i]]==i
                    of.write(f"{wids_rev[i]}\t{wcounts[wids_rev[i]]}\n")
                        
        # transform
        self.paras_to_wordcounts = []
        self.years_final = []

        for idx, doc in enumerate(self.paras_processed):
            token = [w for s in doc for w in s if w in wcounts]
            type_counts = Counter(token)
            # PREPROCESS: at least 15 token and >5 types per document
            if len(token)>15 and len(type_counts)>5: 
                id_counts = [f"{len(type_counts)}"]+[f"{wids[k]}:{v}" for k,v in type_counts.most_common()]
                self.paras_to_wordcounts.append(' '.join(id_counts))
                self.years_final.append(self.rdates[idx])

    def write_files(self):
        # write -mult file and -seq file
        outmult = open(os.path.join(self.model_root, "model-mult.dat"), 'w+')
        outyear = open(os.path.join(self.model_root, "model-year.dat"), 'w+')
        outseq = open(os.path.join(self.model_root, "model-seq.dat"), 'w+')
        year_dict = {}

        print(len(self.years_final))
        print(len(self.paras_to_wordcounts))

        yearcount = defaultdict(lambda:0)
        for year in range(min(self.dates), max(self.dates), 1):
            for idx, yy in enumerate(self.years_final):
                if year ==yy:
                    yearcount[year]+=1
                    outyear.write(f"{str(yy)}\n")
                    outmult.write(f"{self.paras_to_wordcounts[idx]}\n")

        outseq.write(f"{len(yearcount)}\n")
        for year in sorted(yearcount.keys()):
            outseq.write(f"{yearcount[year]}\n")
            year_dict[len(year_dict)]=year
            
        outyear.close()
        outmult.close()
        outseq.close()
    
    def create(self):
        self.preprocess_paras(write_vocab=True)
        self.write_files()


def create_model_inputs(model_root, csv_path, text_col_name='section_txt', date_col_name='date', bigram=True, seed=42, limit=None):
    jdtmc = DTMCreator(model_root, csv_path, text_col_name=text_col_name, date_col_name=date_col_name, bigram=bigram, seed=seed, limit=limit)
    jdtmc.create()

def fit_mult_model(model_root_dir):
    with open("runs.json", "r") as fp:
        data = json.load(fp)
    for run in data:
        outpath = f"/data/greyroads/energy-roadmap/DTM/{model_root_dir}/model_run_topics{run['topics']}_alpha{run['alpha']}_topic_var{run['topic_var']}"
        print(outpath)
        if not os.path.isdir(outpath):
            os.mkdir(outpath)
        os.system(f"/data/greyroads/energy-roadmap/DTM/dtm/dtm/main   --ntopics={run['topics']}   --mode=fit   --rng_seed=0   --initialize_lda=true   --corpus_prefix=/data/greyroads/energy-roadmap/DTM/{model_root_dir}/eiajournal   --outname={outpath}   --top_chain_var={run['topic_var']}   --alpha={run['alpha']}   --lda_sequence_min_iter=6   --lda_sequence_max_iter=20   --lda_max_em_iter=10")
    return 1

def create_mult_datasets():
    datasets_to_create = [
        {
            "model_root": "journal_energy_policy_applied_energy_all_years_abstract_all",
            "df_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract.csv"),
            "bigram": False
        },
        {
            "model_root": "journal_energy_policy_applied_energy_all_years_abstract_all_bigram",
            "df_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract.csv"),
            "bigram": True
        },
        {
            "model_root": "journal_energy_policy_applied_energy_all_years_abstract_biofuels",
            "df_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract_biofuels.csv"),
            "bigram": False
        },
        {
            "model_root": "journal_energy_policy_applied_energy_all_years_abstract_biofuels_bigram",
            "df_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract_biofuels.csv"),
            "bigram": True
        },
        {
            "model_root": "journal_energy_policy_applied_energy_all_years_abstract_solar",
            "df_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract_solar.csv"),
            "bigram": False
        },
        {
            "model_root": "journal_energy_policy_applied_energy_all_years_abstract_solar_bigram",
            "df_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract_solar.csv"),
            "bigram": True
        },
        {
            "model_root": "journal_energy_policy_applied_energy_all_years_abstract_coal",
            "df_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract_coal.csv"),
            "bigram": False
        },
        {
            "model_root": "journal_energy_policy_applied_energy_all_years_abstract_coal_bigram",
            "df_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract_coal.csv"),
            "bigram": True
        }
    ]
    for dataset in datasets_to_create:
        create_model_inputs(dataset['model_root'], dataset['df_path'], dataset['bigram'])

# def fit_mult_datasets():
#     datasets = [
#         # "journal_energy_policy_applied_energy_all_years_abstract_all_bigram",
#         "journal_energy_policy_applied_energy_all_years_abstract_coal_bigram",
#         "journal_energy_policy_applied_energy_all_years_abstract_solar_bigram",
#         "journal_energy_policy_applied_energy_all_years_abstract_biofuels_bigram",
#         # "journal_energy_policy_applied_energy_all_years_abstract_all",
#         "journal_energy_policy_applied_energy_all_years_abstract_coal",
#         "journal_energy_policy_applied_energy_all_years_abstract_solar",
#         "journal_energy_policy_applied_energy_all_years_abstract_biofuels",
#     ]
#     pool = Pool(8)
#     a = [pool.apply_async(fit_mult_model, args=(dataset,))
#             for dataset in datasets]
#     for fut in a:
#         res = fut.get()
#         if res == 1:
#             print("one successful completion")
#     print("done!")

if __name__ == "__main__":
    # the path to the directory that you want all the files saved in, e.g. *-mult.dat, *-seq.dat, vocab.txt, etc.
    # model_root = os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "journal_energy_policy_applied_energy_all_years_abstract_biofuels_bigram")
    # # the path to the journal paragraphs that are to become part of the fitting data for the topic model.
    # data_path = os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract_biofuels.csv")
    create_model_inputs("1000subset_coal", os.path.join(), bigram=False)
    # create_mult_datasets()
    # fit_mult_datasets()
    # fit_mult_model("journal_energy_policy_applied_energy_all_years_abstract_all_bigram")

    # pid 7380