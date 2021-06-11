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

bigram_path = os.path.join(os.environ['HANSARD'], "coal_data", "04_model_inputs", "BIGRAMS.txt")
# bigram_path = os.path.join(os.environ['ROADMAP_SCRAPER'], "BIGRAMS.txt")
SEED = 42


class DTMCreator:
    def __init__(self, model_root, csv_path, text_col_name='section_txt', date_col_name='date', bigram=True, limit=None, years_per_step=1):
        self.df = pd.read_csv(csv_path)
        self.df = self.df.dropna(subset=[text_col_name])
        self.paragraphs = self.df[text_col_name].tolist()
        self.dates = self._extract_dates(date_col_name)
        self.bigram = bigram
        self.years_per_step = years_per_step
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
        self.nlp.add_pipe('sentencizer')
        self.rdocs =[]
        self.rdates = []
        rand_indexes = [idx for idx in random.RandomState(SEED).permutation(len(self.paragraphs))]
        if limit:
            self.rdocs = self.nlp.pipe([self.paragraphs[i] for i in rand_indexes[:limit]], n_process=11, batch_size=256)
            self.rdates = [self.dates[i] for i in rand_indexes[:limit]]
        else:
            self.rdocs = self.nlp.pipe([self.paragraphs[i] for i in rand_indexes], n_process=11, batch_size=256)
            self.rdates = [self.dates[i] for i in rand_indexes]
        return

    def _extract_dates(self, date_col_name):
        """Here we extract the dates for each document by taking a particular column of the dataframe containing the corpus
        and extracting the year from a date stamp. This will need to be overidden depending on the format of your date and the column name. 
        In this case, the df has a column labelled 'date' which contains a date of form 2020-10-09. 
        We extract the year (2020) for each doc in the df.

        An simple example function could be:
            return [int(d.split("-")[0]) for d in self.df[date_col_name].tolist()]

        MAKE GENERAL
        """
        raise NotImplementedError

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

    def _get_year_batches(self):
        year_mapping = {}
        for date in self.rdates:
            batch_num = int((date - min(self.rdates)) / self.years_per_step)
            year_mapping[date] = batch_num
        return year_mapping

    def preprocess_paras(self, min_freq=150, write_vocab=False):
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
        wcounts = {k:v for k,v in wcounts.items() if v>min_freq} 

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

        # if we need to merge years, then it is done through the years_per_step var
        if self.years_per_step != 1:
            self.year_mapping = self._get_year_batches()

        for idx, doc in enumerate(self.paras_processed):
            token = [w for s in doc for w in s if w in wcounts]
            type_counts = Counter(token)
            # PREPROCESS: at least 15 token and >5 types per document
            if len(token)>15 and len(type_counts)>5: 
                id_counts = [f"{len(type_counts)}"]+[f"{wids[k]}:{v}" for k,v in type_counts.most_common()]
                self.paras_to_wordcounts.append(' '.join(id_counts))
                if self.years_per_step != 1:
                    self.years_final.append(self.year_mapping[self.rdates[idx]])
                else:
                    self.years_final.append(self.rdates[idx])

    def write_files(self, min_year=None, max_year=None):
        # write -mult file and -seq file
        outmult = open(os.path.join(self.model_root, "model-mult.dat"), 'w+')
        outyear = open(os.path.join(self.model_root, "model-year.dat"), 'w+')
        outseq = open(os.path.join(self.model_root, "model-seq.dat"), 'w+')
        year_dict = {}

        print(len(self.years_final))
        print(len(self.paras_to_wordcounts))

        yearcount = defaultdict(lambda:0)

        if self.years_per_step == 1:
            min_date = min_year if min_year else min(self.dates)
            max_date = max_year if max_year else max(self.dates)
        else:
            min_date = min(self.year_mapping.values())
            max_date = max(self.year_mapping.values())

        for year in range(min_date, max_date, 1):
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
    
    def create(self, min_freq=150):
        self.preprocess_paras(min_freq=min_freq, write_vocab=True)
        self.write_files()

def fit(run, model_root_dir):
    outpath = os.path.join(os.environ['DTM_ROOT'], "dtm", model_root_dir, f"model_run_topics{run['topics']}_alpha{run['alpha']}_topic_var{run['topic_var']}")
    print(outpath)
    if not os.path.isdir(outpath):
        os.mkdir(outpath)
    cmd = os.path.join(os.environ['DTM_ROOT'], "dtm", "dtm", "main") + f" --ntopics={run['topics']}   --mode=fit   --rng_seed=0   --initialize_lda=true   --corpus_prefix={os.environ['DTM_ROOT']}dtm/{model_root_dir}/model   --outname={outpath}   --top_chain_var={run['topic_var']}   --alpha={run['alpha']}   --lda_sequence_min_iter=6   --lda_sequence_max_iter=20   --lda_max_em_iter=10"
    print(cmd)
    os.system(cmd)
    return 1

def fit_mult_model(model_root_dir):
    with open("runs.json", "r") as fp:
        data = json.load(fp)
    with Pool(processes=11) as pool:
        multiple_results = [pool.apply_async(fit, (run, model_root_dir, ))for run in data]
    print("here")
    for res in multiple_results:
        print(res.get())
    return 1

# def create_mult_datasets():
#     datasets_to_create = [
#         {
#             "model_root": "journal_energy_policy_applied_energy_all_years_abstract_all",
#             "df_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract.csv"),
#             "bigram": False
#         },
#         {
#             "model_root": "journal_energy_policy_applied_energy_all_years_abstract_all_bigram",
#             "df_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract.csv"),
#             "bigram": True
#         },
#         {
#             "model_root": "journal_energy_policy_applied_energy_all_years_abstract_biofuels",
#             "df_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract_biofuels.csv"),
#             "bigram": False
#         },
#         {
#             "model_root": "journal_energy_policy_applied_energy_all_years_abstract_biofuels_bigram",
#             "df_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract_biofuels.csv"),
#             "bigram": True
#         },
#         {
#             "model_root": "journal_energy_policy_applied_energy_all_years_abstract_solar",
#             "df_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract_solar.csv"),
#             "bigram": False
#         },
#         {
#             "model_root": "journal_energy_policy_applied_energy_all_years_abstract_solar_bigram",
#             "df_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract_solar.csv"),
#             "bigram": True
#         },
#         {
#             "model_root": "journal_energy_policy_applied_energy_all_years_abstract_coal",
#             "df_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract_coal.csv"),
#             "bigram": False
#         },
#         {
#             "model_root": "journal_energy_policy_applied_energy_all_years_abstract_coal_bigram",
#             "df_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract_coal.csv"),
#             "bigram": True
#         }
#     ]
#     for dataset in datasets_to_create:
#         create_model_inputs(dataset['model_root'], dataset['df_path'], dataset['bigram'])

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
    # create_model_inputs("tom_test", os.path.join(os.environ['HANSARD'], "coal_data", "04_model_inputs", "coal_full_downloaded.csv"), text_col_name="main_text", date_col_name="date", bigram=False, limit=100)
    # create_mult_datasets()
    # fit_mult_datasets()
    fit_mult_model("dataset_20000subset_coal")

    # pid 7380