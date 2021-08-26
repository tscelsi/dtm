import os
import pandas as pd
from coherence import CoherenceAnalysis
from pprint import pprint
import re

def analyse(coh, ds, model, coherences, save_dir, plot=True, coherence=True, terms=True, document_topic_dist=True):
    analysis_save_dir = save_dir
    # topic_analysis_save_dir = os.path.join(analysis_save_dir, "topic_analysis")
    model_topic_analysis_save_dir = os.path.join(analysis_save_dir, model)
    if not os.path.isdir(analysis_save_dir):
        os.mkdir(analysis_save_dir)
    if not os.path.isdir(model_topic_analysis_save_dir):
        os.mkdir(model_topic_analysis_save_dir)
    if document_topic_dist:
        coh.save_gammas(os.path.join(model_topic_analysis_save_dir, "doc_topic_distribution.csv"))
    if plot:
    # plot the topic distributions over time for this model
        print("plotting topics...")
        try:
            coh.plot_topics_ot(os.path.join(model_topic_analysis_save_dir, f"{model}.png"))
        except Exception as e:
            print(f"plot failed for model: {model}")
    if coherence:
        # get the coherence of this model
        print("calculating coherence...")
        pmi_coh = coh.get_coherence()
        npmi_coh = coh.get_coherence("c_npmi")
        coherences[model] = {}
        coherences[model]['pmi'] = pmi_coh
        coherences[model]['npmi'] = npmi_coh
        with open(os.path.join(analysis_save_dir, "coherences.txt"), "w") as fp:
            fp.write("Model\tPMI\tNPMI\n")
            for k,v in coherences.items():
                fp.write(f"{k}\t{v['pmi']}\t{v['npmi']}\n")
    if terms:
        tfidf_topic_names = coh.get_topic_names(detailed=True)
        emb_topic_names = coh.get_topic_names(detailed=True, _type="embedding")
        topw_df = coh.create_top_words_df(n=30)
        with open(os.path.join(model_topic_analysis_save_dir, "all_topics_top_terms.txt"), "w") as fp1, \
            open(os.path.join(model_topic_analysis_save_dir, f"all_topics_top_terms_ot.txt"), "w") as fp2:
            for i in range(len(tfidf_topic_names)):
                tfidf_topic_name, topic_top_terms = tfidf_topic_names[i]
                emb_topic_name, _ = emb_topic_names[i]
                # word_dist_arr_ot = coh.get_topic_word_distributions_ot(i)
                # topic_top_terms = coh.get_words_for_topic(word_dist_arr_ot, with_prob=False)
                fp1.write(f"tfidf topic {i} labels: ({tfidf_topic_name})\n{topic_top_terms}\n==========\n")
                fp2.write(f"\n=========\ntfidf topic {i} labels: ({tfidf_topic_name})\nemb topic {i} labels: ({emb_topic_name})\n=========\n\n")
                top_words_for_topic = topw_df[topw_df['topic_idx'] == i].loc[:, ['year', 'top_words']]
                for row in top_words_for_topic.itertuples():
                    fp2.write(f"{row.year}\t{row.top_words}\n")
                
                

def analyse_model():
    dataset = {
        "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "journal_energy_policy_applied_energy_all_years_abstract_all_bigram"),
        "data_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract.csv"),
        "ndocs": 19971,
        "bigram": True
    }
    MODEL_NAME = "model_run_topics30_alpha0.01_topic_var0.05"
    coherences = {}
    # coh = CoherenceAnalysis(
    #             dataset['data_path'],
    #             "journals",
    #             "section_txt",
    #             "date",
    #             dataset['bigram'],
    #             dataset.get("limit"),
    #             dataset['ndocs'], 
    #             int(MODEL_NAME.split("_")[2].split("topics")[1]), 
    #             model_root=dataset['model_root'],
    #             doc_year_map_file_name="eiajournal-year.dat",
    #             seq_dat_file_name="eiajournal-seq.dat",
    #             vocab_file_name="vocab.txt",
    #             model_out_dir=MODEL_NAME,
    #             eurovoc_whitelist=False
    #         )
    coh = CoherenceAnalysis(
                None,
                "reverse",
                None,
                None,
                dataset['bigram'],
                dataset.get("limit"),
                dataset['ndocs'], 
                int(MODEL_NAME.split("_")[2].split("topics")[1]),
                model_root=dataset['model_root'],
                doc_year_map_file_name="greyroads-year.dat",
                seq_dat_file_name="greyroads-seq.dat",
                vocab_file_name="vocab.txt",
                model_out_dir=MODEL_NAME,
                eurovoc_whitelist=True
            )
    # coh.init_coherence()
    analyse(coh, dataset, MODEL_NAME, coherences, coherence=False)

def journals_analyse_multi_models():
    datasets = [
        {
            "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "dtm", "datasets", "journal_energy_policy_applied_energy_all_years_abstract_all_ngram"),
            "data_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract.csv"),
            "bigram": True,
        },
        # {
        #     "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "dtm", "datasets", "journal_energy_policy_applied_energy_all_years_abstract_biofuels_ngram_min_freq_50"),
        #     "data_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract_biofuels.csv"),
        #     "bigram": True,
        # },
        # {
        #     "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "dtm", "datasets", "journal_energy_policy_applied_energy_all_years_abstract_coal_ngram_min_freq_50"),
        #     "data_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract_coal.csv"),
        #     "bigram": True,
        # },
        # {
        #     "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "dtm", "datasets", "journal_energy_policy_applied_energy_all_years_abstract_solar_ngram_min_freq_50"),
        #     "data_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract_solar.csv"),
        #     "bigram": True,
        # }
    ]
    for ds in datasets:
        print("=========")
        print(f"DATASET: {ds['model_root']}")
        dirs = os.listdir(ds['model_root'])
        df_models = [x for x in dirs if x.startswith("k")]
        coherences = {}
        ndocs = sum([int(x) for x in open(os.path.join(ds['model_root'], 'model-seq.dat')).readlines()[1:]])
        for model in df_models:
            print(f"analysing model {model}")
            coh = CoherenceAnalysis(
                None,
                "reverse",
                None,
                None,
                ds['bigram'],
                ds.get("limit"),
                ndocs,
                int(model.split("_")[0].split("k")[1]), 
                model_root=ds['model_root'],
                doc_year_map_file_name="model-year.dat",
                seq_dat_file_name="model-seq.dat",
                vocab_file_name="vocab.txt",
                model_out_dir=model,
                eurovoc_whitelist=True
            )
            # coh.init_coherence()
            analyse(coh, ds, model, coherences, coherence=False)


def analyse_multi_models(root_save_dir=os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "dtm", "analysis")):
    if not os.path.isdir(root_save_dir):
        os.mkdir(root_save_dir)
    datasets = [
        {
            "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "dtm", "datasets", "journal_energy_policy_applied_energy_all_years_abstract_all_ngram"),
            "data_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract.csv"),
            "bigram": True,
        },
        {
            "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "dtm", "datasets", "greyroads_aeo_all_ngram"),
            "data_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "greyroads_aeo_all.csv"),
            "bigram": True
        },
        {
            "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "dtm", "datasets", "greyroads_ieo_all_ngram"),
            "data_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "greyroads_ieo_all.csv"),
            "bigram": True
        }
    ]
    whitelist_k = [30]
    whitelist_a = [0.01]
    whitelist_var = [0.05]
    for ds in datasets:
        dirs = os.listdir(ds['model_root'])
        df_models = [x for x in dirs if x.startswith("k")]
        pattern = re.compile(r"k(?P<k>\d+)_a(?P<a>\d+\.\d+)_var(?P<var>\d+\.\d+)")
        for model in df_models:
            m = pattern.match(model)
            k = int(m.group('k'))
            a = float(m.group('a'))
            var = float(m.group('var'))
            if k not in whitelist_k or a not in whitelist_a or var not in whitelist_var:
                continue
            ndocs = sum([int(x) for x in open(os.path.join(ds['model_root'], 'model-seq.dat')).readlines()[1:]])
            coh = CoherenceAnalysis(
                None,
                "reverse",
                None,
                None,
                ds['bigram'],
                ds.get("limit"),
                ndocs, 
                int(model.split("_")[0].split("k")[1]), 
                model_root=ds['model_root'],
                doc_year_map_file_name="model-year.dat",
                seq_dat_file_name="model-seq.dat",
                vocab_file_name="vocab.txt",
                model_out_dir=model,
                eurovoc_whitelist=True
            )
            model_dir = os.path.join(root_save_dir, ds['model_root'].split("/")[-1])
            if not os.path.isdir(model_dir):
                os.mkdir(model_dir)
            save_dir = os.path.join(model_dir, "analysis_n30")
            analyse(coh, ds, model, {}, save_dir, coherence=False, plot=False, document_topic_dist=False)


def hansard_analyse_multi_models():
    datasets = [
        {
            "model_root": os.path.join(os.environ['DTM_ROOT'], "dtm", "datasets", "dataset_greens_1990_2021_min_freq_40"),
            "data_path": os.path.join(os.environ['HANSARD'],"coal_data", "04_model_inputs", "coal_so_300t_Greens.csv"),
            "bigram": True,
        },
        {
            "model_root": os.path.join(os.environ['DTM_ROOT'], "dtm", "datasets", "dataset_liberal_min_freq_80"),
            "data_path": os.path.join(os.environ['HANSARD'],"coal_data", "04_model_inputs", "coal_so_300t_Liberal.csv"),
            "bigram": True,
        },
        {
            "model_root": os.path.join(os.environ['DTM_ROOT'], "dtm", "datasets", "dataset_labor_min_freq_80"),
            "data_path": os.path.join(os.environ['HANSARD'],"coal_data", "04_model_inputs", "coal_so_300t_Labor.csv"),
            "bigram": True,
        },
        {
            "model_root": os.path.join(os.environ['DTM_ROOT'], "dtm", "datasets", "dataset_2a_last_20_years_ngram"),
            "data_path": os.path.join(os.environ['HANSARD'],"coal_data", "06_dtm", "dataset_2a_last_20_years.csv"),
            "bigram": True,
        },
        {
            "model_root": os.path.join(os.environ['DTM_ROOT'], "dtm", "datasets", "dataset_2a_ngram"),
            "data_path": os.path.join(os.environ['HANSARD'],"coal_data", "06_dtm", "dataset_2a.csv"),
            "bigram": True,
        },
    ]
    for ds in datasets:
        dirs = os.listdir(ds['model_root'])
        df_models = [x for x in dirs if x.startswith("k")]
        coherences = {}
        for model in df_models:
            ndocs = sum([int(x) for x in open(os.path.join(ds['model_root'], 'model-seq.dat')).readlines()[1:]])
            coh = CoherenceAnalysis(
                None,
                "reverse",
                None,
                None,
                ds['bigram'],
                ds.get("limit"),
                ndocs, 
                int(model.split("_")[0].split("k")[1]), 
                model_root=ds['model_root'],
                doc_year_map_file_name="model-year.dat",
                seq_dat_file_name="model-seq.dat",
                vocab_file_name="vocab.txt",
                model_out_dir=model,
                eurovoc_whitelist=True
            )
            # print("initialising coherence...")
            # coh.init_coherence(os.path.join(ds['model_root'], "model-mult.dat"), os.path.join(ds['model_root'], "vocab.txt"))
            # print("done! analysing...")
            analyse(coh, ds, model, coherences, coherence=False)

def greyroads_analyse_multi_models():
    datasets = [
        {
            "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "dtm", "datasets", "greyroads_aeo_all_ngram"),
            "data_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "greyroads_aeo_all.csv"),
            "bigram": True
        },
        {
            "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "dtm", "datasets", "greyroads_ieo_all_ngram"),
            "data_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "greyroads_ieo_all.csv"),
            "bigram": True
        }
    ]
    for ds in datasets:
        dirs = os.listdir(ds['model_root'])
        df_models = [x for x in dirs if x.startswith("k")]
        coherences = {}
        ndocs = sum([int(x) for x in open(os.path.join(ds['model_root'], 'model-seq.dat')).readlines()[1:]])
        for model in df_models:
            coh = CoherenceAnalysis(
                None,
                "reverse",
                None,
                None,
                ds['bigram'],
                ds.get("limit"),
                ndocs, 
                int(model.split("_")[0].split("k")[1]), 
                model_root=ds['model_root'],
                doc_year_map_file_name="model-year.dat",
                seq_dat_file_name="model-seq.dat",
                vocab_file_name="vocab.txt",
                model_out_dir=model,
                eurovoc_whitelist=True
            )
            coh.init_coherence(os.path.join(ds['model_root'], "model-mult.dat"), os.path.join(ds['model_root'], "vocab.txt"))
            analyse(coh, ds, model, coherences, terms=False, document_topic_dist=False)

def validate_multi_models():
    greyroads_datasets = [
        {
            "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "greyroads_steo_all"),
            "data_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "greyroads_steo_all.csv"),
            "ndocs": 1688
        },
        {
            "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "greyroads_steo_all_bigram"),
            "data_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "greyroads_steo_all.csv"),
            "ndocs": 1683
        },
        {
            "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "greyroads_ieo_all"),
            "data_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "greyroads_ieo_all.csv"),
            "ndocs": 1095
        },
        {
            "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "greyroads_ieo_all_bigram"),
            "data_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "greyroads_ieo_all.csv"),
            "ndocs": 1093
        },
        {
            "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "greyroads_aeo_all"),
            "data_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "greyroads_aeo_all.csv"),
            "ndocs": 2449
        },
        {
            "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "greyroads_aeo_all_bigram"),
            "data_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "greyroads_aeo_all.csv"),
            "ndocs": 2447
        }]
    journals_datasets = [
        {
            "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "journal_energy_policy_applied_energy_all_years_abstract_biofuels_bigram"),
            "data_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract_biofuels.csv"),
            "ndocs": 2508
        },
        {
            "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "journal_energy_policy_applied_energy_all_years_abstract_biofuels"),
            "data_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract_biofuels.csv"),
            "ndocs": 2508
        },
        {
            "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "journal_energy_policy_applied_energy_all_years_abstract_solar_bigram"),
            "data_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract_solar.csv"),
            "ndocs": 3078
        },
        {
            "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "journal_energy_policy_applied_energy_all_years_abstract_solar"),
            "data_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract_solar.csv"),
            "ndocs": 3078
        },
        {
            "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "journal_energy_policy_applied_energy_all_years_abstract_coal_bigram"),
            "data_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract_coal.csv"),
            "ndocs": 1577
        },
        {
            "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "journal_energy_policy_applied_energy_all_years_abstract_coal"),
            "data_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract_coal.csv"),
            "ndocs": 1582
        },
        {
            "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "journal_energy_policy_applied_energy_all_years_abstract_all"),
            "data_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract.csv"),
            "ndocs": 19973
        },
        {
            "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "journal_energy_policy_applied_energy_all_years_abstract_all_bigram"),
            "data_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract.csv"),
            "ndocs": 19971
        },
    ]
    for ds in journals_datasets:
        dirs = os.listdir(ds['model_root'])
        df_models = [x for x in dirs if x.startswith("model_run_")]
        for model in df_models:
            try:
                coh = CoherenceAnalysis(
                    ds['data_path'],
                    "journal",
                    False,
                    ds['ndocs'], 
                    int(model.split("_")[2].split("topics")[1]),
                    model_root=ds['model_root'],
                    doc_year_map_file_name="eiajournal-year.dat",
                    seq_dat_file_name="eiajournal-seq.dat",
                    vocab_file_name="vocab.txt",
                    model_out_dir=model,
                    eurovoc_whitelist=True
                )
            except AssertionError as e:
                print("==================")
                print(f"Something wrong with dataset: {ds['model_root']}")
                print(f"model: {model}")
                print(e)
                print("==================")
                break


if __name__ == "__main__":
    # greyroads_analyse_multi_models()
    # validate_multi_models()
    # hansard_analyse_multi_models()
    # journals_analyse_multi_models()
    analyse_multi_models()
    # analyse_model()