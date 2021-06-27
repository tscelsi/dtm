import os
import pandas as pd
from coherence import CoherenceAnalysis
from pprint import pprint

def analyse(coh, ds, model, coherences):
    analysis_save_dir = os.path.join(ds['model_root'],"analysis_all_eurovoc_topics")
    if not os.path.isdir(analysis_save_dir):
        os.mkdir(analysis_save_dir)
    # plot the topic distributions over time for this model
    print("plotting topics...")
    coh.plot_topics_ot(os.path.join(analysis_save_dir, f"{model}.png"))
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
    topic_analysis_save_dir = os.path.join(analysis_save_dir, "topic_analysis")
    if not os.path.isdir(topic_analysis_save_dir):
        os.mkdir(topic_analysis_save_dir)
    model_topic_analysis_save_dir = os.path.join(topic_analysis_save_dir, model)
    if not os.path.isdir(model_topic_analysis_save_dir):
        os.mkdir(model_topic_analysis_save_dir)
    # get the top 10 topic words for all time
    topic_names = coh.get_topic_names()
    with open(os.path.join(model_topic_analysis_save_dir, "all_topics_top_terms.txt"), "w") as fp:
        for i in range(len(topic_names)):
            word_dist_arr_ot = coh.get_topic_word_distributions_ot(i)
            topic_top_terms = coh.get_words_for_topic(word_dist_arr_ot, with_prob=False)
            fp.write(f"topic {i} ({topic_names[i]})\n{topic_top_terms}\n==========\n")
    # get the top 10 topic words for each topic over time
    with open(os.path.join(model_topic_analysis_save_dir, f"all_topics_top_terms_ot.txt"), "w") as fp:
        for i in range(len(topic_names)):
            topw_df = coh.create_top_words_df()
            fp.write(f"\n=========\ntopic {i} ({topic_names[i]})\n=========\n\n")
            top_words_for_topic = topw_df[topw_df['topic_idx'] == i].loc[:, ['year', 'top_words']]
            for row in top_words_for_topic.itertuples():
                fp.write(f"{row.year}\t{row.top_words}\n")

def analyse_model():
    dataset = {
        "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "journal_energy_policy_applied_energy_all_years_abstract_all_bigram"),
        "data_path": os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract.csv"),
        "ndocs": 19971,
        "bigram": True
    }
    MODEL_NAME = "model_run_topics30_alpha0.01_topic_var0.05"
    coherences = {}
    coh = CoherenceAnalysis(
                dataset['data_path'],
                "journals",
                "section_txt",
                "date",
                dataset['bigram'],
                dataset.get("limit"),
                dataset['ndocs'], 
                int(MODEL_NAME.split("_")[2].split("topics")[1]), 
                model_root=dataset['model_root'],
                doc_year_map_file_name="eiajournal-year.dat",
                seq_dat_file_name="eiajournal-seq.dat",
                vocab_file_name="vocab.txt",
                model_out_dir=MODEL_NAME,
                eurovoc_whitelist=False
            )
    coh.init_coherence()
    analyse(coh, dataset, MODEL_NAME, coherences)


def hansard_analyse_multi_models():
    datasets = [
        {
            "model_root": os.path.join(os.environ['DTM_ROOT'], "dtm", "dataset_lea_test"),
            "data_path": os.path.join(os.environ['HANSARD'],"coal_data", "04_model_inputs", "final_1000_40000_lea_0906.tsv"),
            "ndocs": 6198,
            "bigram": False,
        },
    ]
    for ds in datasets:
        dirs = os.listdir(ds['model_root'])
        df_models = [x for x in dirs if x.startswith("model_run_")]
        coherences = {}
        for model in df_models:
            coh = CoherenceAnalysis(
                None,
                "reverse",
                None,
                None,
                ds['bigram'],
                ds.get("limit"),
                ds['ndocs'], 
                int(model.split("_")[2].split("topics")[1]), 
                model_root=ds['model_root'],
                doc_year_map_file_name="model-year.dat",
                seq_dat_file_name="model-seq.dat",
                vocab_file_name="vocab.txt",
                model_out_dir=model,
                eurovoc_whitelist=False
            )
            print("initialising coherence...")
            coh.init_coherence(os.path.join(ds['model_root'], "model-mult.dat"), os.path.join(ds['model_root'], "vocab.txt"))
            print("done! analysing...")
            analyse(coh, ds, model, coherences)

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
    hansard_analyse_multi_models()
    # analyse_model()