#! env/bin/python3
"""
Author: Thomas Scelsi
Email: tscelsi@student.unimelb.edu.au
File that creates a DTM-compatible dataset for the 'hansard' dataset.
"""

from .dtm_creator import DTMCreator
from collections import defaultdict
import os
import pandas as pd


class HansardDTMCreator(DTMCreator):
    def _extract_dates(self, date_col_name):
        return [int(d.split("-")[2]) for d in self.df[date_col_name].tolist()]
    
    def create_model_input_from_df(self, df, n=None):
        if n:
            df = df.sample(n)
        df['year'] = df['date'].apply(lambda x: int(x.split("-")[2]))
        batched_years, year_mapping = self.batch_years(df['year'].to_list(), return_mapping=True)
        self.write_files_hansard(year_mapping, batched_years, df['wbow'].to_list())

    def write_files_hansard(self, year_mapping, years, wbow, min_year=None, max_year=None):
        # write -mult file and -seq file
        outmult = open(os.path.join(self.model_root, "model-mult.dat"), 'w+')
        outyear = open(os.path.join(self.model_root, "model-year.dat"), 'w+')
        outseq = open(os.path.join(self.model_root, "model-seq.dat"), 'w+')
        year_dict = {}

        yearcount = defaultdict(lambda:0)

        min_date = min_year if min_year else min(year_mapping.values())
        max_date = max_year if max_year else max(year_mapping.values())
        for year in range(min_date, max_date + 1, 1):
            for idx, yy in enumerate(years):
                if year ==yy:
                    yearcount[year]+=1
                    outyear.write(f"{str(yy)}\n")
                    outmult.write(f"{len(wbow[idx].split(' '))} {wbow[idx]}\n")
        outseq.write(f"{len(yearcount)}\n")
        for year in sorted(yearcount.keys()):
            outseq.write(f"{yearcount[year]}\n")
            year_dict[len(year_dict)]=year
            
        outyear.close()
        outmult.close()
        outseq.close()

def create_model_inputs(model_root, csv_path, text_col_name='section_txt', date_col_name='date', bigram=True, limit=None, years_per_step=1):
    jdtmc = HansardDTMCreator(model_root, csv_path, text_col_name=text_col_name, date_col_name=date_col_name, bigram=bigram, limit=limit, years_per_step=years_per_step)
    jdtmc.create()


if __name__ == "__main__":
    hdtm = HansardDTMCreator(
        "dataset_lea_test", 
        os.path.join(os.environ['HANSARD'], "coal_data", "06_dtm", "dtm", "lea_test", "final_100_40000.tsv"), 
        text_col_name="main_text", 
        date_col_name="date", 
        bigram=False, 
        limit=None,
        years_per_step=10
    )
    df = pd.read_csv(os.path.join(os.environ['HANSARD'], "coal_data", "06_dtm", "dtm", "lea_test", "final_100_40000.tsv"), sep="\t").dropna(subset=['wbow'])
    hdtm.create_model_input_from_df(df)
