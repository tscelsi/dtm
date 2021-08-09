#! env/bin/python3
"""
Author: Thomas Scelsi
Email: tscelsi@student.unimelb.edu.au
File that creates a DTM-compatible dataset for the 'greyroads' dataset as mentioned in the
accompanying paper.
"""

from dtm_creator import DTMCreator
import os

class GreyroadsDTMCreator(DTMCreator):
    def _extract_dates(self, date_col_name):
        return [int(x) for x in self.df[date_col_name].tolist()]


def create_model_inputs(model_root, csv_path, bigram=True, limit=None, min_freq=20, ds_upper_limit=1000):
    jdtmc = GreyroadsDTMCreator(model_root, csv_path, text_col_name='para_text', date_col_name='year', bigram=bigram, limit=limit)
    jdtmc.create(min_freq, ds_upper_limit)

if __name__ == "__main__":
    # replace
    # create_model_inputs(
    #     "test", 
    #     os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "greyroads_aeo_all.csv"), 
    #     bigram=True, 
    #     limit=None,
    # )
    docs = [("the international energy outlook 2004 (ieo2004) presents an assessment by the energy information administra tion (eia) of the outlook for international energy mar kets through 2025. u.s. projections appearing in ieo2004 are consistent with those published in eia’s annual energy outlook 2004 (aeo2004), which was pre pared using the national energy modeling system (nems). ieo2004 is provided as a service to energy managers and analysts, both in government and in the private sector. the projections are used by international agencies, federal and state governments, trade associa tions, and other planners and decisionmakers. they are published pursuant to the department of energy orga nization act of 1977 (public law 95-91), section 205(c). the ieo2004 projections are based on u.s. and foreign government laws in effect on october 1, 2003. the report begins with a review of world trends in energy demand and the macroeconomic assumptions used as a major driver in deriving the projections that appear in the ieo2004. the historical time frame begins with data from 1970 and extends to 2001, providing readers with a 31-year historical view of energy demand. the ieo2004 projections extend to 2025, giving readers a 24-year forecast period. new to this report is a discussion on regional end-use consumption issues in the residential, commercial, and industrial sectors. high economic growth and low economic growth cases were developed to depict a set of alternative growth paths for the energy forecast. the two cases consider alternative growth paths for regional gross domestic product (gdp). the resulting projections and the uncer tainty associated with making international energy pro jections in general are discussed in the first chapter of the report. the status of environmental indicators, includ ing global carbon emissions, is reviewed. the next part of the report is organized by energy source. regional consumption projections for oil, natu ral gas, and coal are presented in the three fuel chapters, along with a review of the current status of each fuel on a worldwide basis. a chapter on electricity markets fol lows, with a review of trends for nuclear power and hydroelectricity and other marketed renewable energy resources. the report ends with a discussion of energy and environmental issues, with particular attention to the outlook for global carbon dioxide emissions. appendix a contains summary tables of the ieo2004 reference case projections for world energy consump tion, gdp, energy consumption by fuel, electricity con sumption, carbon dioxide emissions, nuclear generating capacity, energy consumption measured in oil-equiva lent units, and regional population growth. the reference case projections of total foreign energy con sumption and consumption of oil, natural gas, coal, and renewable energy were prepared using eia’s system for the analysis of global energy markets (sage), as were projections of net electricity consumption, energy con sumed by fuel for the purpose of electricity generation, and carbon dioxide emissions. in addition, the nems coal export submodule (ces) was used to derive flows in international coal trade, presented in the coal chapter. nuclear capacity projections for the reference case were based on analysts’ knowledge of the nuclear programs in different countries. appendixes b and c present projections for the high and low economic growth cases, respectively. appendix d contains summary tables of projections for world oil production capacity and oil production in the reference case and two alternative cases: high oil price and low oil price. the projections were derived from sage and from the u.s. geological survey.", 2020)]
    gdtmc = GreyroadsDTMCreator("test", docs, text_col_name='para_text', date_col_name='year', bigram=True, limit=None)
    gdtmc.preprocess_paras(min_freq=20, ngrams=True, write_vocab=False)