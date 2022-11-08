import numpy as np
import pandas as pd


"""
Module used to import real time data.
"""

CANADA_CASES_URL = "https://raw.githubusercontent.com/ccodwg/Covid19Canada/master/timeseries_canada/cases_timeseries_canada.csv"
UN_POPULATION_CSV_URL = "https://raw.githubusercontent.com/owid/covid-19-data/152b2236a32f889df3116c7121d9bb14ce2ff2a8/scripts/input/un/population_2020.csv"

JHU_CSV_URL = "https://raw.githubusercontent.com/datasets/covid-19/main/data/countries-aggregated.csv"

def get_cases():
    population_df = pd.read_csv(
        UN_POPULATION_CSV_URL,
        keep_default_na=False,
        usecols=["entity", "year", "population"],
    )
    population_df = population_df.loc[population_df["entity"] == "Canada"]
    population_df = population_df.loc[population_df["year"] == 2020]
    population = population_df["population"].to_numpy(dtype=float).item()

    cases = pd.read_csv(CANADA_CASES_URL, index_col="date_report")["cases"].to_numpy(dtype=float)
    cum_cases = pd.read_csv(CANADA_CASES_URL, index_col="date_report")["cumulative_cases"].to_numpy(dtype=float)

    return cases, cum_cases, population