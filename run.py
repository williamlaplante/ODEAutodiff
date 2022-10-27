import matplotlib.pyplot as plt
import torch as tc
import pandas as pd
from torchdiffeq import odeint
from torch import nn

CANADA_CASES_URL = "https://raw.githubusercontent.com/ccodwg/Covid19Canada/master/timeseries_canada/cases_timeseries_canada.csv"
UN_POPULATION_CSV_URL = "https://raw.githubusercontent.com/owid/covid-19-data/152b2236a32f889df3116c7121d9bb14ce2ff2a8/scripts/input/un/population_2020.csv"


def get_data():
    population_df = pd.read_csv(
        UN_POPULATION_CSV_URL,
        keep_default_na=False,
        usecols=["entity", "year", "population"],
    )
    population_df = population_df.loc[population_df["entity"] == "Canada"]
    population_df = population_df.loc[population_df["year"] == 2020]
    population = tc.tensor(population_df["population"].to_numpy()).float()

    cases = tc.tensor(pd.read_csv(CANADA_CASES_URL, index_col="date_report")["cases"].to_numpy())
    

    return cases, population


class ODE(nn.Module):

    def __init__(self, y0):
        super().__init__()
        self.y0 = y0
        self.gamma = 0.1
        self.population = y0[-2] #second to last parameter in y0 = S, I, R, N, beta
        
        self.net = nn.Sequential(
            nn.Linear(2, 10),
            nn.Tanh(),
            nn.Linear(10, 1),
        )

    def forward(self, t, y):
        S, I, R, N, beta = y
        dB = self.net(beta, t)
        dS = - self.params[0] * S * I / N
        dI = self.params[0] * S * I / N - self.gamma * I
        dR = self.gamma * I
        dN = tc.tensor(0.0)

        return tc.stack([dS, dI, dR, dN, dB])
    
    def predict(self, t):
        S, I, R, N, beta = odeint(self, self.y0, t).T
        pred = beta * S * I / self.population
        return pred

    def likelihood(self, data, t):
        pred = self.predict(t)

        return (1/self.population) * (pred.mean() - (1/len(t)) * tc.dot(pred.log(), data))


def run_all():
    data, N = get_data()

    tmin = 0
    tmax = 200 #days

    tspan = tc.arange(tmin, tmax).float()

    N = 10**6
    I0 = 1
    R0 = 0
    S0 = N - I0 - R0
    beta0 = 0.01

    y0 = tc.tensor([S0, I0, R0, N, beta0]).float()

    func = ODE(y0=y0)

    optimizer = tc.optim.Adam(func.parameters(), lr=1e-1)

    losses = []
    for _ in range(10):
        optimizer.zero_grad()
        loss = func.likelihood(data, tspan)
        loss.backward()
        optimizer.step()
        losses.append(loss.clone().detach())
    
    print(losses)

if __name__=="__main__":
    run_all()