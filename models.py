import torch as tc
from torch import nn
from torchdiffeq import odeint_adjoint as odeint
from functools import partial

class SIR(nn.Module):

    def __init__(self, beta, gamma, S0, I0, R0, N, data = None, tmin = 0, tmax = 200, dt = 0.1):
        super(SIR, self).__init__()

        self.params = nn.Parameter(tc.tensor([beta, gamma]).float())
        self.init_conditions = tc.tensor([S0, I0, R0, N]).float()
        self.t = tc.arange(tmin, tmax, dt)

    def forward(self, t, y):
        beta, gamma = self.params

        S, I, R, N = y

        dSdt = - (beta / N) * I * S
        dIdt = (beta / N) * I * S - gamma * I
        dRdt = gamma * I
        dNdt = tc.tensor(0.0)

        return tc.stack([dSdt, dIdt, dRdt, dNdt])


        
