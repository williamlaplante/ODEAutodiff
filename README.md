# ODEAutodiff
A project that aims at fitting ODE parameters to data by differentiating through a likelihood that is a function of an ODE solver. 


To do:
1 ) Use an RNN, Decoder, and regular network to model the data with an ODE net. i.e. if $x'(t) = f(x(t);u)$, then $f$ is represented by a neural net.

2) Model $\beta (t)$ using a neural network (instead of a fourier series). See if it improves performance.

3) Try with different models (other than SIR). Create a models.py file to keep track of different models. Create a simulate.py file to simulate noisy data for the other models (unless there is real data to use).

