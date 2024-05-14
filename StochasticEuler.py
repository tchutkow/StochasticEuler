import matplotlib.pyplot as plt
import random
import math

# N(0,1)
def N():
    return random.normalvariate(0,1)

# stochastic variation of Euler method for solving SDEs
# returns a list of t values and a list of y values that describe the SDE over an interval
def Euler(h,y0,t0,tn):
    w, t = [y0], [t0]
    for i in range(1,int(tn/h)+1):
        w.append(w[-1] + f(t[-1],w[-1])*h + g(t[-1],w[-1])*(N()*math.sqrt(h)))
        t.append(t[-1] + h)
    return t, w

# drift term
# describes the deterministic part of SDE
def f(t,y):
    r = 0.05
    return r * y

# diffusion term
# describes random part of SDE
def g(t,y):
    sigma = 0.35
    return sigma * y

# Monte Carlo function
# simulates X(T) using Euler-Maruyama to calculate max{XT-K,0}
# performs 10000 simulations and takes the average to get E[max{XT-K,0}]
def E():
    nTrials, out, K = 10000, 0, 15
    for _ in range(nTrials):
        (t,y) = Euler(0.001,12,0,0.5)
        XT = y[-1]
        out += max([XT-K,0])
    return out / nTrials

# call price formula
# C(X,T) = e^{-tT} * E[max{XT-K,0}]
def C(T):
    r = 0.05
    return math.exp(-r*T)*E()

# prints the solution to example stock question
print(C(0.5))


# for plotting purposes
# returns graph showing 10 simulated realizations of the trajectory of stock price over the given interval
for _ in range(10):
    (t,y) = Euler(0.001,12,0,0.5)
    plt.plot(t,y)

plt.title("Monte Carlo Realizations of Stock Price")
plt.xlabel("Time (years)")
plt.ylabel("Stock Price (USD)")
plt.grid()
plt.show()