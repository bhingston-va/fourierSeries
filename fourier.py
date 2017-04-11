#!/usr/bin/python
import sys
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
#from typing import overload

# global vars
N = 1       # initializing sum-to value
L = np.pi   # length of L
aCoef = []  # all a_1 to a_n coefficients
bCoef = []  # all b_1 to b_n coefficients
debugMode = False
runF1 = True

x = sp.Symbol('x')

'''
param n: argument index for sys.argv
args -N: partial sum limit, default 1
args -f: run hardcoded function f(x) or f2(x), default f(x)
args -d: prints contents of a_n's b_n's and a_0, default False
'''
def CheckArg(n, errMsg):
    global N, runF1, debugMode

    # partial sum limit
    if sys.argv[n] == '-N':
        N = int(sys.argv[n+1])
    # which function to run
    elif sys.argv[n] == '-f':
        if int(sys.argv[n+1]) == 1: runF1 = True
        else: runF1 = False
    # debug on/off
    elif sys.argv[n] == '-d':
        if sys.argv[n+1] == 'T': debugMode = True
        else: debugMode = False
    else:
        print(errMsg)
        sys.exit()
    return

usageMsg = "usage: python fourier.py [-N num] [-f 1/2] [-d T/F]"
# make sure the right amount of args is given before continuing
# TODO: args: script.py, f(x), N, L
# too many args
if len(sys.argv) > 7:
    print(usageMsg)
    sys.exit()
# one arg without param
elif len(sys.argv) == 2:
    CheckArg(1,usageMsg)
# one arg with param
elif len(sys.argv) == 3:
    CheckArg(1,usageMsg)
# two args
elif len(sys.argv) == 5:
    CheckArg(1,usageMsg)
    CheckArg(3,usageMsg)
# three args
elif len(sys.argv) == 7:
    CheckArg(1,usageMsg)
    CheckArg(3,usageMsg)
    CheckArg(5,usageMsg)


# needed for symbolic function calculations
def pos1(x): return 1
def neg1(x): return -1
def pos2(x): return 0.5
def neg2(x): return -2


#TODO: give a function as an argument instead of hardcoding
'''
functions Fouier Series is being applied to.
'''
def f(x):
    if(x >= 0): return 1
    else: return -1

def f2(x):
    if  (x >= 1.0): return 0.5
    elif(x >= 0 and x < 1.0): return 1
    elif(x < 0 and x >= -1.0): return -1
    else: return -2


'''
multivariable cosine function
param x: symbolic variable x
param n: nth term, n=1,2,...,N
returns: cos(n*pi*x/L)
'''
def MCos(x,n):
    return np.cos((n * np.pi * x) / L)


'''
multivariable sine function
param x: symbolic variable x
param n: nth term, n=1,2,...,N
returns: sin(n*pi*x/L)
'''
def MSin(x,n):
    return np.sin((n * np.pi * x) / L)


'''
multiples single var functions f(x)*g(x) = integrand F
param f,g: single var functions
param a,b: evaluate integral from a to b
returns: evaluated integral of F from a to b
'''
#@overload
def CalcIntegral(f,g,a,b,n=None):
    if n is None:
        F = lambda x : f(x) * g(x)
    else:
        F = lambda x : f(x) * g(x,n)
    ans, err = quad(F,a,b)
    return ans


'''
multiples functions f(x)*g(x,n) = integrand F
param f: single var function
param g: multiple var function
param a,b: evaluate integral from a to b
returns: evaluated integral of F from a to b
'''
#@CalcIntegral.overload
'''
def CalcIntegral(f,g,a,b,n):
    F = lambda x : f(x) * g(x,n)
    ans, err = quad(F,a,b)
    return ans
'''


'''
param x: symbolic variable x
param a_0: a_0 coefficient of Fourier Series
param a_n: all a_i coefficients of Fourier Series, i from 1 to N
param b_n: all b_i coefficients of Fourier Series, i from 1 to N
returns: partial sum of fourier series, approx of original function
f(x) ~ a_0 + sum [a_n*cos(n*pi*x/L) + b_n*sin(n*pi*x/L)], n = 1toN
'''
def FourierSeries(x, a_0, a_n, b_n):
    n = len(a_n)    # sizeOf(a_n) == sizeOf(b_n)
    sum = a_0
    for i in range(0,n):
        sum += a_n[i] * MCos(x,i+1)
        sum += b_n[i] * MSin(x,i+1)
    return sum



#MAIN

# a_0 = (1/2L)integral of f(x) from -L to L)
# f(x):
if runF1 is True:
    a_0=(1.0/(2.0*L)) *(CalcIntegral(f,pos1,0,L)
                      + CalcIntegral(f,pos1,-L,0))
# f2(x):
else:
    a_0=(1.0/(2.0*L)) *(CalcIntegral(f2,pos1, 0, 1.0)
                      + CalcIntegral(f2,pos1, 1.0, L)
                      + CalcIntegral(f2,pos1,-L, -1.0)
                      + CalcIntegral(f2,pos1,-1.0, 0))

if debugMode:
    print("value of a0:%d" % a_0)


# split piecewise f into pos, neg halves to evaluate integral
if runF1 is True:
    for n in range(1,N+1):
        aNeg = (1.0/L) * CalcIntegral(f,MCos, -L , 0.0, n)
        aPos = (1.0/L) * CalcIntegral(f,MCos, 0.0,  L , n)
        bNeg = (1.0/L) * CalcIntegral(f,MSin, -L , 0.0, n)
        bPos = (1.0/L) * CalcIntegral(f,MSin, 0.0,  L , n)
        aCoef.append(aNeg + aPos)
        bCoef.append(bNeg + bPos)
else:
    for n in range(1,N+1):
        aNeg2 = (1.0/L) * CalcIntegral(f2,MCos, -L,  -1.0, n)
        aNeg1 = (1.0/L) * CalcIntegral(f2,MCos, -1.0, 0.0, n)
        aPos1 = (1.0/L) * CalcIntegral(f2,MCos,  0.0, 1.0, n)
        aPos2 = (1.0/L) * CalcIntegral(f2,MCos,  1.0,  L , n)

        bNeg2 = (1.0/L) * CalcIntegral(f2,MSin, -L,  -1.0, n)
        bNeg1 = (1.0/L) * CalcIntegral(f2,MSin, -1.0, 0.0, n)
        bPos1 = (1.0/L) * CalcIntegral(f2,MSin,  0.0, 1.0, n)
        bPos2 = (1.0/L) * CalcIntegral(f2,MSin,  1.0,  L , n)

        aCoef.append(aNeg2+aNeg1+aPos1+aPos2)
        bCoef.append(bNeg2+bNeg1+bPos1+bPos2)

if debugMode:
    print(aCoef)
    print(bCoef)

# seting up x and y values for plotting
x = np.arange(-np.pi, np.pi, 0.01)

y1 = FourierSeries(x,a_0,aCoef,bCoef) # Fourier partial sum
y2 = np.sin(x)                        # for comparison only

# ploting original function and partal sum approximation
plt.plot(x, y1, '-g', label="Fouier N=" + str(N))
plt.plot(x, y2, '-r', label="Sine")
if runF1 is True: plt.plot(x, list(map(f, x)), '-b', label="f (x)")
else: plt.plot(x, list(map(f2, x)), '-b', label="f (x)")
plt.legend(loc='upper left')

# setting x,y axis limits for f(x), f2(x)
plt.xlim(-np.pi, np.pi)
if runF1 is True: plt.ylim(-1.5, 2)
else: plt.ylim(-2.5, 2.5)
plt.show()
