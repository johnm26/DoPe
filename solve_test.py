'''
Module to test solvers.
'''
import pylab
import numpy as np

from solve import EulerSolver, RungeKuttaSolver

N = 100
DT = 0.1

def testEuler():
    f = lambda x : x[1]
    g = lambda x : -x[0]
    x0 = np.array([0.0, 1.0])
    s = EulerSolver([f, g], x0)
    soln = np.zeros(N)
    t = np.arange(N) * DT
    exact = np.sin(t)
    
    for i in range(N):
        soln[i] = s.vars()[0]
        s.step(DT)
    
    pylab.figure()
    pylab.title(r"Euler Integration Test: $\Delta t = %s$" % DT)
    pylab.plot(t, soln, label = "Numerical")
    pylab.plot(t, exact, label = "Analytic")
    pylab.legend(loc = "lower left")

def testRungeKutta():
    f = lambda x : x[1]
    g = lambda x : -x[0]
    x0 = np.array([0.0, 1.0])
    s = RungeKuttaSolver([f, g], x0)
    soln = np.zeros(N)
    t = np.arange(N) * DT
    exact = np.sin(t)
    
    for i in range(N):
        soln[i] = s.vars()[0]
        s.step(DT)
    
    pylab.figure()
    pylab.title(r"Runge-Kutta Integration Test: $\Delta t = %s$" % DT)
    pylab.plot(t, soln, label = "Numerical")
    pylab.plot(t, exact, label = "Analytic")
    pylab.legend(loc = "lower left")

if __name__ == "__main__":
    testEuler()
    testRungeKutta()
    pylab.show()