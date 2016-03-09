import numpy as np
import pylab

from solve import RungeKuttaSolver
import params as p

rm = p.m2 / (p.m1 + p.m2)
rl = p.l2 / p.l1

def C(args):
    th, ph, thd, phd = args
    return 1.0 / (1.0 - rm)

def THD(args):
    th, ph, thd, phd = args
    return thd

def PHD(args):
    th, ph, thd, phd = args
    return phd

def THDD(args):
    th, ph, thd, phd = args
    t1 = -rl * th
    t2 = rl * rm * ph
    return C(args) * (t1 + t2)

def PHDD(args):
    th, ph, thd, phd = args
    t1 = th
    t2 = -ph
    return C(args) * (t1 + t2)

x0 = np.array([p.t0, p.p0, p.td0, p.pd0])
funcs = [THD, PHD, THDD, PHDD]
solver = RungeKuttaSolver(funcs, x0)

if __name__ == "__main__":
    th = []
    ph = []
    thd = []
    phd = []
    
    for i in range(p.n):
        vars = solver.vars()
        th.append(vars[0])
        ph.append(vars[1])
        thd.append(vars[2])
        phd.append(vars[3])
        
        solver.step(p.dt)
    
    t = p.dt * np.arange(p.n)
    
    fig = pylab.figure()
    ax1 = fig.add_subplot(411)
    ax2 = fig.add_subplot(412, sharex = ax1)
    ax3 = fig.add_subplot(413, sharex = ax2)
    ax4 = fig.add_subplot(414, sharex = ax3)
    ax1.plot(t, th)
    ax2.plot(t, ph)
    ax3.plot(t, thd)
    ax4.plot(t, phd)
    
    pylab.show()