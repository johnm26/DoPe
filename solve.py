import abc

import numpy as np

class Solver(object):
    __metaclass__ = abc.ABCMeta
    
    f  = None
    x  = None
    t  = None
    t0 = None
    x0 = None
    
    def __init__(self, fs, x0):
        self.load(fs)
        self.set_init(x0)
        self.reset()
    
    # fs is a list of n functions of n arguments [f1, ..., fn] defining
    # the derivatives of the n dependent variables in terms of their current
    # values. E.g. for the 3 variable case, fs would be
    # 
    # [f(x, y, z), g(x, y, z), h(x, y, z)]
    # specifying dx/dt = f(x, y, z), dy/dt = g(x, y, z), dz/dt = h(x, y, z)
    # 
    # These functions must take these arguments as a single list and parse
    # them internally.
    # 
    # The solver should internalizes these functions and erases its present
    # state, effectively becoming the new system defined by the 'fs'.
    def load(self, fs):
        self.x = None
        self.t = None
        self.t0 = None
        self.x0 = None
        self.f = [fcn for fcn in fs]
    
    # Set initial conditions (t = 0 values)
    def set_init(self, x0):
        self.t0 = 0.0
        self.x0 = np.array([x for x in x0])
    
    # Reset the state of the solver to the initial conditions
    def reset(self):
        if self.x0 == None:
            raise ValueError("No initial conditions to reset to.")
        self.x = self.x0.copy()
        self.t = self.t0
    
    # Returns the current values of the dependent variables
    def vars(self):
        return self.x.copy()
    
    # Return current derivative values
    def derivs(self, x = None):
        if x == None:
            return np.array([fcn(self.x) for fcn in self.f])
        else:
            return np.array([fcn(x) for fcn in self.f])
    
    # Return the current solution time
    def now(self):
        return self.t
    
    # Particular implementations need implement only this one function :-)
    @abc.abstractmethod
    def step(self, dt):
        """Given the current solution state t, advance to state at t + dt"""
        return

class EulerSolver(Solver):
    
    # Simple Euler step.
    def step(self, dt):
        self.x += dt * self.derivs()
        self.t += dt
        return self.vars()

class RungeKuttaSolver(Solver):
    
    # 4th-order Runge-Kutta step. Formula taken from Strogatz 2nd Ed. pg 148
    def step(self, dt):
        x = self.vars()
        k1 = dt * self.derivs(x)
        k2 = dt * self.derivs(x + 0.5 * k1)
        k3 = dt * self.derivs(x + 0.5 * k2)
        k4 = dt * self.derivs(x + k3)
        self.x += 1. / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return self.vars()