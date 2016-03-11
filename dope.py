import numpy as np
import pylab
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

from solve import RungeKuttaSolver
import params as p

rm = p.m2 / (p.m1 + p.m2)
rl = p.l2 / p.l1

def C(args):
    th, ph, thd, phd = np.transpose(args)
    return 1.0 / (1.0 - rm * np.cos(th - ph)**2)

def THD(args):
    th, ph, thd, phd = np.transpose(args)
    return thd

def PHD(args):
    th, ph, thd, phd = np.transpose(args)
    return phd

def THDD(args):
    th, ph, thd, phd = np.transpose(args)
    t1 = -rl * rm * np.sin(th - ph) * phd**2
    t2 = -rm / 2.0 * np.sin(2 * (th - ph)) * thd**2
    t3 = rl * rm * np.sin(ph) * np.cos(th - ph)
    t4 = -rl * np.sin(th)
    return C(args) * (t1 + t2 + t3 + t4)

def PHDD(args):
    th, ph, thd, phd = np.transpose(args)
    t1 = 1.0 / rl * np.sin(th - ph) * thd**2
    t2 = rm / 2.0 * np.sin(2 * (th - ph)) * phd**2
    t3 = np.sin(th) * np.cos(th - ph)
    t4 = -np.sin(ph)
    return C(args) * (t1 + t2 + t3 + t4)

x0 = np.array([p.t0, p.p0, p.td0, p.pd0])
funcs = [THD, PHD, THDD, PHDD]
solver = RungeKuttaSolver(funcs, x0)

# Class to facilitate double pendulum simulation as well as animating several
# panels at once.
# And yes.. it is dope.
class DopeSimulation(object):
    funcs = None
    solver = None
    dt = None
    nsteps = None
    
    def __init__(self, animationFuncs, dt = p.dt, nsteps = p.n, x0 = x0):
        self.funcs = animationFuncs
        self.solver = RungeKuttaSolver(funcs, x0)
        self.dt = dt
        self.nsteps = nsteps
    
    def __call__(self, i):
        # Reset the animation if we have passed the number of steps we
        # originally set out to simulate.
        if self.nsteps != None and i % self.nsteps == 0:
            return self.reset()
        
        self.solver.step(self.dt)
        currentVariables = self.solver.vars()
        
        objectsToDraw = []
        for f in self.funcs:
            newObject = f(i, currentVariables)
            if newObject != None:
                try:
                    for obj in newObject:
                        objectsToDraw.append(obj)
                except:
                    objectsToDraw.append(newObject)
            
        return objectsToDraw
    
    def reset(self):
        self.solver.reset()
        initVariables = self.solver.vars()
        
        objectsToDraw = []
        for f in self.funcs:
            newObject = f.reset(initVariables)
            if newObject != None:
                try:
                    for obj in newObject:
                        objectsToDraw.append(obj)
                except:
                    objectsToDraw.append(newObject)
            
        return objectsToDraw

# For a point (theta, phi) on the torus, return the x, y, and z coordinates of
# that point.
def mapToTorus(theta, phi):
    x = (2 + np.cos(theta)) * np.cos(phi)
    y = (2 + np.cos(theta)) * np.sin(phi)
    z = np.sin(theta)
    
    return x, y, z

def getCartesian(theta, phi):
    goodRatio = float(p.l1) / (p.l1 + p.l2)
    l1 = goodRatio
    l2 = (1.0 - goodRatio)
    
    x1 = l1 * np.sin(theta)
    y1 = -l1 * np.cos(theta)
    x2 = x1 + l2 * np.sin(phi)
    y2 = y1 - l2 * np.cos(phi)
    
    return x1, y1, x2, y2

# Plot a 3d torus on the axis ax.
def torus3d(ax, elev = None, azim = None):
    theta = np.linspace(0, 2 * np.pi, num = 30)
    phi = np.linspace(0, 2 * np.pi, num = 30)
    
    t, p = np.meshgrid(theta, phi)
    x, y, z = mapToTorus(t, p)
    
    ax.plot_surface(x, y, z, rstride = 1, cstride = 1, linewidth = 0.2, color = 'c', alpha = 0.25)
    ax.set_xlim3d(-3.5, 3.5)
    ax.set_ylim3d(-3.5, 3.5)
    ax.set_zlim3d(-3.5, 3.5)
    ax.tick_params(which = 'both', axis = 'both', \
        labelbottom = 'off', labeltop = 'off', labelleft = 'off', labelright = 'off')
    ax.set_axis_off()
    ax.set_title(r"Phase Space Projection in $(\theta, \phi)$")
    
    if elev != None and azim != None:
        ax.view_init(elev = elev, azim = azim)
    elif elev != None:
        ax.view_init(elev = elev)
    elif azim != None:
        ax.view_init(azim = azim)

def pendSetup(ax, titleHigh = False):
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_axis_off()
    ax.tick_params(which = 'both', axis = 'both', \
        labelbottom = 'off', labeltop = 'off', labelleft = 'off', labelright = 'off', \
        top = 'off', bottom = 'off', left = 'off', right = 'off')
    if titleHigh:
        ax.set_title("Physical Space")
    else:
        ax.text(0, 1.1, "Physical Space", ha = 'center', va = 'center', fontsize = 14)

class TorAnimate(object):
    allTh = None
    allPh = None
    allX = None
    allY = None
    allZ = None
    line = None
    
    # This object must be initialized with the line it will draw onto at each
    # animation step.
    def __init__(self, line):
        self.allTh = []
        self.allPh = []
        self.allX = []
        self.allY = []
        self.allZ = []
        self.line = line
    
    def __call__(self, i, stepVariables):
        th, ph = stepVariables[:2]
        self.allTh.append(th)
        self.allPh.append(ph)
        
        x, y, z = mapToTorus(th, ph)
        self.allX.append(x)
        self.allY.append(y)
        self.allZ.append(z)
        
        self.line.set_data(self.allX, self.allY)
        self.line.set_3d_properties(self.allZ)
        
        return self.line
    
    def reset(self, initVariables):
        self.allTh = []
        self.allPh = []
        self.allX = []
        self.allY = []
        self.allZ = []
        
        self.line.set_data([], [])
        self.line.set_3d_properties(self.allZ)
        
        return self.line
        # return self(0, initVariables)

class PendulumAnimate(object):
    line = None
    number = None
    
    def __init__(self, line, pendNumber = None):
        self.line = line
        self.number = pendNumber
        
    def __call__(self, i, stepVariables):
        if self.number != None:
            th, ph = stepVariables[self.number, :2]
        else:
            th, ph = stepVariables[:2]
        x1, y1, x2, y2 = getCartesian(th, ph)
        
        self.line.set_data([0.0, x1, x2], [0.0, y1, y2])
        
        return self.line
    
    def reset(self, initVariables):
        self.line.set_data([], [])
        return self.line

class PendulumAnimateTrajectory(object):
    allX1 = None
    allX2 = None
    allY1 = None
    allY2 = None
    number = None
    line = None
    
    def __init__(self, line, pendNumber = None):
        self.allX1 = []
        self.allX2 = []
        self.allY1 = []
        self.allY2 = []
        self.number = pendNumber
        self.line = line
        
    def __call__(self, i, stepVariables):
        if self.number != None:
            th, ph = stepVariables[self.number, :2]
        else:
            th, ph = stepVariables[:2]
        x1, y1, x2, y2 = getCartesian(th, ph)
        self.allX1.append(x1)
        self.allX2.append(x2)
        self.allY1.append(y1)
        self.allY2.append(y2)
        
        self.line.set_data(self.allX2, self.allY2)
        
        return self.line
    
    def reset(self, initVariables):
        self.allX1 = []
        self.allX2 = []
        self.allY1 = []
        self.allY2 = []
        
        self.line.set_data([], [])
        return self.line
        # return self(0, initVariables)

class TorusAdvancer(object):
    ax = None
    period = None
    
    def __init__(self, ax, period = p.n):
        self.ax = ax
        self.period = period

class EnsembleAnimate(object):
    line = None
    
    def __init__(self, scatterObject):
        self.line = scatterObject
    
    def __call__(self, i, stepVariables):
        th, ph = np.transpose(stepVariables[:, :2])
        x, y, z = mapToTorus(th, ph)
        
        self.line.set_data(x, y)
        self.line.set_3d_properties(z)

        return self.line
    
    def reset(self, initVariables):
        self.line.set_data([], [])
        self.line.set_3d_properties([])
        
        return self.line

class PhaseDiffAnimator(object):
    line = None
    itNums = []
    allData = None
    dt = None
    
    def __init__(self, line, dt = p.dt):
        self.line = line
        self.itNums = []
        self.allData = []
        self.dt = dt
        self.itCount = 0
        
    def __call__(self, i, stepVariables):
        th1, ph1, thd1, phd1 = stepVariables[0, :]
        th2, ph2, thd2, phd2 = stepVariables[1, :]
        
        self.itCount += 1
        diff = np.sqrt((th2 - th1)**2 + (ph2 - ph1)**2 + (thd2 - thd1)**2 + (phd2 - phd1)**2)
        self.allData.append(np.log(diff))
        self.itNums.append(self.itCount)
        self.line.set_data(self.dt * np.array(self.itNums), self.allData)
        
        return self.line
    
    def reset(self, initVariables):
        self.itNums = []
        self.allData = []
        self.line.set_data(self.itNums, self.allData)
        self.itCount = 0
        
        return self.line

class TorusViewChanger(object):
    itCount = None
    period = None
    ax = None
    
    def __init__(self, ax, period = p.n):
        self.period = p.n
        self.itCount = 0
        self.ax = ax
    
    def __call__(self, i, stepVariables):
        phase = float(self.itCount) / self.period
        self.ax.view_init(azim = phase * 360.0)
        
        self.itCount += 1
        
    def reset(self, initVariables):
        self.itCount = 0
        self.ax.view_init(azim = 0.0)

class AngleAnimator(object):
    thline = None
    phline = None
    allIt = None
    allTh = None
    allPh = None
    itNum = None
    dt = None
    
    def __init__(self, thetaLine, phiLine, dt = p.dt):
        self.thline = thetaLine
        self.phline = phiLine
        self.allIt = []
        self.allTh = []
        self.allPh = []
        self.itNum = 0
        self.dt = dt
    
    def __call__(self, i, stepVariables):
        th, ph = stepVariables[:2]
        self.allTh.append(th)
        self.allPh.append(ph)
        self.allIt.append(self.itNum)
        self.itNum += 1
        
        self.thline.set_data(self.dt * np.array(self.allIt), self.allTh)
        self.phline.set_data(self.dt * np.array(self.allIt), self.allPh)
        
        return self.thline, self.phline
    
    def reset(self, initVariables):
        self.allIt = []
        self.allTh = []
        self.allPh = []
        self.itNum = 0
        
        self.thline.set_data([], [])
        self.phline.set_data([], [])
        
        return self.thline, self.phline

def initEnsemble(dt, dp, t0, p0, N):
    th = np.linspace(-dt / 2, dt / 2, num = N) + t0
    ph = np.linspace(-dp / 2, dp / 2, num = N) + p0
    th, ph = np.meshgrid(th, ph)
    return th.ravel(), ph.ravel()

def plotPendulumOnTorus(animate = True, nsteps = p.n):
    # Plot the pendulum and torus side by side.
    fig = pylab.figure()
    if animate:
        ax1 = pylab.subplot2grid((2, 4), (0, 0), rowspan = 2, colspan = 2)
        ax2 = pylab.subplot2grid((2, 4), (0, 2), rowspan = 2, colspan = 2, projection = "3d")
    else:
        ax1 = fig.gca()
        fig2 = pylab.figure()
        ax2 = fig2.gca(projection = "3d")
        
    pendSetup(ax1)
    line1, = ax1.plot([], [], 'k', linestyle = '-', marker = 'o', \
        markeredgecolor = 'k', markerfacecolor = (0.5, 0.5, 1.0), markersize = 10)
    line1.set_markevery((1, 1))
    line1b, = ax1.plot([], [], 'm--', animated = True)
    
    torus3d(ax2, elev = 30.0, azim = 30.0)
    line2, = ax2.plot([], [], [], 'k-')
    
    pendAnimator = PendulumAnimate(line1)
    pendulumTrackAnimator = PendulumAnimateTrajectory(line1b)
    torAnimator = TorAnimate(line2)
    torRotator = TorusViewChanger(ax2)
    simulation = DopeSimulation([pendAnimator, torAnimator, \
        pendulumTrackAnimator, torRotator], nsteps = nsteps)
    
    if animate:
        ax1.set_axis_on()
        simulation.reset()
        anim = animation.FuncAnimation(fig, simulation, \
            blit = True, init_func = simulation.reset, \
            interval = 10, repeat = True)
        
        pylab.suptitle("Runge-Kutta Pendulum Simulation", fontsize = 20)
        pylab.tight_layout(3.0)
        pylab.show()
    else:
        for i in range(nsteps):
            simulation(i)
        
        ax1.set_axis_off()
        fig.savefig("OnePendulumEvolution.png")
        fig2.savefig("OnePendulumOnTorus.png")

# Plot multiple pendulum initial conditions on the torus.
def plotPendulumEnsemble(animate = True, saveEvery = 100, nsteps = p.n):
    fig = pylab.figure()
    ax = pylab.gca(projection = "3d")
    torus3d(ax, elev = 30.0, azim = 0.0)
    line, = ax.plot([], [], [], 'ro')
    t0, p0 = initEnsemble(p.blobdt, p.blobdp, p.blobt0, p.blobp0, p.blobnt)
    td0 = p.blobtd0 * np.ones(t0.shape)
    pd0 = p.blobpd0 * np.ones(p0.shape)
    initialState = np.transpose(np.vstack( (t0, p0, td0, pd0) ))
    simulation = DopeSimulation([EnsembleAnimate(line)], x0 = initialState, \
        nsteps = nsteps)
    
    if animate:
        anim = animation.FuncAnimation(fig, simulation, blit = True, \
              interval = 10, repeat = True)
        
        pylab.show()
    else:
        ax.set_title("")
        for i in range(nsteps):
            simulation(i)
            if i % saveEvery == 1:
                pylab.savefig("ensembleOnTorus_step%s.png" % str(i).zfill(4))
    
def plotLyapunovDivergence(animate = True, lyapSteps = 200, \
    t0 = np.pi, p0 = 1 * np.pi / 180.0, td0 = 0.0, pd0 = 0.0):
    
    fig = pylab.figure()
    if animate:
        ax1 = pylab.subplot2grid((1, 2), (0, 1), aspect = 1.0) # Pendula
        ax2 = pylab.subplot2grid((1, 2), (0, 0)) # System difference
    else:
        ax1 = fig.gca(aspect = 1.0)
        fig2 = pylab.figure()
        ax2 = fig2.gca()
    
    pendSetup(ax1, titleHigh = True)
    ax1.set_axis_on()
    line1, = ax1.plot([], [], 'k', linestyle = '-', marker = 'o', \
        markeredgecolor = 'k', markerfacecolor = (0.5, 0.5, 1.0), markersize = 10)
    line1.set_markevery((1, 1))
    line2, = ax1.plot([], [], 'k', linestyle = '-', marker = 'o', \
        mec = 'k', mfc = 'g', markersize = 10)
    line1b, = ax1.plot([], [], linestyle = "--", color = (0.5, 0.5, 1.0), animated = True)
    line2b, = ax1.plot([], [], linestyle = "--", color = 'g', animated = True)
    
    line3, = ax2.plot([], [], 'k-')
    ax2.set_xlim((0, lyapSteps * p.dt))
    ax2.set_ylim((-6, 3))
    ax2.set_xlabel(r"simulation time $\tau$", fontsize = 14)
    ax2.set_ylabel(r"phase space distance $\log(\delta(t))$", fontsize = 14)
    
    pendAnimator1 = PendulumAnimate(line1, pendNumber = 0)
    pendAnimator2 = PendulumAnimate(line2, pendNumber = 1)
    pendulumTrackAnimator1 = PendulumAnimateTrajectory(line1b, pendNumber = 0)
    pendulumTrackAnimator2 = PendulumAnimateTrajectory(line2b, pendNumber = 1)
    phaseDiffAnimator = PhaseDiffAnimator(line3)
    initial2 = x0 + np.array([0, 0.1, 0, 0]) # np.array([p.t0, p.p0 + 0.01, p.td0, p.pd0])
    initial = np.vstack( (x0, initial2) )
    
    simulation = DopeSimulation([pendAnimator1, pendAnimator2, \
        pendulumTrackAnimator1, pendulumTrackAnimator2, phaseDiffAnimator], \
        x0 = initial, nsteps = lyapSteps)
    
    if animate:
        anim = animation.FuncAnimation(fig, simulation, \
            blit = True, init_func = simulation.reset, \
            interval = 10, repeat = True)
        
        simulation.reset()
        
    for i in range(lyapSteps):
        simulation(i)
        
    x, y = line3.get_data()
    coeff = np.polyfit(x, y, 1)
    ax2.plot(x, np.polyval(coeff, x), 'r-', label = "slope = %.3f" % coeff[0])
    pylab.legend(loc = 'lower right')
    pylab.tight_layout()
    
    if animate:
        pylab.savefig("2PendulumLyapunov.png")
        pylab.show()
    else:
        fig.savefig("pendula_2PendulumLyapunov.png")
        fig2.savefig("phasediff_2PendulumLyapunov.png")

def plotPendulumWithThetaAndPhi(animate = True, \
    title = "PendulumNormalModes.png", t0 = p.t0, p0 = p.p0, nsteps = 120, massRat = rm):
    
    global rm
    old = rm
    rm = massRat
    
    fig = pylab.figure()
    if animate:
        ax1 = pylab.subplot2grid((1, 2), (0, 1), aspect = 1.0) # Pendulum
        ax2 = pylab.subplot2grid((1, 2), (0, 0)) # Theta/Phi plots
    else:
        ax1 = fig.gca(aspect = 1.0)
        fig2 = pylab.figure()
        ax2 = fig2.gca()
    
    pendSetup(ax1, titleHigh = True)
    ax1.set_axis_on()
    line1, = ax1.plot([], [], 'k', linestyle = '-', marker = 'o', \
        markeredgecolor = 'k', markerfacecolor = (0.5, 0.5, 1.0), markersize = 10)
    line1.set_markevery((1, 1))
    
    line2, = ax2.plot([], [], 'b-', label = r"$\theta$")
    line3, = ax2.plot([], [], 'r-', label = r"$\phi$")
    ax2.set_xlim((0, nsteps * p.dt))
    ax2.set_ylim((-30 * np.pi / 180.0, 30 * np.pi / 180))
    ax2.set_xlabel(r"simulation time $\tau$", fontsize = 14)
    ax2.set_ylabel("pendula angles", fontsize = 14)
    
    pendAnimator1 = PendulumAnimate(line1)
    angleAnimator = AngleAnimator(line2, line3)
    
    simulation = DopeSimulation([pendAnimator1, angleAnimator], \
        x0 = np.array([t0, p0, 0.0, 0.0]), nsteps = nsteps)
    
    if animate:
        anim = animation.FuncAnimation(fig, simulation, \
            blit = True, init_func = simulation.reset, \
            interval = 10, repeat = True)
        
        simulation.reset()
    
    for i in range(nsteps):
        simulation(i)
        
    pylab.legend(loc = 'lower right')
    pylab.tight_layout()
    
    if animate:
        pylab.show()
    else:
        fig.savefig("pendulum_" + title)
        fig2.savefig("angles_" + title)
    
    rm = old

if __name__ == "__main__":
    # These function calls generate the plots in our report.
    
    plotPendulumOnTorus(animate = False, nsteps = 200)
    inPhaseT0 = 10 * np.pi / 180.0
    inPhaseP0 = np.sqrt(2) * inPhaseT0
    outPhaseT0 = inPhaseT0
    outPhaseP0 = -np.sqrt(2) * outPhaseT0
    plotPendulumWithThetaAndPhi(animate = False, title = "InPhaseMode.png", \
        t0 = inPhaseT0, p0 = inPhaseP0, nsteps = 125)
    plotPendulumWithThetaAndPhi(animate = False, title = "OutPhaseMode.png", \
        t0 = outPhaseT0, p0 = outPhaseP0, nsteps = 91)
    plotPendulumWithThetaAndPhi(animate = False, title = "m2Zero.png", \
        t0 = inPhaseT0, p0 = inPhaseP0, nsteps = 90, massRat = 0.0)
    plotLyapunovDivergence(animate = False)
    plotPendulumEnsemble(animate = False, nsteps = 500)
    
    # Uncomment one at a time to view the animations shown during our
    # 03/09/2016 presentation.
    pylab.close("all")
    plotPendulumOnTorus(animate = True, nsteps = 200)
    # plotPendulumWithThetaAndPhi(animate = True, title = "InPhaseMode.png", \
    #    t0 = inPhaseT0, p0 = inPhaseP0, nsteps = 125)
    # plotPendulumWithThetaAndPhi(animate = True, title = "OutPhaseMode.png", \
    #    t0 = outPhaseT0, p0 = outPhaseP0, nsteps = 91)
    # plotPendulumWithThetaAndPhi(animate = True, title = "m2Zero.png", \
    #    t0 = inPhaseT0, p0 = inPhaseP0, nsteps = 90, massRat = 0.0)
    # plotLyapunovDivergence(animate = True)
    # plotPendulumEnsemble(animate = True, nsteps = 500)