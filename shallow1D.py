#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn
import sys

# -------- SET UP SIMULATION PARAMETERS --------
L = 1000                        # 1000m 
n = 200                         # Number of simulation points
g = 9.81                        # m/s^2
h0 = 10                         # 10m water
dx = L / n                      # dx = 1m

CFL_target = 0.2
dt = CFL_target*dx/np.sqrt(g*(h0+2))   # Choose largest time step that satisfies the CFL-condition to CFL_target<1

# -------- SET UP UNITS --------
xs = np.linspace(0,L,n)
plot_interval   = int(np.ceil( 0.5/dt))  # Update plot every 0.5 simulation second
plot_resolution = int(np.ceil(  10/dx))  # Plot points at 1m intervals

# Shapiro filter parameter
epsilon = 0.002*np.sqrt(n) #0.0015
dtype=np.float64

# -------- ALLOCATE MEMORY --------
u          = np.zeros(n,dtype=dtype)
eta        = np.zeros(n,dtype=dtype)
eta_star   = np.zeros(n,dtype=dtype)
eta_smooth = np.zeros(n,dtype=dtype)
h          = np.zeros(n,dtype=dtype)
u_x        = np.zeros(n-2,dtype=dtype)
eta_t      = np.zeros(n-2,dtype=dtype)


# -------- VARIOUS INITIAL CONDITIONS --------
#Square dam break
def square_dam_break(dam_width,x0=L/2,height=1):
    drop_start, drop_end = x0/dx-dam_width/(2*dx), x0/dx+dam_width/(2*dx)
    eta[int(round(drop_start)):int(round(drop_end))] = 1

#Gaussian dam break
def smooth_dam_break(width,x0=L/2,height=1):
    xs = np.linspace(0,L,n)
    eta[:] += height*np.exp(-(xs-x0)**2/(2*width**2))

def wave_paddle(width,x0=L/2,t=0,period=100,amplitude=1):
    paddle_start, paddle_end = x0/dx-width/(2*dx), x0/dx+width/(2*dx)
    eta[int(round(paddle_start)):int(round(paddle_end))] = amplitude*np.sin(t*2*np.pi/period)

    

# -------- THE ACTUAL SHALLOW WATER SIMULATION--------
# Staggered grid first-order central finite difference
def Dx0(f,dx): return (1/dx)*(f[2:]-f[1:-1])
def Dx1(f,dx): return (1/dx)*(f[1:-1]-f[:-2])

# We can write it in a single function, so that we can call it more neatly.
def Dx(f,dx,grid=0):
    n = len(f)
    return (1/dx)*(f[2-grid:n-grid]-f[1-grid:-1-grid])

# -- STRAIGHT FORWARD IMPLEMENTATION --
# u_t   = -g eta_x
# eta_t = - d/dx (u*h) = -u*h_x -u_x*h
def simulation_step_naive():
    # OMB Eq. (4.12)/(4.17):
    # u_t = -g eta_x
    eta_x    =  Dx0(eta,dx);
    u[1:-1] += -g*eta_x*dt      # Forward-Euler time integration

    # Velocity boundary-conditions
    u[:1]  =  0
    u[-2:] =  0
    
    # OMB Eq. (4.13)
    h        = h0+eta
    h_x      = Dx0(h,dx)
    u_x      = Dx1(u,dx)
    eta_t    = -u[1:-1]*h_x - u_x*h[1:-1];
    
    eta[1:-1]  += eta_t*dt      # Forward-Euler time integration

    # Height boundary-conditions
    eta[0]  = eta[1]
    eta[-1] = eta[-2]
    
    # First order Shapiro filter, OMB Eq. (4.21)
    eta[1:-1]  = (1-epsilon)*eta[1:-1] + epsilon*0.5 * (eta[2:] + eta[:-2])
 
    
# -------- PLOT FUNCTION --------
def plot_update(frames):
    global t,t0

    for j in range(plot_interval):  # Run plot_interval simulation steps between each plot
        simulation_step_naive()
# Uncomment the next line to place a wave-generating paddle, waving for 100 seconds
#        if(t<100): wave_paddle(40,period=43,amplitude=0.3,t=t,x0=L/2)
        t += dt
    
    if (t-t0>10): # Check water volume every 10 simulation seconds
        t0 = t
        print(f"at {round(t,2)} seconds, water volume in eta is {round(dx*np.sum(eta[1:-1]),8)}")

    water_plot.set_ydata(eta[::plot_resolution])
    ax.set_title(f"time: {round(t)}s")



# -------- INITIALIZE AND RUN THE SIMULATION + PLOTS --------    
#square_dam_break(100,x0=L/2)
smooth_dam_break(25,x0=200,height=1)
smooth_dam_break(25,x0=500,height=1)
smooth_dam_break(25,x0=700,height=1)
smooth_dam_break(25,x0=900,height=1)


#smooth_dam_break(25,x0=L/2,height=1)
# CFL condition

print(f"epsilon={epsilon}\n"
      f"dt     ={dt}\n"
      f"dx     ={dx}\n"
      f"CFL    ={(dt/dx)*np.sqrt(g*(h0+eta.max()))}\n"
      f"time steps per plot update={plot_interval}\n"
      f"steps per plot point={plot_resolution}\n"
)


t,t0 = 0,0     # Global time


seaborn.set(style='ticks')
fig, ax = plt.subplots()
water_plot, = ax.plot(xs[::plot_resolution],eta[::plot_resolution], '-')
#ax.set_ylim((-h0,2))
ax.set_ylim((-2,2))
ax.grid(True, which='both')
title  = ax.set_title("time: 0s")
ani = FuncAnimation(fig, plot_update, interval=5)

plt.show()
