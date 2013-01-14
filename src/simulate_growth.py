#!/usr/bin/env python
from pydisk1D import pydisk1D
import os
from relative_velocities import relative_velocities_cont
from numpy import * #@UnusedWildImport
from matplotlib.pyplot import * #@UnusedWildImport
from constants import AU,year,k_b,mu,m_p,Grav
from uTILities import progress_bar
from matplotlib.mlab import find

def v_rel_eq(x,data,ir,it,delta_a,BM=True,TM=True,VS=True):
    """
    this calculates the "equal size particle relative velocity" of a given
    array of grainsizes x
    
    data      = see function relative_velocities_cont
      ir      = radius index
      it      = radius index
      x       = grain radii
      delta_a = x1/x2
      
    Keywords:
    ---------
    BM  = turn on Brownian motion relative velocities
    TM  = turn on turbulent motion relative velocities
    VS  = turn on vertical settling relative velocities
    """ 
    v1,_,_,_ = relative_velocities_cont(data,ir,it,x,x*delta_a,BM=BM,TM=TM,VS=VS)
    x = array(x,ndmin=1)
    v  = zeros(len(x))
    for i in arange(len(x)):
        v[i] = v1[i,i]
    return v


sim_data = '~/Desktop/new_results/sean/deadzone-paper/referee_runs/data_referee1_nofrag_nodrift_fixeddisk_AL33_M20_201'
d = pydisk1D(os.path.expanduser(sim_data))
#
# get data from simulation
#
x           = d.x 
n_t         = d.n_t
n_m         = d.n_m
sigma_d     = d.sigma_d
sigma_g     = d.sigma_g
grainsizes  = d.grainsizes
timesteps   = d.timesteps
alpha       = d.alpha
m_grid      = d.m_grid
T           = d.T
m_star      = d.m_star
RHO_S       = d.nml['RHO_S']
CONST_ALPHA = d.nml['CONST_ALPHA']
VREL_BM     = d.nml['VREL_BM']
VREL_TM     = d.nml['VREL_TV']
VREL_VS     = d.nml['VREL_VS']
#
# set positions
#
ir     = find(d.x>10*AU)[0]
it     = 1
R_plot = x[ir]/AU
#
# find the position of the maximum
#
dslice  = zeros([n_t,n_m])
for i in arange(n_t):
    dslice[i,:]=sigma_d[i*n_m+arange(n_m),ir]
max_val = zeros(n_t)
max_pos = zeros(n_t)
for i in arange(n_t):
    max_val[i] = dslice[i,:].max()
    max_pos[i] = dslice[i,:].argmax()

ia    = 0
#a     = zeros(n_t)
#a[0]  = grainsizes[ia]
sig_d = sum(sigma_d[it*n_m+arange(n_m),ir])
if VREL_BM==1: BM = True
if VREL_TM==1: TM = True
if VREL_VS==1: VS = True
#
# define functions
#
H_g      = sqrt(k_b*T[0,ir]/mu/m_p)/sqrt(Grav*m_star[it]/x[ir]**3)
St       = lambda x: x*RHO_S/sigma_g[0,ir]*pi/2
h_d      = lambda x: H_g*minimum(1,sqrt(CONST_ALPHA/(min(0.5,St(x))*(1+St(x)))))
rho_d    = lambda x: sig_d/(2*sqrt(pi)*h_d(x))
dv       = lambda x: v_rel_eq(x,d,ir,it,1.4,BM=BM,TM=TM,VS=VS)[0]
f        = lambda x,t: rho_d(x)/RHO_S*dv(x)
#
# RK4 integration
#
# initial conditions
t0 = timesteps[0]
t1 = timesteps[-1]
y0 = 1e-4
#
# define number of snapshot times
#
n_times = 400
times   = logspace(log10(t0),log10(t1),n_times)

# iteration
t       = t0
y       = y0
counter = 1
i       = 0
j       = 1
stepF   = 500.
N       = log10(t1/t0)/log10(1.+1./stepF)
t_out   = zeros(n_times)
y_out   = zeros(n_times)
while t<t1:
    i  = i+1
    h  = t/stepF
    k1 = h*f(y,t)
    k2 = h*f(y+1/2*k1,t+1/2*h)
    k3 = h*f(y+1/2*k2,t+1/2*h)
    k4 = h*f(y+k3,t+h)
    y  = y + 1./6.*k1 + 1./3.*k2 + 1./3.*k3 + 1./6.*k4
    t  = t+h
    if i>j*N/100:
        progress_bar(i/N*100,'RK4 integration')
        j=j+1
    if t>times[counter]:
        t_out[counter] = t
        y_out[counter] = y
        counter = counter + 1
#
# plot it
#
peak_sizes = array([grainsizes[i] for i in max_pos])
loglog(timesteps/year,peak_sizes,label='simulation')
loglog(t_out/year,y_out,'r--',label='monodisperse')
xlabel('t [yr]')
ylabel('a [cm]')
legend(loc=0)
xlim(timesteps[0]/year,timesteps[-1]/year)