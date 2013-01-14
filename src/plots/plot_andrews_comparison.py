#!/usr/bin/env python
"""
Compare the TW Hya results with a drift limited distribution
"""
from numpy import * #@UnusedWildImport
from constants import * #@UnusedWildImport
from uTILities import dlydlx
from matplotlib.pyplot import * #@UnusedWildImport
from matplotlib.mlab import find
#
# set the default colors
#
mycol = 'w'
rc('lines', linewidth=2, color=mycol)
rc('text',  color=mycol)
rc('axes',  edgecolor=mycol, linewidth=2, labelcolor=mycol)
rc('xtick', color=mycol)
rc('ytick', color=mycol)
#
# reproduce Sean's best fits
#
x_1 = logspace(-1,3,200)*AU
om    = sqrt(Grav*0.8*M_sun/x_1**3)
rc    = 35*AU
sigg0 = 0.29
d2g   = 0.014
sig_g = 1./d2g*sigg0*(x_1/rc)**(-1)*exp(-(x_1/rc))
r0    = 60.*AU
sigd0 = 0.39
sig_d = (x_1/r0)**(-0.75)
sig_d = sig_d/interp(10*AU,x_1,sig_d)*sigd0
sig_d[x_1>r0] = 1e-300
sig_d[x_1<=4*AU] = 1e-300
#
# derive the drift estimate
#
ir = find(x_1/AU>=10)[0]
sig_dr = sqrt(sig_g/(x_1**2.*sqrt(Grav*0.8*M_sun/x_1**3)))
sig_dr = sig_dr/sig_dr[ir]*sig_d[ir]
#
# the fragmentation estimate
#
f_d    = 0.55;
m_dot  = 2*pi*x_1**2*f_d*om*sig_dr**2/sig_g
m_dot  = mean(m_dot)
gamma  = abs(dlydlx(x_1,sig_g)-1/4-3/2)
V_FRAG = 1000
f_f    = 0.37
sig_fr = 3*m_dot/(2*pi)*1e-3*om/(f_f*gamma*V_FRAG**2)
#
# plot it
#
fig=figure(facecolor='none')
hold(True)
loglog(x_1/AU,sig_g,mycol)
loglog(x_1/AU,sig_d,mycol,linestyle='--')
#
# plot everythin or just the same range
#
#mask=array(x_1<r0) & array(x_1>=4*AU)
mask=arange(len(x_1))
loglog(x_1[mask]/AU,sig_dr[mask],linestyle='-',color='r')

#xlim([1e0,3e2])
xlim([4,2e2])
ylim([2e-3,2e3])
xlabel(r'$r\,\mathrm{[AU]}$')
ylabel(r'$\Sigma\,\mathrm{[g\,cm}^{-2}\mathrm{]}$')
leg=legend((r'$\Sigma_\mathrm{g}$ from (CO $J$ 3-2)',r'$\Sigma_\mathrm{d}$ (from 870 $\mu$m)',r'drift-limited estimate'),numpoints=1)

setp(gca().get_legend().get_texts(),fontsize='small')
gca().set_axis_bgcolor('none')

font = {'size'   : 20}

matplotlib.rc('font', **font)
leg.get_frame().set_color('none')
#
# print it
#
gcf().tight_layout()
fig.savefig('plot.pdf', facecolor=fig.get_facecolor(), edgecolor='none')
