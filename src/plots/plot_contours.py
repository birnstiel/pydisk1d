#!/usr/bin/env python
"""
Compare the TW Hya results with a drift limited distribution
"""
from numpy import * #@UnusedWildImport
from constants import * #@UnusedWildImport
from uTILities import dlydlx
from matplotlib.pyplot import * #@UnusedWildImport
from matplotlib.mlab import find
import h5py
import os

files = [
#        '~/Desktop/new_results/sean/deadzone-paper/active_iter1/data_iter1_fixeddisk_AL22_M20_161.mat',
        '~/Desktop/new_results/sean/deadzone-paper/active_iter1/data_iter1_fixeddisk_AL33_M20_161.mat',
        '~/Desktop/new_results/sean/deadzone-paper/active_iter1/data_iter1_fixeddisk_AL44_M20_161.mat',
        '~/Desktop/new_results/sean/deadzone-paper/active_iter1/data_iter1_fixeddisk_AL55_M20_161.mat'
        ]
TIME = 5e6*year
#
# loop through the files and get the necessary data
#
SD = []
R  = []
A  = []
AF = []
AD = []
AM = []
MX = -Inf
for filename in files:
    #
    # load the matlab file
    #
    f = h5py.File(os.path.expanduser(filename),'r')
    sigma_g      = f['sigma_g_1'][...].transpose()
    sigma_d_1    = f['sigma_d_1'][...].transpose()
    alpha        = f['alpha_1'][...].transpose()
    T            = f['T_1'][...].transpose()
    v_dust       = f['v_dust_1'][...].transpose()
    timesteps    = f['timesteps_1'][...].transpose()
    x            = f['x_1'][...].flatten()
    a            = f['grainsizes_1'][...].flatten()
    m            = f['m_grid_1'][...].flatten()
    V_FRAG       = f['V_FRAG'][...].flatten()
    T_COAG_START = f['T_COAG_START'][...].flatten()
    RHO_S        = f['RHO_S'][...].flatten()
    m_star       = f['m_star_1'][...].flatten()
    grains       = size(a)
    a_0          = a[0]
    n_r          = size(x)
    it           = find(timesteps>=TIME)[0]
    E_drift      = 1.
    #
    # calculate the size limits
    #
    sigma_d   = sum(sigma_d_1[it*grains+arange(grains),:],0)
    fudge_fr = 0.37
    fudge_dr = 0.55
    gamma = dlydlx(x,sigma_g[it])+0.5*dlydlx(x,T[it])-1.5
    a_fr  = fudge_fr*2*sigma_g[it,:]*V_FRAG**2./(3*pi*alpha[it]*RHO_S*k_b*T[it]/mu/m_p)
    a_dr  = fudge_dr/E_drift*2/pi*sigma_d/RHO_S*x**2.*(Grav*m_star[it]/x**3)/(abs(gamma)*(k_b*T[it]/mu/m_p))
    N     = 0.5
    a_df  = fudge_fr*2*sigma_g[it]/(RHO_S*pi)*V_FRAG*sqrt(Grav*m_star[it]/x)/(abs(gamma)*k_b*T[it]/mu/m_p*(1.-N))
    mask  = a_dr<a_fr;
    a_max = maximum(a_0*ones([1,n_r]),minimum(a_dr,a_fr))
    a_max_out = minimum(a_dr,a_fr)
    a_max = maximum(a_0*ones([1,n_r]),minimum(a_df,a_max))
    a_max_out = minimum(a_df,a_max)
    mask  = (a_dr<a_fr) & (a_dr<a_df)
    #
    # calculate the growth time scale and thus a_1(t)
    #
    o_k      = sqrt(Grav*m_star[it]/x**3)
    tau_grow = sigma_g[it]/(sigma_d*o_k)
    a_max_t      = minimum(a_max,a_0*exp((TIME-T_COAG_START)/tau_grow))
    a_max_t_out  = maximum(a_max_out,a_0*exp((TIME-T_COAG_START)/tau_grow))
    #
    # save only the necessary arrays
    #
    SD += [sigma_d_1[it*grains+arange(grains),:]]
    R  += [x]
    A  += [a]
    AF += [a_fr]
    AD += [a_dr]
    AM += [a_max_t_out]
    MX =  max(MX,sigma_d_1[it*grains+arange(grains),:].max())
#
# plot it
#
mx = ceil(log10(MX))
sc = 6.
fs = 20 
fig=figure(facecolor='none',figsize=[sc,sc*len(files)*3./5.])
hold(True)
for iax in arange(len(files)):
    x = R[iax]
    a = A[iax]
    S = SD[iax]
    a_fr = AF[iax]
    a_dr = AD[iax]
    
    ax=subplot(len(files),1,iax+1)
    contourf(x/AU,a,log10(S),arange(mx-10,mx+1))
    xscale('log')
    yscale('log')
    L1,=loglog(x/AU,a_fr,'k--')
    L2,=loglog(x/AU,a_dr,linestyle='-',color='r')
    xlim([2e-1,3e2])
    ylim([a[0],a[-1]])
    
    yt=get(ax,'yticks')
    yticks(10**arange(log10(yt[0]),log10(yt[-1])+1,2))
    if iax==0:
        leg=legend([L1,L2],["fragmentation limit","drift limit"])
        leg.get_frame().set_color('none')
    if iax==len(files)-1:
        ax.set_xlabel(r'$r\,\mathrm{[AU]}$')
    else:
        ax.set_xticklabels([])
    #
    # other formating
    #
    ax.set_ylabel(r'$\mathrm{grain\,size\,[cm]}$')
    ax.set_axis_bgcolor('none')
#
# set font size
#
font = {'size'   : fs}
matplotlib.rc('font', **font)
setp(leg.get_texts(),fontsize=9)
#
# draw color bar
#
pos=ax.get_position()
cb=colorbar(shrink=0.8)
ax.set_position([pos.x0,pos.y0,pos.size[0],pos.size[1]])
tx = array([])
for t in cb.ax.get_yticklabels():
    t.set_fontsize(12)
    tx = append(tx,'$10^{'+t.get_text().replace(u'\u2212','-')+'}$')
cb.ax.set_yticklabels(tx)
cb.ax.set_title('$\sigma(r,a) \mathrm{[g\,cm}^{-2}\mathrm{]}$',fontsize=10)
#
# update
#
gcf().tight_layout()
#
# save the figure
#
savefig('plot.pdf', facecolor=fig.get_facecolor(), edgecolor='none')
