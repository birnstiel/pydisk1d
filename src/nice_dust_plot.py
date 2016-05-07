#!/usr/bin/env python
# coding: utf-8
"""
Script to plot snapshots of the dust evolution code (Birnstiel et al. 2010)
"""

def plot(d, time, sizelimits=True, justdrift=True, stokesaxis=False, usefudgefactors=True,
    fluxplot=False, colormap='viridis', xlim=None, ylim=None, zlim=None, v_frag=None, ncont=20, outfile=None):
    """
    Plot a snapshots of the dust evolution code (Birnstiel et al. 2010).
    
    Arugments:
    ---------
    
    d : pydisk1D instance
    :   the simulation data
    
    time : float
    :   time of the snapshots in seconds
    
    Keywords:
    ---------
    
    justdrift : bool
    :   if true, remove the gas velocity from the drift speed (for flux plot)
    
    stokesaxis : bool
    :   whether or not to plot the y-axis as stokes number or particle size
    
    sizelimits : bool
    :   whether or not to overplot the size limits
    
    usefudgefactors  : bool
    :   whether or not to include the fudge factors in the size limits
    
    fluxplot : bool
    :   whether or not to plot the flux density instead of the density
    
    colormap : string
    :   name of the colormap
    
    [x,y,z]lim : 2 element array like
    :   the limits
        
    v_frag : [None|np.array|function]
    :   None: use constant from code, otherwise provide 1D, 2D array or function
        that takes the time index as only argument
        
    ncont : int
    :   number of contour levels
    
    outfile : [str|None]
    :   if None: no output
        if '': create name automatically
        other string: use this name
    """
        
    import brewer2mpl, warnings
    import numpy               as np
    import matplotlib.pyplot   as plt
    import matplotlib.gridspec as gridspec
    
    from matplotlib.colors import LogNorm
    from matplotlib.pyplot import rcParams, cycler
    from matplotlib        import ticker
    from uTILities         import get_St, dlydlx, num2tex
    from constants         import AU, year, M_earth, pi, Grav, mu, m_p, k_b, sig_h2
    
    # Plotting environment
    
    fs              = 18
    lw              = 2
    lc              = brewer2mpl.get_map('Dark2', 'Qualitative', 8).mpl_colors
    #lc              = [[1,0,0],[0,0.5,0],[0,0,1]]
    front           = 'k'
    back            = 'w'
    
    rcParams['figure.facecolor']            = back
    rcParams['axes.edgecolor']              = front
    rcParams['axes.facecolor']              = front
    rcParams['axes.linewidth']              = 1.5*lw
    rcParams['axes.labelcolor']             = front
    rcParams['axes.prop_cycle']             = cycler('color',lc)
    rcParams['axes.formatter.use_mathtext'] = True
    rcParams['xtick.color']                 = front
    rcParams['ytick.color']                 = front
    rcParams['xtick.major.size']            = 6*lw
    rcParams['ytick.major.size']            = 6*lw
    rcParams['ytick.major.width']           = 1*lw
    rcParams['xtick.major.width']           = 1*lw
    rcParams['xtick.minor.size']            = 3*lw
    rcParams['ytick.minor.size']            = 3*lw
    rcParams['ytick.minor.width']           = 0.75*lw
    rcParams['xtick.minor.width']           = 0.75*lw
    rcParams['lines.linewidth']             = 1.5*lw
    rcParams['image.cmap']                  = colormap
    rcParams['font.size']                   = fs
    rcParams['text.color']                  = front
    rcParams['savefig.facecolor']           = back
    rcParams['mathtext.fontset']            = 'stix'
    rcParams['font.family']                 = 'STIXGeneral'
    rcParams['text.usetex']                 = True
    
    it        = d.timesteps.searchsorted(time)
    
    # Define a fragmentation velocity function
    
    v_frag_in = v_frag
    if v_frag_in is None:
        v_frag_in = d.nml['V_FRAG']*np.ones(d.n_r)
        v_frag = lambda N: v_frag_in
    elif hasattr(v_frag, '__call__'):
        pass
    else:
        v_frag_in = np.asarray(v_frag)
        if v_frag_in.ndim==1:
            v_frag = lambda N: v_frag_in
        elif v_frag_in.ndim==2:
            v_frag = lambda N: v_frag_in[N]
        else:
            raise ValueError('could not translate v_frag into a function, use float, 1D array, 2D array or function.')
    
    # Define fudge factors
    
    if usefudgefactors:
        fudge_fr = 0.37
        fudge_dr = 0.55
    else:
        fudge_fr = 1.
        fudge_dr = 1.
    
    # Complicated part to get the size limits.
    
    if sizelimits==True:
        RHO_S     = d.nml['RHO_S']
        N         = it
        sigma_d   = d.get_sigma_dust_total()[N,:]
        gamma     = dlydlx(d.x,d.sigma_g[N])+0.5*dlydlx(d.x,d.T[N])-1.5
        #
        # the standard fomula with the fudge factor
        #
        #a_fr  = fudge_fr*2*self.sigma_g[N,:]*v_frag(N)**2./(3*pi*self.alpha[N]*RHO_S*k_b*self.T[N]/mu/m_p)
        #
        # calculate limits in terms of stokes numbers
        #
        om    = np.sqrt(Grav*d.m_star[N]/d.x**3)
        cs    = np.sqrt(k_b*d.T[N]/mu/m_p)
        H     = cs/om
        n     = d.sigma_g[N]/(np.sqrt(2*pi)*H*m_p)
        mfp   = 0.5/(sig_h2*n)
        #
        # fragmentation limit in terms of Stokes number
        #
        b       = 3.*d.alpha[N]*cs**2/v_frag(N)**2
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
            lim_fr  = fudge_fr*0.5*(b-np.sqrt(b**2-4.))
    
        if stokesaxis:
            #
            # St = 1 as Stokes number
            #
            lim_St1    = np.ones(d.n_r)
            #
            # drift limit as Stokes number
            #
            lim_dr_e   = fudge_dr*sigma_d/d.sigma_g[N]*(d.x*om/cs)**2/np.abs(gamma) # St_drift in Epstein
            lim_dr_St1 = fudge_dr*1./3.*sigma_d**2/d.sigma_g[N]/mfp*RHO_S*(d.x*om/cs)**4/np.abs(gamma)**2 # St_drift in Stokes 1 regime
            if d.nml['STOKES_REGIME']==1:
                lim_dr = np.maximum(lim_dr_e,lim_dr_St1)
            else:
                lim_dr = lim_dr_e
        else:
            #
            # stokes number of unity in grain size
            #
            St_e   = 2.*d.sigma_g[N]/(pi*RHO_S)           # Stokes number = 1 in Epstein regime
            St_St1 = (9.*d.sigma_g[N]*mfp/(2.*pi*RHO_S))  # Stokes number = 1 in first Stokes regime
            if d.nml['STOKES_REGIME']==1:
                lim_St1 = np.minimum(St_e,St_St1)
            else:
                lim_St1 = St_e
            #
            # convert fragmentation limit from Stokes number to particle size
            #
            lim_fr_e    = 2.*d.sigma_g[N]/(pi*RHO_S)*lim_fr
            lim_fr_St1  = (9.*lim_fr*d.sigma_g[N]*mfp/(2.*pi*RHO_S) )**0.5
            if d.nml['STOKES_REGIME']==1:
                lim_fr = np.minimum(lim_fr_e,lim_fr_St1)
            else:
                lim_fr = lim_fr_e
            #
            # convert drift limit from Stokes number to particle size
            #
            lim_dr   = fudge_dr/(d.nml['DRIFT_FUDGE_FACTOR']+1e-20)*2/pi*sigma_d/RHO_S*d.x**2.*(Grav*d.m_star[N]/d.x**3)/(np.abs(gamma)*cs**2)
    
    # calculate the stokes number if needed
    
    if fluxplot or stokesaxis:    
        St       = np.zeros([d.n_m,d.n_r])
        for ir in range(d.n_r):
            St[:,ir] = get_St(d.grainsizes, d.T[it,ir], d.sigma_g[it,ir], d.x[ir], d.m_star[it], rho_s=d.nml['RHO_S'],Stokesregime=d.nml['STOKES_REGIME'],fix_error=False)
            
    # choose the axes
            
    if stokesaxis:
        X,_ = np.meshgrid(d.x,d.grainsizes)
        Y   = St
    else:
        X = d.x
        Y = d.grainsizes
        
    # Get the Y-data -- flux or density.
    
    if fluxplot:
        Z = 2*pi*d.x*d.sigma_d[d.n_m*it+np.arange(d.n_m),:]*(d.v_dust[d.n_m*it+np.arange(d.n_m),:]-d.v_gas[it,:]/(1.+St**2)*justdrift)
        Z = Z/M_earth*year
    else:
        Z = d.sigma_d[d.n_m*it+np.arange(d.n_m),:]
        
    gsf=np.log(d.grainsizes[1]/d.grainsizes[0])
    Z = Z/gsf
    
    # choose the plotting limits

    if xlim is None: xlim = d.x[[0,-1]]
    if ylim is None:
        if stokesaxis:
            ylim = [1e-5,2]
        else:
            ylim = d.grainsizes[[0,-1]]
    
    if zlim is None: zlim = np.log10(np.array([1e-10,1])*abs(Z).max())
        
    # plotting
    
    f   = plt.figure(figsize=(9,6))
    gs  = gridspec.GridSpec(1, 2,width_ratios=[20,1])
    ax  = plt.subplot(gs[0])
    cax = plt.subplot(gs[1])
    #gs.update(wspace=0.15)
    
    c1 = ax.contourf(X/AU,Y,(abs(Z+1e-200)), np.logspace(zlim[0],zlim[1],ncont),norm=LogNorm())
    
    ax.plot(d.x/AU,lim_St1,'-',label='$\mathrm{St}=1$')
    ax.plot(d.x/AU,lim_fr, '-',label='$a_\mathrm{frag}$')
    ax.plot(d.x/AU,lim_dr, '-',label='$a_\mathrm{drift}$')
    
    if fluxplot: ax.contour(X/AU,Y,Z,0,colors='w',linestyles='--')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(ylim)
    ax.set_xlabel('$r$ $[\mathrm{AU}]$');
    ax.set_axis_bgcolor(plt.cm.get_cmap(colormap).colors[0])
    
    leg=ax.legend(loc='upper right',prop={'size':fs},handlelength=2)
    for t in leg.get_texts(): t.set_color('w') 
    leg.get_frame().set_alpha(0)
    
    ax.text(0.5, 0.95,num2tex(d.timesteps[it]/year,2,2)+' $\mathrm{years}$', horizontalalignment='center',
        verticalalignment='top',transform=ax.transAxes,color='k',bbox=dict(facecolor='white', alpha=0.6))
    
    cb = plt.colorbar(c1,cax=cax,ax=ax)
    cb.ax.set_axis_bgcolor('none')
    cb.solids.set_edgecolor('none')
    cb.solids.set_linewidth(0)
    cb.solids.set_antialiased(True)
    cb.patch.set_visible(False)
    #cb.locator = ticker.MaxNLocator(nbins=7)
    cb.locator = ticker.LogLocator()
    if fluxplot:
        if justdrift:
            cb.set_label('$2\pi r\Sigma_\mathrm{d}(r,a)v_\mathrm{drift}$ $[M_\oplus\,\mathrm{yr}^{-1}]$') 
        else:
            #cb.set_label('$2\pi r\Sigma_\mathrm{d}(r,a)v_\mathrm{d}$ $[M_\oplus\,\mathrm{yr}^{-1}]$') # Christians version
            cb.set_label('$a \cdot \dot M (r,a)$ [$M_\oplus$ yr$^{-1}$]')
    else:
        cb.set_label('$a \cdot \Sigma(r,a)$ [g cm$^{-2}$]')
    cb.update_ticks()
    
    if stokesaxis:
        ax.set_ylabel('$\mathrm{Stokes number}$')
    else:
        ax.set_ylabel('$\mathrm{particle}$ $\mathrm{size}$ $[\mathrm{cm}]$')
        
    f.tight_layout()
    
    if outfile is not None:
        if outfile == '':
            fname = '{}_{:2.2f}Myr.pdf'.format(d.data_dir,time/1e6/year)
        else:
            fname = outfile
        f.savefig(fname)
    
if __name__ == '__main__':
    from pydisk1D          import pydisk1D
    import argparse
    from constants import year
    
    RTHF   = argparse.RawTextHelpFormatter
    PARSER = argparse.ArgumentParser(description=__doc__,formatter_class=RTHF)
    PARSER.add_argument('inputfile',               help='input data file or folder',type=str)
    PARSER.add_argument('time',                    help='time of the snapshot [yr]',type=float)
    PARSER.add_argument('-j', '--justdrift',       help='ignore the gas velocity',  action='store_true',default=False)
    PARSER.add_argument('-st','--stokesaxis',      help='turn on stokes axis',      action='store_true',default=False)
    PARSER.add_argument('-s', '--sizelimits',      help='over plot size limits',    action='store_true',default=True)
    PARSER.add_argument('-ff', '--usefudgefactors',help='ignore the gas velocity',  action='store_true',default=True)
    PARSER.add_argument('-f',  '--fluxplot',       help='plot flux density',        action='store_true',default=False)
    PARSER.add_argument('-xl', '--xlim',           help='x limits [AU]',            nargs=2, type=int,  default=None)
    PARSER.add_argument('-yl', '--ylim',           help='y limits [AU]',            nargs=2, type=int,  default=None)
    PARSER.add_argument('-zl', '--zlim',           help='z limits [AU]',            nargs=2, type=int,  default=None)
    PARSER.add_argument('-c',  '--colormap',       help='name of the color map',    type=str,           default='viridis')
    PARSER.add_argument('-n',  '--ncont',          help='number of contour levels', type=int,           default=20)
    PARSER.add_argument('-o',  '--outfile',        help='name of output file',      type=str,           default='')
    ARGS  = PARSER.parse_args()
    
    #d=pydisk1D('/Users/birnstiel/DATA/sean/disklifetime/initial_runs/data_disklifetime_alpha-3_MD005Msun_RD200_VF1000_STR0_q04_g1_189.hdf5')
    d=pydisk1D(ARGS.inputfile)
    
    plot(d, ARGS.time*year, outfile=ARGS.outfile, sizelimits=ARGS.sizelimits, justdrift=ARGS.justdrift, stokesaxis=ARGS.stokesaxis, usefudgefactors=ARGS.usefudgefactors,
        fluxplot=ARGS.fluxplot, colormap=ARGS.colormap, xlim=ARGS.xlim, ylim=ARGS.ylim, zlim=ARGS.zlim, v_frag=None, ncont=ARGS.ncont)
