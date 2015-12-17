from numpy import array,append,arange,trapz,pi,zeros,ceil,sqrt,log,log10,minimum,\
meshgrid,savetxt,isnan,interp,loadtxt,random,ndarray,sum
import matplotlib,h5py,re,glob,os,sys
from uTILities import parse_nml, dlydlx,write_nml,progress_bar,my_colorbar
from constants import AU,year, k_b, m_p, mu,Grav, sig_h2
from matplotlib.mlab import find
from scipy.interpolate import RectBivariateSpline

class pydisk1D:
    """
    Class to read, load, save and plot the simulation
    results of the diskev code by Til Birnstiel.
    
    For questions and comments please write
    to tbirnstiel@cfa.harvard.edu

    Example:
        * Shortest way to get the data (data_dir is directory or file
            >>> D=pydisk1D(data_dir)
        * Define a simulation data object:
            >>> D = pydisk1D()
            no data directory given, will not read data
        * Read from a data folder
            >>> D.read_diskev('data/')
            Reading from directory data/
        * Save the data to hdf5 file
            >>> D.save_diskev('data')
        * Load the data from a hdf5 file
            >>> D.load_diskev('data.mat')
            >>> D.load_diskev('data.hdf5')
    """

    def __init__(self,data_dir=""):
        """ 
        The initialization routine. Without argument, nothing happens.
        With argument, a directory or file is already loaded.
        
        Arguments:
        	data_dir	can be either a file or a directory. The
        				simulation name is parsed from the input
        				and if a hdf5 file with the same name as a
        				folder exists,  the file is preferred.
        """
        self.data_dir            = data_dir
        self.AU                  = None
        self.m_grid              = None
        self.sig_dot_t           = None
        self.v_dust              = None
        self.D_grain1            = None
        self.dust_flux           = None
        self.m_star              = None
        self.sigma_coag          = None
        self.v_gas               = None
        self.T                   = None
        self.dust_flux_o         = None
        self.grainsizes          = None
        self.n_m                 = None
        self.sigma_d             = None
        self.v_gas_dead          = None
        self.accretion_dust      = None
        self.dust_flux_o_e       = None
        self.kappa_p             = None
        self.n_r                 = None
        self.sigma_dead          = None
        self.x                   = None
        self.accretion_dust_e    = None
        self.fallen_disk_mass    = None
        self.kappa_p_r           = None
        self.n_t                 = None
        self.r_centri            = None
        self.sigma_g             = None
        self.x05                 = None
        self.accretion_gas       = None
        self.flim                = None
        self.kappa_r             = None
        self.nml                 = None
        self.r_min               = None
        self.steps               = None
        self.year                = None
        self.alpha               = None
        self.flim_dead           = None
        self.kappa_r_r           = None
        self.nu                  = None
        self.r_snow              = None
        self.alpha_dead          = None
        self.gas_flux_o          = None
        self.peak_position       = None
        self.timesteps           = None
        self.d_evap              = None
        self.m_dot_star          = None
        self.ttable              = None
        self.stored_data         = None

        if data_dir=='':
            print "no data directory given, will not read data"
        else:
            #
            # first check if the folder is there 
            #
            folder_exist = os.path.isdir(data_dir)
            #
            # second, check if file exists
            #
            filename = data_dir
            if filename[-1]=='/':
                filename=filename[0:len(filename)+1]
            filename=filename.replace('.hdf5','')
            filename=filename.replace('.mat','')
            h_file_exist = os.path.isfile(filename+".hdf5")
            m_file_exist = os.path.isfile(filename+".mat")
            #
            # now read the file if it exists,
            # otherwise read from the folder if that exists,
            # otherwise throw an error that none exists
            #
            if h_file_exist or m_file_exist:
                if folder_exist:
                    print "preferring to read from file instead of folder"
                if h_file_exist:
                    self.load_diskev(filename+".hdf5")
                elif m_file_exist:
                    self.load_diskev(filename+".mat")
            elif folder_exist:
                self.read_diskev(data_dir)
            else:
                raise Exception('Input error', 'Neither file nor folder to read data was found!') 
        #
        # now set some constants
        #
        self.AU   = AU
        self.year = year

    def __add__(self,b):
        """
        Method to concatenate simulations. Will not return new object, but
        add the second to the first one, i.e. `sim1+sim2` will result in all
        the `sim2` data being appended to the respective fields of `sim1`.
        
        Arguments:
        ----------
        
        b: pydisk1D object
        :    the data to be added to the current object
        
        """
        #
        # compare the nml
        #
        merge_keys = ['T_0','INTERVAL','M_STAR_INITIAL','T_COAG_START']
        for key,val in self.nml.iteritems():
            if val!=b.nml[key] and key not in merge_keys:
                print('WARNING: nml entries for \'%s\' differ!'%key)
        #
        # merge specific keys
        #
        self.nml['INTERVAL'] += 1
        #
        # redefine data_dir
        #
        self.data_dir+='+%s'%b.data_dir
        #
        # redefine number of snapshots
        #
        self.n_t=self.n_t+b.n_t
        #
        # check if those agree
        #
        eqkeys = [
            'm_grid',
            'n_m',
            'ttable',
            'n_r',
            'grainsizes',
            'x',
            'x05',
            'kappa_p',
            'kappa_r',
            'kappa_r_r',
            'kappa_p_r']
        for k in eqkeys:
            if any(array(getattr(self,k)!=getattr(self,k),ndmin=1)):
                print('WARNING: key \'%s\' is not identical'%k)
        #
        # merge the 0D arrays
        #
        addkeys0D = [
            'fallen_disk_mass',
            'm_dot_star',
            'steps',
            'accretion_dust',
            'accretion_dust_e',
            'gas_flux_o',
            'timesteps',
            'm_star',
            'r_centri',
            'r_min',
            'accretion_gas',
            'r_snow',
            'd_evap',
            'peak_position',
            'dust_flux',
            'dust_flux_o',
            'dust_flux_o_e']
        for k in addkeys0D:
            setattr(self, k, append(getattr(self,k),getattr(b,k)))
        #
        # merge the 1D and 2D arrays
        #
        addkeys1D = [
            'sigma_g',
            'D_grain1',
            'T',
            'flim',
            'flim_dead',
            'alpha',
            'v_gas',
            'alpha_dead',
            'v_gas_dead',
            'nu',
            'sig_dot_t',
            'sigma_dead']
        addkeys2D = ['v_dust','sigma_d']
        for k in addkeys1D+addkeys2D:
            setattr(self, k, append(getattr(self,k),getattr(b,k),0))

    def sigma_d_movie(self,i0=0,i1=-1,steps=1,dpi=None,**kwargs):
        """
        This uses the other sub-routine `plot_sigma_d` to produce
        a movie of the time evolution.
        
        Keywords:
        ---------
        
        i0 : int
        : index of the first snapshot
        
        i1 : int
        : index of the last snapshot
        
        steps : int
        : step size between each frame

        dpi : int
        : resolution of the image frames
        
        Examples:
        ---------
        
            d.plot_sigma_d(N=-1,cmap=get_cmap('YlGnBu_r'),clevel=arange(-9,2),ax_color='w',bg_color='k',xl=[0.3,500],fs=14)
        
        The **kwargs are passed to `plot_sigma_d
        """
        from matplotlib.pyplot import figure,savefig,clf,close,rcParams
        from random import choice
        from string import ascii_letters
        import subprocess
        if i0<0: i0=0
        if i1>self.n_t or i1==-1: i1 = self.n_t - 1
        dpi = dpi or rcParams['figure.dpi']
        #
        # check if there is already a movie_images folder
        # otherwise create one
        #
        dirname    = 'movie_images_'+''.join(choice(ascii_letters) for x in range(5))  # @UnusedVariable
        if os.path.isdir(dirname):
            print('%s folder already exists, please delete it first'%dirname)
            sys.exit(-1)
        else:
            os.mkdir(dirname)
        #
        # make the images
        #
        fig=figure()
        for i,i_s in enumerate(arange(i0,i1+1,steps)):
            self.plot_sigma_d(i_s,fig=fig,**kwargs)
            savefig(dirname+os.sep+'img_%3.3i.png'%i,facecolor=fig.get_facecolor(),dpi=dpi)
            clf()
            progress_bar(float(i_s-i0)/float(i1+1-i0)*100., 'making images')
        close(fig)
        #
        # make the movie
        #
        moviename = str(self.data_dir)
        if moviename[-1] == os.sep: moviename = moviename[0:-1]
        
        moviename = os.path.basename(moviename)+'.mp4'
        ret=subprocess.call(['ffmpeg','-i',dirname+os.sep+'img_%03d.png','-c:v','libx264','-crf','20','-maxrate','400k','-pix_fmt','yuv420p','-bufsize','1835k',moviename])

        if ret==0:
            print "Movie created, cleaning up ..."
            for i,i_s in enumerate(arange(i0,i1+1,steps)):
                os.remove(dirname+os.sep+'img_%3.3i.png'%i)
            os.removedirs(dirname)

    def load_diskev(self,filename="data/"):
        """
        load the data of a disk evolution simulation 
        Arguments:
            filename    the file name of the .hdf5 data file
        Examples:
            >>> D.load_diskev('data_sim1')
            >>> D.load_diskev('data_sim1/')
            >>> D.load_diskev('data_sim1.hdf5')
        """
        #
        # format the filename 
        #
        if filename[-1]==os.sep:
            filename=filename[0:-1]
        if filename.split('.')[-1] not in ["hdf5","mat"]:
            filename=filename+".hdf5"
        if filename.split('.')[-1] =="mat":
            ending = "_1"
        else:
            ending = ""
        #
        # check if file exists
        #
        try:
            open(filename)
        except IOError as e:
            print("({})".format(e))
        #
        # open file for reading 
        #
        sys.stdout.write("loading from %s"%filename)
        sys.stdout.flush()
        #
        # load data from file
        #
        if filename.split('.')[-1]=='mat':
            import scipy.io
            f = scipy.io.loadmat(filename)
            #
            # the data from the .mat files needs to be transposed 
            #
            self.data_dir             = f['dir'+ending][0]
            self.n_m                  = int(f['grains'][...])
#            self.D_grain1             = f['D_grain1'+ending][...].squeeze()
            self.T                    = f['T'+ending][...]
            self.accretion_dust       = f['accretion_dust'+ending][...].squeeze()
            self.accretion_dust_e     = f['accretion_dust_e'+ending][...].squeeze()
            self.accretion_gas        = f['accretion_gas'+ending][...].squeeze()
            self.alpha                = f['alpha'+ending][...]
            self.alpha_dead           = f['alpha_dead'+ending][...]
            self.d_evap               = f['d_evap'+ending][...]
            #self.dust_flux        = f['dust_flux'+ending][...]
            self.dust_flux_o          = f['dust_flux_o'+ending][...].squeeze()
            self.dust_flux_o_e        = f['dust_flux_o_e'+ending][...].squeeze()
            self.fallen_disk_mass     = f['fallen_disk_mass'+ending][...].squeeze()
            self.flim                 = f['flim'+ending][...].squeeze()
            self.flim_dead            = f['flim_dead'+ending][...].squeeze()
            self.gas_flux_o           = f['gas_flux_o'+ending][...].squeeze()
            self.grainsizes           = f['grainsizes'+ending][...].squeeze()
            self.m_dot_star           = f['m_dot_star'+ending][...].squeeze()
            self.m_grid               = f['m_grid'+ending][...].squeeze()
            self.m_star               = f['m_star'+ending][...].squeeze()
            self.n_r                  = int(f['n_r'][...])
            self.n_t                  = int(f['n_t'][...])
            self.nu                   = f['nu'+ending][...]
            self.peak_position        = f['peak_position'+ending][...].squeeze()
            self.r_centri             = f['r_centri'+ending][...].squeeze()
            self.r_min                = f['r_min'+ending][...].squeeze()
            self.r_snow               = f['r_snow'+ending][...].squeeze()
            self.sig_dot_t            = f['sig_dot_t'+ending][...].squeeze()
            self.sigma_coag           = f['sigma_coag'+ending][...]
            self.sigma_d              = f['sigma_d'+ending][...]
            self.sigma_dead           = f['sigma_dead'+ending][...]
            self.sigma_g              = f['sigma_g'+ending][...]
            self.steps                = f['steps'+ending][...].squeeze()
            self.timesteps            = f['timesteps'+ending][...].squeeze()
            self.v_dust               = f['v_dust'+ending][...]
            self.v_gas                = f['v_gas'+ending][...]
            self.v_gas_dead           = f['v_gas_dead'+ending][...]
            self.x                    = f['x'+ending][...].squeeze()
            self.x05                  = f['x05'+ending][...].squeeze()
            #
            # now read the name list variables:
            # these are all scalars with upper case names
            #
            self.nml=dict()
            for key in f.iterkeys():
                if key==key.upper() and len(f[key][...])==1:
                    self.nml[key]=float(f[key][...])
        else:
            f = h5py.File(filename,'r')
            if 'stored_data' in f.keys():
                for varname in f['stored_data'][()]:
                    setattr(self,varname,f[varname][()])
            else:
                #
                # all the try/except stuff is for reading in an file which lacks most of the variables
                # but which can be used as input of the vertical structure simulations via vertical_structure_from_data
                #
                try: self.data_dir         	  = str(f['data_dir'][...])
                except: pass
                try: self.n_m              	  = f['n_m'][...]
                except: pass
                try: self.D_grain1             = f['D_grain1'][...]
                except: pass
                try: self.T                    = f['T'][...]
                except: pass
                try: self.accretion_dust       = f['accretion_dust'][...]
                except: pass
                try: self.accretion_dust_e     = f['accretion_dust_e'][...]
                except: pass
                try: self.accretion_gas        = f['accretion_gas'][...]
                except: pass
                try: self.alpha                = f['alpha'][...]
                except: pass
                try: self.alpha_dead           = f['alpha_dead'][...]
                except: pass
                try: self.d_evap               = f['d_evap'][...]
                except: pass
                #try: self.dust_flux        = f['dust_flux'][...]
                #except: pass
                try: self.dust_flux_o          = f['dust_flux_o'][...]
                except: pass
                try: self.dust_flux_o_e        = f['dust_flux_o_e'][...]
                except: pass
                try: self.fallen_disk_mass     = f['fallen_disk_mass'][...]
                except: pass
                try: self.flim                 = f['flim'][...]
                except: pass
                try: self.flim_dead            = f['flim_dead'][...]
                except: pass
                try: self.gas_flux_o           = f['gas_flux_o'][...]
                except: pass
                try: self.grainsizes           = f['grainsizes'][...]
                except: pass
                try: self.m_dot_star           = f['m_dot_star'][...]
                except: pass
                try: self.m_grid               = f['m_grid'][...]
                except: pass
                try: self.m_star               = f['m_star'][...]
                except: pass
                try: self.n_r                  = f['n_r'][...]
                except: pass
                try: self.n_t                  = f['n_t'][...]
                except: pass
                try: self.nu                   = f['nu'][...]
                except: pass
                try: self.peak_position        = f['peak_position'][...]
                except: pass
                try: self.r_centri             = f['r_centri'][...]
                except: pass
                try: self.r_min                = f['r_min'][...]
                except: pass
                try: self.r_snow               = f['r_snow'][...]
                except: pass
                try: self.sig_dot_t            = f['sig_dot_t'][...]
                except: pass
                try: self.sigma_coag           = f['sigma_coag'][...]
                except: pass
                try: self.sigma_d              = f['sigma_d'][...]
                except: pass
                try: self.sigma_dead           = f['sigma_dead'][...]
                except: pass
                try: self.sigma_g              = f['sigma_g'][...]
                except: pass
                try: self.steps                = f['steps'][...]
                except: pass
                try: self.timesteps            = f['timesteps'][...]
                except: pass
                try: self.v_dust               = f['v_dust'][...]
                except: pass
                try: self.v_gas                = f['v_gas'][...]
                except: pass
                try: self.v_gas_dead           = f['v_gas_dead'][...]
                except: pass
                try: self.x                    = f['x'][...]
                except: pass
                try: self.x05                  = f['x05'][...]
                except: pass
            #
            # now load the namelist variables
            #
            self.nml = dict()
            try:
                for key,val in f['nml'].iteritems(): self.nml[key]=float(val[...])
            except:
                pass
        #
        # close the file
        #
        if hasattr(f,'close'): f.close()
        print " ... Done!"

    def get_m_gas(self):
        """
        Calculates and returns the total gas mass as function of time
        
        Arguments:
        None
        Example:
        >>>m_g = d.get_m_gas()
        >>>loglog(d.timesteps/year,m_g/M_sun)
        """
        m_g = array([trapz(2*pi*self.x*sig_g,self.x) for sig_g in self.sigma_g])
        return m_g
       
    def get_m_dust(self):
        """
        Calculates and returns the total dust mass as function of time
        
        Arguments:
        None
        Example:
        >>>m_d = d.get_m_dust()
        >>>loglog(d.timesteps/year,m_d/M_sun)
        """
        m_d = zeros(self.n_t)
        for it in arange(self.n_t):
            sig_d_t = sum(self.sigma_d[it*self.n_m+arange(self.n_m),:],0)
            m_d[it]=trapz(2*pi*self.x*sig_d_t,self.x)
        return m_d

    def get_sigma_dust_total(self):
        """
        Calculates and returns the total dust mass as function of time
        
        Arguments:
        None
        Example:
        >>>sig_d = d.sigma_dust_total()
        >>>loglog(d.x/AU,sid_d[-1])
        """
        sig_d_t = array([sum(self.sigma_d[it*self.n_m+arange(self.n_m),:],0) for it in arange(self.n_t)])
        return sig_d_t
    
    def get_d2g(self):
        """
        Calculates and returns the total dust mass as function of time
        
        Arguments:
        None
        Example:
        >>>sig_d = d.sigma_dust_total()
        >>>loglog(d.x/AU,sid_d[-1])
        """
        return self.get_sigma_dust_total()/self.sigma_g

    def plot_d2g_widget(self,N=0):
        """ 
        Produces a plot-widget of the dust-to-gas mass ratio.
        Arguments:
            N  = optional: start with snapshot #N
        Example:
            >>> D.plot_d2g_widget(133)
        """ 
        import widget as widget #@UnresolvedImport
        #
        # get dust-to-gas ratio
        #
        d2g = self.get_sigma_dust_total()/self.sigma_g
        MX = d2g.max()
        #
        # plot gas surface density at the given snapshot 'N' 
        #
        if (N+1>self.n_t):
            N=0;
        widget.plotter(x=self.x/self.AU,data=d2g,
                           times=self.timesteps/self.year,xlog=1,ylog=1,
                           xlim=[self.x[0]/self.AU,self.x[-1]/self.AU],
                           ylim=[1e-6*MX,MX],xlabel='r[AU]',i_start=N,
                           ylabel=r'dust-to-gas')
            
    def plot_sigma_g(self,N=0):
        """ 
        Produces a plot of the gas surface density at snapshot number N.
        Nothing fancy yet.
        Arguments:
            N   index of the snapshot, defaults to first snapshot
        Example:
            >>> D.plot_sigma_g(133)
        """ 
        from matplotlib.pyplot import loglog,title,xlabel,ylabel,ylim
        #
        # plot gas surface density at the given snapshot 'N' 
        #
        if (N+1>self.sigma_g.shape[0]):
            N=0;
        loglog(self.x,self.sigma_g[N],'-')
        title('Gas surface density')
        xlabel('r [AU]')
        ylabel('$\Sigma_\mathrm{g}$ [g cm$^{-2}$]')
        ylim((1e-4,1e4))
        #
        # get the time of the snapshot
        # and format it in latex
        #
        timestr="%5.0g" % round(self.timesteps[N]/3.15e7)
        timestr=timestr.replace('e','\\times 10^{')
        timestr=re.sub('\+[0]+','',timestr)
        timestr=re.sub('\-[0]+','-',timestr)
        timestr="$"+timestr+"}$"
        #
        # plot the snapshot time in the title
        #
        title('time = '+timestr+' yr')

    def plot_sigma_g_widget(self,N=0):
        """ 
        Produces a plot of the gas surface density at snapshot number N.
        Arguments:
            N   index of the snapshot, defaults to first snapshot
        Example:
            >>> D.plot_sigma_g_widget(133)
        """ 
        import widget as widget #@UnresolvedImport
        #
        # plot gas surface density at the given snapshot 'N' 
        #
        if (N+1>self.sigma_g.shape[0]):
            N=0;
        widget.plotter(x=self.x/self.AU,data=self.sigma_g,
                           times=self.timesteps/self.year,xlog=1,ylog=1,
                           xlim=[self.x[0]/self.AU,self.x[-1]/self.AU],
                           ylim=[1e-4,1e4],xlabel='r[AU]',i_start=N,
                           ylabel='$\Sigma_g$ [g cm $^{-2}$]')

    def plot_sigma_widget(self,N=0,ispec=None,**kwargs):
        """ 
        Produces a plot of the gas and total dust surface density at snapshot number N.
        
        Keywords:
        ---------
        
        N : int
        : index of the snapshot, defaults to first snapshot
        
        ispec : index or array of indices or array of arrays of indices
        :   int:             plot only this surface density
            array of ints:   plot these surface densities
            array of arrays: sum up over every array each sum
            
        **kwargs are passed to widget.plotter, some are set to default values if not given
        
        Example:
        --------
        >>> D.plot_sigma_widget(133)
        
        Reproduce plot from Andrews et al. 2014, ApJ
        
        i0=abs(d.grainsizes-0.5e-4).argmin()
        i1=abs(d.grainsizes-5e-4).argmin()
        i2=abs(d.grainsizes-500e-4).argmin()
        i3=abs(d.grainsizes-5).argmin()

        d.plot_sigma_widget(N=244,ispec=[arange(i0,i1+1),arange(i2,i3+1)],ylim=[5e-4,4],xlim=[70,445],xlog=False,ylog=True,lstyle=['r-','g-','c-'])
        """ 
        import widget
        #
        # plot gas surface density at the given snapshot 'N' 
        #
        if (N+1>self.sigma_g.shape[0]):
            N=0;
            
        if ispec is None:
            data2 = self.get_sigma_dust_total()
        if type(ispec) == int:
            data2 = self.sigma_d[arange(self.n_t)*self.n_m+ispec,:]
        if type(ispec) in [list, ndarray]:
            data2 = []
            for i in ispec:
                if type(i)==int:
                    data2 += [self.sigma_d[arange(self.n_t)*self.n_m+ispec,:]]
                else:
                    data2 += [array([self.sigma_d[[j+it*self.n_m for j in i],:].sum(0) for it in range(self.n_t)])]
        #
        # set default values
        # 
        if 'xlim'   not in kwargs.keys(): kwargs['xlim']   = [self.x[0]/self.AU,self.x[-1]/self.AU]
        if 'ylim'   not in kwargs.keys(): kwargs['ylim']   = [1e-4,1e4]
        if 'xlabel' not in kwargs.keys(): kwargs['xlabel'] = 'r[AU]'
        if 'ylabel' not in kwargs.keys(): kwargs['ylabel'] = '$\Sigma$ [g cm $^{-2}$]'
        if 'xlog'   not in kwargs.keys(): kwargs['xlog']   = 1
        if 'ylog'   not in kwargs.keys(): kwargs['ylog']   = 1
            
        widget.plotter(x=self.x/self.AU,data=self.sigma_g,data2=data2,
                           times=self.timesteps/self.year,i_start=N,**kwargs)

    def plot_sigma_d_widget(self,N=0,sizelimits=False,stokesaxis=False,**kwargs):
        """ 
        Produces a plot of the 2D dust surface density at snapshot number N.
        
        Keywords:
        N
        :    index of the snapshot, defaults to first snapshot
        
        sizelimits
        :    wether or not to show the drift / fragmentation size limits
        
        stokesaxis : bool
        :    if true, use stokes number instead of particle size on y-axis

        **kwargs
        : will be passed forward to the widget
        
        Example:
        --------
        
            >>> D.plot_sigma_d_widget(200)
        """ 
        import widget
        from uTILities import get_St, progress_bar
        add_arr = []
        if sizelimits==True:
            RHO_S     = self.nml['RHO_S']
            a_fr = zeros([self.n_t,self.n_r])
            a_dr = a_fr.copy()
            a_df = a_fr.copy()
            for N in arange(self.n_t):
                sigma_d   = sum(self.sigma_d[N*self.n_m+arange(self.n_m),:],0)
                fudge_fr = 0.37
                fudge_dr = 0.55
                gamma = dlydlx(self.x,self.sigma_g[N])+0.5*dlydlx(self.x,self.T[N])-1.5
                #
                # the standard fomula with the fudge factor
                #
                #a_fr  = fudge_fr*2*self.sigma_g[N,:]*self.nml['V_FRAG']**2./(3*pi*self.alpha[N]*RHO_S*k_b*self.T[N]/mu/m_p)
                #
                # the nonlinear one
                #
                om    = sqrt(Grav*self.m_star[N]/self.x**3)
                cs    = sqrt(k_b*self.T[N]/mu/m_p)
                b     = 3.*self.alpha[N]*cs**2/self.nml['V_FRAG']**2
                a_fr[N,:]  = self.sigma_g[N]/(pi*RHO_S)*(b-sqrt(b**2-4.))
                a_dr[N,:]  = fudge_dr/(self.nml['DRIFT_FUDGE_FACTOR']+1e-20)*2/pi*sigma_d/RHO_S*self.x**2.*(Grav*self.m_star[N]/self.x**3)/(abs(gamma)*cs**2)
                NN     = 0.5
                a_df[N,:]  = fudge_fr*2*self.sigma_g[N]/(RHO_S*pi)*self.nml['V_FRAG']*sqrt(Grav*self.m_star[N]/self.x)/(abs(gamma)*cs**2*(1.-NN)) #@UnusedVariable
                St  = 2.*self.sigma_g/(pi*RHO_S)
                if self.nml['STOKES_REGIME']==1:
                    St = minimum(St,sqrt(9.*sqrt(2.*pi)/16.*mu*m_p*cs/(om*RHO_S*sig_h2)))
            add_arr += [St,a_fr,a_dr]
        #
        # plot gas surface density at the given snapshot 'N' 
        #
        if (N+1>self.sigma_g.shape[0]):
            N=0;
        gsf=log(self.grainsizes[1]/self.grainsizes[0])
        #
        # convert to stokes number as y-axis
        # 
        if stokesaxis:
            R,_ = meshgrid(self.x,self.grainsizes)
            Y   = zeros([self.n_t*self.n_m,self.n_r])
            for it in range(self.n_t):
                progress_bar(it/(self.n_t-1.)*100,'calculating St-axis')
                for ir in range(self.n_r):
                    Y[it*self.n_m+arange(self.n_m),ir] = get_St(self.grainsizes, self.T[it,ir], self.sigma_g[it,ir], self.x[ir], self.m_star[it], rho_s=self.nml['RHO_S'],Stokesregime=self.nml['STOKES_REGIME'])
            #
            # now the size limits:
            # they have all been derived for epstein drag, so let's revert this
            #
            if sizelimits:
                for i,limit in enumerate(add_arr):
                    add_arr[i] = limit*RHO_S/self.sigma_g*pi/2.0
        else:
            R = self.x
            Y = self.grainsizes
        #
        # call the widget
        #
        widget.plotter(x=R/self.AU,y=Y,
                       data=self.sigma_d/gsf,
                       data2=add_arr,i_start=N,times=self.timesteps/self.year,xlog=1,ylog=1,
                       zlog=1,zlim=array([1e-10,1e1])/gsf,xlabel='r [AU]',ylabel='grain size [cm]',lstyle=['k','w-','r-','y--'],**kwargs)

    def plot_sigma_d(self,N=-1,sizelimits=True,cmap=matplotlib.cm.get_cmap('hot'),fs=None,plot_style='c',xl=None,yl=None,xlog=True,ylog=True,clevel=None,ax_color='k',leg=True,bg_color='w',cb_color='w',fig=None,contour_lines=False,showtitle=True,colbar=True,time=None):
        """
        Produces my default plot of the dust surface density.
        
        N : integer
        : index of the snapshot, defaults to last snapshot
        
        sizelimits : bool
        : wether or not to overplot the growth barriers
        
        cmap : colormap
        : colormap to be used for the plot

        fs : int
        : font size
        
        plot_style : str
        : 'c' for contourf, 's' for pcolor
        
        xl : list
        : x limits
        
        yl: list
        : y limits

        xlog : bool
        : log x axis
        
        ylog : bool
        : log y axis
        
        clevel : array or list
        : contour levels for the contour plot
        
        ax_color : str
        : color of the fonts and axes
        
        leg : bool
        : wether or not to print a legend for the size limits
        
        bg_color : color
        : color to use as background
        
        cb_color : color
        : color to use for the color bar axes and labels
        
        fig : figure
        : into which figure to plot
        
        contour_lines : bool
        : whether or not to print lines between the contour areas

        showtitle : bool
        : whether or not to print the snapshot time as plot title

        colbar : bool
            whether or not to show a colorbar
            
        time : float
        : if not none, plot the snapshot at that time (or the closest one)
        
        Example:
            >>> D.plot_sigma_d(133)

        """
        from matplotlib.pyplot import figure,loglog,title,xlabel,ylabel,\
        xlim,contourf,pcolor,gca,xscale,yscale,colorbar,contour,setp,legend,rcParams
        params = rcParams.copy()
        if time is not None: N = abs(self.timesteps/year-time).argmin()
        if clevel==None: clevel=arange(-10,1)+ceil(log10(self.sigma_d.max()/log(self.grainsizes[1]/self.grainsizes[0])))
        if fs is not None: rcParams['font.size']=fs
        #
        # check input
        #
        if (N>self.sigma_d.shape[0]/self.n_m-1):
            print("\"N\" larger than n_t, using first element");
            N=0;
        #
        # select the plot style
        #
        if plot_style not in ['c','s']:
            raise NameError("plot style\""+plot_style+"\" does not exist.")
        #
        # create a figure
        #
        #
        # calculate size limits
        #
        if sizelimits==True:
            RHO_S     = self.nml['RHO_S']
            sigma_d   = sum(self.sigma_d[N*self.n_m+arange(self.n_m),:],0)
            #fudge_fr = 0.37
            fudge_dr = 0.55
            gamma = dlydlx(self.x,self.sigma_g[N])+0.5*dlydlx(self.x,self.T[N])-1.5
            #
            # the standard fomula with the fudge factor
            #
            #a_fr  = fudge_fr*2*self.sigma_g[N,:]*self.nml['V_FRAG']**2./(3*pi*self.alpha[N]*RHO_S*k_b*self.T[N]/mu/m_p)
            #
            # the nonlinear one without fudge factor
            #
            b     = 3.*self.alpha[N]*k_b*self.T[N]/mu/m_p/self.nml['V_FRAG']**2
            a_fr  = self.sigma_g[N]/(pi*RHO_S)*(b-sqrt(b**2-4.))
            a_dr  = fudge_dr/(self.nml['DRIFT_FUDGE_FACTOR']+1e-20)*2/pi*sigma_d/RHO_S*self.x**2.*(Grav*self.m_star[N]/self.x**3)/(abs(gamma)*(k_b*self.T[N]/mu/m_p))
            #NN     = 0.5
            #a_df  = fudge_fr*2*self.sigma_g[N]/(RHO_S*pi)*self.nml['V_FRAG']*sqrt(Grav*self.m_star[N]/self.x)/(abs(gamma)*k_b*self.T[N]/mu/m_p*(1.-NN)) #@UnusedVariable
        if fig==None:
            fig=figure()
        figure(fig.number)
        fig.set_facecolor(bg_color)
        #
        # draw the data
        #
        if plot_style=='c':
            cont=contourf(self.x/1.4e13,self.grainsizes,log10(#10**clevel[0]+ 
              self.sigma_d[N*self.n_m+arange(0,self.n_m),:]/log(self.grainsizes[1]/self.grainsizes[0])),clevel,cmap=cmap,extend='both')
            cont.cmap.set_under('k')
            for p in cont.collections: p.set_edgecolor('face')
        if plot_style=='s':
            X,Y = meshgrid(self.x,self.grainsizes)
            pcolor(X/1.4e13,Y,log10(1e-10+self.sigma_d[N*self.n_m+arange(0,self.n_m),:]/log(self.grainsizes[1]/self.grainsizes[0])))
        gca()
        if sizelimits==True:
            lim_lines   = []
            lim_strings = []
            if yl==None:
                yl=array(gca().get_ylim())
                yl[0]=max(yl[0],self.grainsizes[0])
            if self.nml['FRAG_SWITCH']==1:
                lim_lines+=loglog(self.x/AU,a_fr,'r-',linewidth=2)
                lim_strings+=['fragmentation barrier']
            lim_lines+=loglog(self.x/AU,a_dr,'r--',linewidth=2)
            lim_strings+=['drift barrier']
        if xl is not None:
            gca().set_xlim(xl)
        else:
            xlim(self.x[0]/AU,self.x[-1]/AU)
        if yl is not None:
            gca().set_ylim(yl)
        #
        # set logarithmic scales
        #
        if not xlog: xscale('linear')
        if not ylog: yscale('linear')
        #
        # draw color bar
        #
        ax=gca()
        if colbar:
            my_colorbar(gca(),logify=True,col=cb_color,fs=(fs or 14)-2,title_string='$\sigma(a)$ [g cm$^{-2}$]',doplot=lambda x: True)  # @UnusedVariable
            if False:
                pos=ax.get_position()
                cb=colorbar(shrink=0.8)
                cb.ax.collections[0].set_edgecolor('face')
                ax.set_position([pos.x0,pos.y0,pos.size[0],pos.size[1]])
                tx = array([])
                for t in cb.ax.get_yticklabels():
                    t.set_fontsize(12)
                    tx = append(tx,'$10^{'+t.get_text().replace(u'\u2212','-').replace('$','')+'}$')
                cb.ax.set_yticklabels(tx[:-1],color=cb_color,fontsize=(fs or 14))
                cb.ax.set_title('$\log_{10}\,\sigma(r,a) \mathrm{[g\,cm}^{-2}\mathrm{]}$',fontsize=(fs or 14),color=cb_color)
        #
        # set axes label
        #
        xlabel("r [AU]",**{'color':ax_color})
        ylabel("grain size [cm]",**{'color':ax_color})
        #
        # now draw the contour lines
        #
        if plot_style=='c' and contour_lines:
            contour(self.x/1.4e13,self.grainsizes,log10(1e-10+self.sigma_d[N*self.n_m+arange(0,self.n_m),:]),
                         clevel[1:],colors='k',linestyles="-")
        #
        # get the time of the snapshot
        # and format it in latex
        #
        timestr="%5.0g" % round(self.timesteps[N]/3.15e7)
        timestr=timestr.replace('e','\\times 10^{')
        timestr=re.sub('\+[0]+','',timestr)
        timestr=re.sub('\-[0]+','-',timestr)
        timestr="$"+timestr+"}$"
        #
        # plot the snapshot time in the title
        #
        if showtitle: title('time = '+timestr+' yr',**{'color':ax_color})
        if ax_color!='k':
            ax.xaxis.label.set_color(ax_color)
            ax.yaxis.label.set_color(ax_color)
            ax.tick_params(axis='both', which='both',colors=ax_color)
            setp(ax.spines.values(), color=ax_color)
        #
        # print a legend
        #
        if leg==True and sizelimits==True:
            leg=legend(lim_lines,lim_strings,loc='upper left')
            for t in leg.get_texts(): t.set_color(cb_color) 
            leg.get_frame().set_color('None')
        #
        # back to previous settings
        # 
        rcParams=params

    def read_diskev(self,data_dir="data/"):
        """
        Reads in the simulation data from the given folder.
   
        Arguments:
            data_dir  direcdtory to read from 
        Example:
            >>> D.read_diskev('data_simulationname')
        """
        #
        # check if directory exists
        #
        if not os.path.isdir(data_dir):
            raise NameError("data directory\""+data_dir+"\" does not exist.")
        #
        # change data_dir without slash in the end
        #
        if data_dir[len(data_dir)-1]=='/':
            data_dir = data_dir[0:-1]
        #
        # now save data_dir as attribute
        #
        pos = data_dir.rfind('/')
        if pos == -1:
            self.data_dir = data_dir
        else:
            self.data_dir = data_dir[pos+1:len(data_dir)+1]
        #
        # reformat it to end with a slash
        #
        data_dir=data_dir+'/';
        self.stored_data = ['stored_data','data_dir']
        #
        # get the list of files and iterate through them
        #
        print("Reading from directory "+data_dir)
        file_list = glob.glob(data_dir+'/*.dat')
        for infile in file_list:
            filename=infile.replace(data_dir,'');
            if filename=='debug_distri.dat':
                print("    skipping "+filename)
                continue
            print("    working on file: " + filename)
            #
            # read data and set variable name
            #
            varname = filename.replace('.dat','');
            with open(infile) as fid:
                try:
                    setattr(self,varname,array(fid.read().strip().split(),dtype='float'))
                    self.stored_data+=[varname]
                except:
                    print('ERROR: unable to read data for varialbe `%s`'%varname)
                    #raise Exception('unable to read in variable')
        #
        # now read the name list if it exists
        #
        if os.path.isfile(data_dir+"usedinput.nml"):
            print "    reading namelist"
            self.nml          = parse_nml(data_dir+"usedinput.nml")
        #
        # now get some constants
        #
        self.n_r     = len(self.x)
        self.n_t     = len(self.timesteps)
        self.n_m     = len(self.m_grid)
        self.stored_data += ['n_r','n_t','n_m']
        #
        # reformat the variables
        #
        n_r = self.n_r
        n_m = self.n_m
        n_t = self.n_t
        if self.alpha      is not None: self.alpha      = self.alpha.reshape(-1,n_r)
        if self.alpha_dead is not None: self.alpha_dead = self.alpha_dead.reshape(-1,n_r)
        if self.nu         is not None: self.nu         = self.nu.reshape(n_t,n_r)
        if self.sigma_dead is not None: self.sigma_dead = self.sigma_dead.reshape(-1,n_r)
        if self.sigma_g    is not None: self.sigma_g    = self.sigma_g.reshape(-1,n_r)
        if self.sigma_d    is not None: self.sigma_d    = self.sigma_d.reshape(n_t*n_m,n_r)
        if self.T          is not None: self.T          = self.T.reshape(n_t,n_r)
        if self.v_gas      is not None: self.v_gas      = self.v_gas.reshape(n_t,n_r)
        if self.v_gas_dead is not None: self.v_gas_dead = self.v_gas_dead.reshape(n_t,n_r)
        if self.v_dust     is not None: self.v_dust     = self.v_dust.reshape(n_t*n_m,n_r)
        #
        # convert the integer variables from float to int
        #
        self.peak_position.astype('int32')
        self.r_min.astype('int32')
        self.steps.astype('int32')
        print "... Done"

    def save_diskev(self,data_dir=''):
        """
        Saves the data to a .hdf5 file.

        Arguments:
            data_dir   filename to write to. Defaults to the 
            		   data_dir attribute or to "data" if data_dir is
            		   empty.
        Example:
            >>> save_diskev('data')
        """
        #
        # use input or default values
        #
        if data_dir=="":
            data_dir=getattr(self,"data_dir","data")
        #
        # open file for writing 
        #
        filename = data_dir.replace('/','')+".hdf5"
        f = h5py.File(filename,mode='w')
        sys.stdout.write('saving as '+filename)
        sys.stdout.flush()
        #
        # write data into file
        #
        for dataname in self.stored_data:
            data = getattr(self,dataname)
            if data is not None :
                if type(data) in [int,str,float]:
                    f.create_dataset(dataname, data=data)
                else:
                    f.create_dataset(dataname, data=data, compression=4)
        #
        # save the nml
        #
        if self.nml is not None:
            grp=f.create_group("nml")
            for key,val in self.nml.iteritems():
                grp.create_dataset(key, data=val)
        #
        # close file
        #
        f.close()
        print(' ... Done!')
        
    # -----------------------------------------------------------------------------
    def write_setup(self,it,overwrite=None):
        """
        This function writes out the setup files to continue a simulation from a given snapshot
        
        Arguments:
        ----------
        
        it
        :    snapshot index
        
        Keyword:
        --------
        
        overwrite = [*None* | bool]
            if true:    always overwrite
            if false:   never overwrite
            if None:    ask
        
        Output:
        -------
        the namelist file and the other input files will be written out
        """
        from constants import M_sun
        #
        # create output folder or ask if it should be overwritten
        #
        dirname = 'setup_files'
        simname = self.data_dir
        simname = re.sub('^data_','',simname)
        simname = re.sub('^in_','',simname)
        if os.path.isdir(dirname):
            if overwrite!=True:
                inp = None
                if overwrite==False:
                    print('output directory exists, aborting')
                    return
                while inp not in ['','y','n']:
                    inp=raw_input('\'%s\' already exists, overwrite [Y/n] '%dirname).lower()
                    if inp=='n': 
                        print('operation cancelled')
                        return
        else:
            os.mkdir(dirname)
        #
        # convert index
        #
        it=find(self.timesteps==self.timesteps[it])[0]
        sys.stdout.write('Writing out index %i at time %g years ... '%(it,self.timesteps[it]/year));
        sys.stdout.flush()
        #
        # write nml file
        #
        nml = self.nml.copy()
        nml['T_0'] = self.timesteps[it]/year
        if nml['INTERVAL']<0: nml['INTERVAL'] += it
        nml['M_STAR_INITIAL'] = self.m_star[it]/M_sun
        nml['T_COAG_START'] = max(self.timesteps[it]/year,nml['T_COAG_START'])
        nml['STARTING_SIGMA'] = 1
        nml['READ_IN_ALPHA'] = 1
        nml['READ_IN_X'] = 1
        write_nml(nml,dirname+os.sep+'input_%s.nml'%simname,'INPUTVARIABLES')
        #
        # write other data
        #
        savetxt(dirname+os.sep+'x_input_'+    simname+'.dat', self.x,          delimiter=' ')
        savetxt(dirname+os.sep+'gas_input_'+  simname+'.dat', self.sigma_g[it],delimiter=' ')
        savetxt(dirname+os.sep+'dust_input_'+ simname+'.dat', self.sigma_d[it*self.n_m+arange(self.n_m),:],delimiter=' ')
        if nml['TEMPERATURE_METHOD']==0:
            savetxt(dirname+os.sep+'T_input_'+simname+'.dat', self.T[it],      delimiter=' ')
        savetxt(dirname+os.sep+'alpha_input_'+simname+'.dat', self.alpha[it],delimiter=' ')
        print('Done!')

    # -----------------------------------------------------------------------------
    def export_general(self,out_dir):
        """
        This routine saves the snapshot to text files, which can then be
        read in by collaborators. The goal of this is to have a consistent
        output for others, even if my model changes its own output (as it 
        already does due to compiler differences). 
        """
        import shutil
        #
        # check if the directory exists. if so, ask the user, else create it
        #
        out_dir = os.path.expanduser(out_dir)
        if os.path.isdir(out_dir):
            yn = None
            while yn not in ['y','n']:
                yn=raw_input('Directory exists, overwrite? [y/n] ').lower()
                if yn=='y':
                    shutil.rmtree(out_dir)
                elif yn=='n':
                    return
            os.mkdir(out_dir)
        else:
            if out_dir==None:
                out_dir=raw_input('Please provide name for output directory: ')
                out_dir=os.path.expanduser(out_dir)
            #
            # create directory
            #
            try:
                os.mkdir(out_dir)
            except:
                print 'failed to create directory \''+out_dir+'\''
                sys.exit(2)
        #
        # now write the data to the files
        #
        savetxt(out_dir+os.sep+'sigma_d.dat',   self.sigma_d)
        savetxt(out_dir+os.sep+'sigma_g.dat',   self.sigma_g)
        savetxt(out_dir+os.sep+'grainsizes.dat',self.grainsizes)
        savetxt(out_dir+os.sep+'T.dat',         self.T)
        savetxt(out_dir+os.sep+'x.dat',         self.x)
        savetxt(out_dir+os.sep+'timesteps.dat', self.timesteps)
        write_nml(self.nml,out_dir+os.sep+'variables.nml',listname='inputvars')
    
    # -----------------------------------------------------------------------------
    def str2num(self,st):
        """
        Convert an input string to int or to float or string.

        Arguments:
            st =   a string representing
                    an int or float value

        Output:
            returns a int or float depending on the input string
            if converting fails, the string is returned.

        Example:
            >>> type(str2num('1.1'))
            <type 'float'>
            >>> type(str2num('1'))
            <type 'int'>
            >>> type(str2num('x'))
            <type 'st'>
        """
        #
        # first try int
        #
        try:
            val = int(st)
        except ValueError:
            #
            # try float
            #
            try:
                val = float(st)
            except ValueError:
                #
                # use string
                #
                val = st
        return val

    def make_sim_interactive(self):
        """
        This function is to make a simulation result locally availabe in the interactive ipython/python environment.
        Note that in order to make it work, you cannot import it but you need to execute it like %run or execfile(...)
        """
        def_load = True
        try:
            from IPython import get_ipython
        except:
            def_load = False
        if not def_load: raise NameError('ERROR: could not import ipython, thus make_sim_interactive could not be defined')
        #
        # define a dictionary and fill it with the names and values of the variables
        #
        variables = dict()
        attributes = dir(self)
        for attr in attributes:
            val = getattr(self,attr)
            #
            # load everything but methods
            #
            if type(val).__name__ != 'instancemethod':
                variables[attr] = val
        #
        # get the workspace and load the dictionary into it
        #
        ns = get_ipython()
        ns.user_ns.update(variables)
        #
        # make the namelist variables also global
        #
        if hasattr(self,'nml'): ns.user_ns.update(self.nml)

def interpolate_for_luca(J,directory='.',lucasgrids='',mask='*'):
    """
    This file interpolate the distribution on a new grid. Will read all mat/hdf5
    files from the given input directory.
    
    Arguments:
    J           = indices of the time array to be written-out
    directory   = where the mat/hdf5 files are. read them all.
    lucasgrids  = where the new grid is located. output will also
                  be written into this directory
    mask        = all files matching mask.mat and mask.hdf5 in
                  the specified directory will be read
    
    e.g.:
    times = append(array([1,2,4,8])*1e5,arange(1,6)*1e6)
    J     = []
    for t in times: J += [find(d.timesteps/year>=t)[0]]
    """
    from IPython import embed
    files=glob.glob(directory+os.sep+mask+'.mat')+glob.glob(directory+os.sep+mask+'.hdf5')
    if lucasgrids=='':
        lucasgrids=directory;
        
    for filename in files:
        for j in J:
            filename = filename.replace('.mat','')
            filename = filename.replace('.hdf5','')
            print('working on simulation '+os.path.basename(filename)+' snapshot #%i'%find(array(J)==j)[0])
            outputdir = directory+os.sep+'numberdensity_'+os.path.basename(filename).replace('data_','')
            if not os.path.isdir(outputdir):
                os.makedirs(outputdir)
            # 
            # now read in the data
            #
            d             = pydisk1D(filename)
            x_in          = d.x
            sigma_d_1     = d.sigma_d
            sigma_g_1     = d.sigma_g
            grainsizes_in = d.grainsizes
            RHO_S         = d.nml['RHO_S']
            T_in          = d.T
            timesteps_1   = d.timesteps
            grains        = len(grainsizes_in)
            sig_in        = sigma_d_1[j*grains+arange(grains),:]/log(grainsizes_in[1]/grainsizes_in[0])            
            #
            # output
            #
            x_out          = loadtxt(lucasgrids+os.sep+'Radius.dat',skiprows=1)
            grainsizes_out = loadtxt(lucasgrids+os.sep+'grainsize.dat',skiprows=1)
            #
            # get total dust sigma at each radius
            #
            sig_in_total = sum(sig_in*log(grainsizes_in[1]/grainsizes_in[0]),0)
            m_in = trapz(2*pi*x_in**2*sig_in_total,x=log(x_in))
            #
            # interpolate dust
            #
            outgrid = RectBivariateSpline(log10(x_in),log10(grainsizes_in),log10(sig_in).transpose(),kx=1,ky=1)
            sig_out = 10**outgrid(log10(x_out),log10(grainsizes_out),mth='array').transpose()
            #
            # remove the extrapolated part
            #
            sig_out[0:find(grainsizes_out<=grainsizes_in[0])[-1],:] = 1e-100
            if any(isnan(sig_out)):
                print('NaN detected in sig_out')
                embed()
            if any(sig_out<0):
                print('negative number detected in sig_out')
                embed()
            #
            # renormalize:
            # get the interpolated normalization constant for each of lucas radii
            #
            norm = 10**interp(log10(x_out),log10(x_in),log10(sig_in_total),left=-100,right=-100)
            for ir in arange(len(x_out)):
                if norm[ir]>1e-50:
                    dummy = norm[ir]*sig_out[:,ir]/sum(sig_out[:,ir]*log(grainsizes_out[1]/grainsizes_out[0]))
                    dummy[dummy<=0] = 1e-300
                    sig_out[:,ir] = dummy
                if any(isnan(sig_out[:,ir])) or any(sig_out[:,ir]<=0):
                    print('error at ir = %i'%ir)
                    sys.exit(2)
            #
            # check normalization            
            #
            sig_out_total = array([sum(S*log(grainsizes_out[1]/grainsizes_out[0])) for S in sig_out.transpose()])
            m_out = trapz(2*pi*x_out**2*sig_out_total,x=log(x_out))
            perc_diff = (m_out-m_in)/m_in*100
            if abs(perc_diff)>10.0:
                print('ERROR: mass difference too large: %2.2g %%'%perc_diff)
                embed()
            else:
                print('mass difference: %2.2g %%'%perc_diff)
                sig_out = sig_out*m_in/m_out
            #
            # convert to vertically integrated number density N(a)
            #
            N = array([S/(4.*pi/3.*RHO_S*grainsizes_out**4) for S in sig_out.transpose()]).transpose()
            if any(isnan(N)):
                print('Error: NaN in N detected')
                embed()
            if any(N<0.0):
                print('Error: negative number in N detected')
                embed()
            # 
            # write out the result in one column
            #
            s = '%0.2fMyr'%(timesteps_1[j]/(1e6*year))
            #
            # file name 
            #
            outfile = outputdir+os.sep+'numberdensity_'+s+'_l.dat'
            if os.path.isfile(outfile):
                os.remove(outfile)
            #
            # file write-out
            #
            fid=file(outfile,'w')
            N.flatten(order='F').tofile(fid, sep="\n", format="%12.12e")
            fid.close()
            #
            # write out the interpolated total sigma_d, T, ans sig_gas
            #
            sig_out_total = array([sum(log(grainsizes_out[1]/grainsizes_out[0])*S) for S in sig_out.transpose()])
            savetxt(outputdir+os.sep+'sigma_d_total_'+s+'_l.dat',sig_out_total)
            
            T_out = 10**interp(log10(x_out),log10(x_in),log10(T_in[j,:]))
            savetxt(outputdir+os.sep+'temperature_'+s+'_l.dat',T_out)
            sigg_out = 10**interp(log10(x_out),log10(x_in),log10(sigma_g_1[j,:]))
            savetxt(outputdir+os.sep+'sigma_g_'+s+'_l.dat',sigg_out)
    print('ALL DONE')

def pydisk1D_readall(mask='data*'):
    """
    This script read in all the directories matching the mask and save them in the hdf5 format
    Arguments:
    mask    =  string to match for data directories, default ) 'data*'
    
    Output:
    The data is saved locally as .hdf5 files
    """
    for i in glob.glob(mask):
        if os.path.isdir(i):
            d=pydisk1D(i)
            d.save_diskev()
    print("========")            
    print("FINISHED")            
    print("========")
                
def setup_diskev(sim_name,R,T,sig_g,alpha,inputvars,sig_d=None,savedir='.',res=10):
    """
    This setup writes the input for a diskev simulation
    
    Arguments:
    ----------
    sim_name:     the name of the simulation
    R:            the radius grid [cm]
    T:            the temperature grid [K]
    sig_g:        the gas surface density grid [g cm^-2]
    alpha:        the turbulence parameter array
    inputvars:    dictionary with the various parameters that should be set.
    
    Keywords:
    ---------
    sig_d : array
        A dust surface density array, which will be written out along with
        the other setup files
        
    savedir : string
        the name of the directory where the output is stored
        
    res : float
        resolution of the mass grid, there are `res` bins per magnitude in mass
    
    Output:
    -------
    The output arrays for alpha,t,sigma_g,x, and the input namelist are written
    in the current directory.
    """
    from constants import R_sun
    #
    # set the default name list variables
    #
    placeholder = random.uniform()
    NML = dict(
    T_STAR              = 4000.,
    R_STAR              = 2.5*R_sun,
    M_STAR_INITIAL      = 0.5,
    M_STAR_FINAL        = 0.5,
    R_IN                = 0.3,
    R_OUT               = 2000,
    T_0                 = 1e3,
    T_MAX               = 1e6,
    T_COAG_START        = 1e3,
    RHO_S               = 1.6,
    INTERVAL            = -100,
    CONST_ALPHA         = 1e-2,
    CONST_ALPHA_DEAD    = 1e-5,
    INFALL_ALPHA_BOOST  = 0,
    CS_CLOUD            = 3e4,
    OMEGA_FACTOR        = 5e-2,
    PHI_IRRAD           = 5e-2,
    V_FRAG              = 1e3,
    FRAG_SWITCH         = 1,
    FRAG_SLOPE          = 0.166,
    MIN_M               = 1.5e-14,
    MAX_M               = 8e3,
    GROWTH_LIMIT        = 8e3,
    GAS2DUST_RATIO      = 100,
    TEMPERATURE_METHOD  = 0,
    STARTING_SIGMA      = 1,
    READ_IN_ALPHA       = 1,
    READ_IN_X           = 1,
    SIGMA_0             = 21,
    FRIDI_DELTA         = 0.8,
    NO_VISC_HEAT        = 1,
    USE_PEAK_POSITION   = 1,
    GAS_EVOL_SWITCH     = 1,
    DUST_EVOL_SWITCH    = 1,
    ALLOW_DEADZONE      = 0,
    TOOMRE_SWITCH       = 0,
    COAGULATION_SWITCH  = 1,
    COAGULATION_METHOD  = 2,
    EQUILIBRIUM_SWITCH  = 0,
    DUST_INFALL_SWITCH  = 0,
    GAS_INFALL_SWITCH   = 0,
    DUST_DIFFUSION      = 1,
    DUST_RADIALDRIFT    = 1,
    DUST_DRAG           = 1,
    DRIFT_FUDGE_FACTOR  = 1,
    VREL_BM             = 1,
    VREL_TV             = 1,
    VREL_RD             = 1,
    VREL_VS             = 1,
    VREL_AZ             = 1,
    FEEDBACK_SWITCH     = 1,
    EVAPORATION_SWITCH  = 0,
    DUMP_COUNTER        = 200,
    STOKES_FACTOR       = pi/2.,
    STOKES_REGIME       = 0,
    SC                  = 1.0)
    #
    # possibly create output directory
    #
    savedir = os.path.expanduser(savedir)
    if not os.path.isdir(savedir): os.mkdir(savedir)
    #
    # now for each iteration, do the setup
    #
    for i in arange(1):#@UnusedVariable
        #
        # copy the name list ...
        #
        nml = NML.copy()
        nml['R_IN']         = R[0]/AU
        nml['R_OUT']        = R[-1]/AU
        #
        # change it ...
        #
        for n,v in inputvars.iteritems():
            if n in nml:
                nml[n] = v
            else:
                print('ERROR: unknown variable %s'%n)
                sys.exit(1)
        for n,v in nml.iteritems():
            if v==placeholder:
                print('ERROR: %s needs to be set'%n)
                sys.exit(1)            
        #
        # remove growth limit
        #
        nml['GROWTH_LIMIT'] = nml['MAX_M']
        #
        # now the number of bins
        #
        NMBINS = log10(nml['MAX_M']/nml['MIN_M'])*res# higher res
        #
        # create the name
        #
        NAME = sim_name
        NAME += '_%i' % (NMBINS)
        #
        # save it.
        # 
        write_nml(nml,savedir+os.sep+'input_'+NAME+'.nml', 'INPUTVARIABLES')
        #
        # write the grid and the surface density
        #
        savetxt(savedir+os.sep+'x_input_'+    NAME+'.dat', R,     delimiter=' ')
        savetxt(savedir+os.sep+'gas_input_'+  NAME+'.dat', sig_g, delimiter=' ')
        if nml['TEMPERATURE_METHOD']==0: savetxt(savedir+os.sep+'T_input_'+    NAME+'.dat', T,     delimiter=' ')
        savetxt(savedir+os.sep+'alpha_input_'+NAME+'.dat', alpha, delimiter=' ')
        if sig_d is not None: savetxt(savedir+os.sep+'dust_input_'+NAME+'.dat', sig_d, delimiter=' ')
