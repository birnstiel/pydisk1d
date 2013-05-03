from matplotlib.pyplot import * #@UnusedWildImport
from numpy import * #@UnusedWildImport

import h5py
import re
import glob
import os
from uTILities import parse_nml, dlydlx, write_nml, progress_bar
from constants import AU,year, k_b, m_p, mu, Grav
from matplotlib.mlab import find
from scipy.interpolate import RectBivariateSpline

class pydisk1D:
    """
    Class to read, load, save and plot the simulation
    results of the diskev code by Til Birnstiel.
    
    For questions and comments please write
    to til.birnstiel@lmu.de

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
    
    def sigma_d_movie(self,i0=0,i1=-1,steps=1):
        """
        This uses the other sub-routine plot_sigma_d() to produce
        a movie of the time evolution
        """
        import subprocess
        if i0<0: i0=0
        if i1>self.n_t or i1==-1: i1 = self.n_t - 1
        #
        # check if there is already a movie_images folder
        # otherwise create one
        #
        if os.path.isdir('movie_images'):
            print('movie_images folder already exists, please delete it first')
            sys.exit(-1)
        else:
            os.mkdir('movie_images')
        #
        # make the images
        #
        fig=figure()
        for i,i_s in enumerate(arange(i0,i1+1,steps)):
            self.plot_sigma_d(i_s,fig=fig)
            savefig('movie_images/img_%3.3i.png'%i)
            clf()
            progress_bar(float(i_s-i0)/float(i1+1-i0)*100., 'making images')
        close(fig)
        #
        # make the movie
        #
        moviename = str(self.data_dir)
        if moviename[-1] == os.sep: moviename = moviename[0:-1]
        moviename = os.path.basename(moviename)+'.mp4'
        ret=subprocess.call(['ffmpeg','-i','movie_images/img_%03d.png','-r','10','-b','512k',moviename]);
        if ret==0:
            print "Movie created, cleaning up ..."
            for i,i_s in enumerate(arange(i0,i1+1,steps)):
                os.remove('movie_images/img_%3.3i.png'%i)
            os.removedirs('movie_images')

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
        print "loading from",filename
        f = h5py.File(filename,'r')
        #
        # load data from file
        #
        if filename.split('.')[-1]=='mat':
            #
            # the data from the .mat files needs to be transposed 
            #
            self.data_dir             = f['dir'+ending][...]
            dummy=''
            for i in self.data_dir: dummy+=chr(i)
            self.data_dir = dummy
            self.n_m                  = int(f['grains'][...])
#            self.D_grain1             = f['D_grain1'+ending][...].squeeze()
            self.T                    = f['T'+ending][...].transpose()
            self.accretion_dust       = f['accretion_dust'+ending][...].squeeze()
            self.accretion_dust_e     = f['accretion_dust_e'+ending][...].squeeze()
            self.accretion_gas        = f['accretion_gas'+ending][...].squeeze()
            self.alpha                = f['alpha'+ending][...].transpose()
            self.alpha_dead           = f['alpha_dead'+ending][...].transpose()
            self.d_evap               = f['d_evap'+ending][...].transpose()
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
            self.nu                   = f['nu'+ending][...].transpose()
            self.peak_position        = f['peak_position'+ending][...].squeeze()
            self.r_centri             = f['r_centri'+ending][...].squeeze()
            self.r_min                = f['r_min'+ending][...].squeeze()
            self.r_snow               = f['r_snow'+ending][...].squeeze()
            self.sig_dot_t            = f['sig_dot_t'+ending][...].squeeze()
            self.sigma_coag           = f['sigma_coag'+ending][...].transpose()
            self.sigma_d              = f['sigma_d'+ending][...].transpose()
            self.sigma_dead           = f['sigma_dead'+ending][...].transpose()
            self.sigma_g              = f['sigma_g'+ending][...].transpose()
            self.steps                = f['steps'+ending][...].squeeze()
            self.timesteps            = f['timesteps'+ending][...].squeeze()
            self.v_dust               = f['v_dust'+ending][...].transpose()
            self.v_gas                = f['v_gas'+ending][...].transpose()
            self.v_gas_dead           = f['v_gas_dead'+ending][...].transpose()
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
        f.close()
        print "... Done"

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

    def plot_sigma_d_widget(self,N=0,sizelimits=False):
        """ 
        Produces a plot of the 2D dust surface density at snapshot number N.
        Arguments:
            N   index of the snapshot, defaults to first snapshot
        Example:
            >>> D.plot_sigma_d_widget(200)
        """ 
        import widget
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
                b     = 3.*self.alpha[N]*k_b*self.T[N]/mu/m_p/self.nml['V_FRAG']**2
                a_fr[N,:]  = self.sigma_g[N]/(pi*RHO_S)*(b-sqrt(b**2-4.))
                a_dr[N,:]  = fudge_dr/(self.nml['DRIFT_FUDGE_FACTOR']+1e-20)*2/pi*sigma_d/RHO_S*self.x**2.*(Grav*self.m_star[N]/self.x**3)/(abs(gamma)*(k_b*self.T[N]/mu/m_p))
                NN     = 0.5
                a_df[N,:]  = fudge_fr*2*self.sigma_g[N]/(RHO_S*pi)*self.nml['V_FRAG']*sqrt(Grav*self.m_star[N]/self.x)/(abs(gamma)*k_b*self.T[N]/mu/m_p*(1.-NN)) #@UnusedVariable
            add_arr += [a_fr,a_dr,2.*self.sigma_g/(pi*RHO_S)]
        #
        # plot gas surface density at the given snapshot 'N' 
        #
        if (N+1>self.sigma_g.shape[0]):
            N=0;
        gsf=log(self.grainsizes[1]/self.grainsizes[0])
        widget.plotter(x=self.x/self.AU,y=self.grainsizes,
                       data=self.sigma_d/gsf,
                       data2=add_arr,i_start=N,times=self.timesteps/self.year,xlog=1,ylog=1,
                       zlog=1,xlim=[self.x[0]/self.AU,self.x[-1]/self.AU],
                       ylim=[self.grainsizes[0],self.grainsizes[-1]],
                       zlim=array([1e-10,1e1])/gsf,xlabel='r [AU]',ylabel='grain size [cm]',lstyle=['w-','r-','r--'])

    def plot_sigma_d(self,N=0,sizelimits=True,cm=cm.hot,plot_style='c',xl=None,yl=None,clevel=arange(-10,1),cb_color='w',fig=None,contour_lines=False):
        """
        Produces a plot of the dust surface density at snapshot number N.
        
        Arguments:
            N   index of the snapshot, defaults to first snapshot
        Example:
            >>> D.plot_sigma_d(133)

        """
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
            fudge_fr = 0.37
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
            NN     = 0.5
            a_df  = fudge_fr*2*self.sigma_g[N]/(RHO_S*pi)*self.nml['V_FRAG']*sqrt(Grav*self.m_star[N]/self.x)/(abs(gamma)*k_b*self.T[N]/mu/m_p*(1.-NN)) #@UnusedVariable
        if fig==None:
            fig=figure()
        figure(fig.number)
        #
        # draw the data
        #
        if plot_style=='c':
            contourf(self.x/1.4e13,self.grainsizes,log10(10**clevel[0]+ 
              self.sigma_d[N*self.n_m+arange(0,self.n_m),:]/log(self.grainsizes[1]/self.grainsizes[0])),clevel,cmap=cm)
        if plot_style=='s':
            X,Y = meshgrid(self.x,self.grainsizes)
            pcolor(X/1.4e13,Y,log10(1e-10+self.sigma_d[N*self.n_m+arange(0,self.n_m),:]/log(self.grainsizes[1]/self.grainsizes[0])))
        gca()
        if sizelimits==True:
            if yl==None:
                yl=array(gca().get_ylim())
                yl[0]=max(yl[0],self.grainsizes[0])
            if self.nml['FRAG_SWITCH']==1:
                loglog(self.x/AU,a_fr,'r-',linewidth=2)
            loglog(self.x/AU,a_dr,'r--',linewidth=2)
        if xl!=None:
            gca().set_xlim(xl)
        else:
            xlim(self.x[0]/AU,self.x[-1]/AU)
        if yl!=None:
            gca().set_ylim(yl)
        #
        # set logarithmic scales
        #
        xscale('log')
        yscale('log')
        #
        # set colorbar it's title 
        #
#        col=colorbar()
#        col.ax.set_title("$\log_{10} \sigma$ [g cm$^{-2}$]")
        #
        # draw color bar
        #
        ax=gca()
        pos=ax.get_position()
        cb=colorbar(shrink=0.8)
        ax.set_position([pos.x0,pos.y0,pos.size[0],pos.size[1]])
        tx = array([])
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(12)
            tx = append(tx,'$10^{'+t.get_text().replace(u'\u2212','-')+'}$')
        cb.ax.set_yticklabels(tx[:-1],color=cb_color,fontsize=14)
        cb.ax.set_title('$\log_{10}\,\sigma(r,a) \mathrm{[g\,cm}^{-2}\mathrm{]}$',fontsize=14,color=cb_color)
        #
        # set axes label
        #
        xlabel("r [AU]")
        ylabel("grain size [cm]")
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
        title('time = '+timestr+' yr')

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
                except:
                    print('ERROR: unable to read data for varialbe `%s`'%varname)
                    #raise Exception('unable to read in variable')
        #
        # now read the name list if it exists
        #
        if os.path.isfile(data_dir+"usedinput.nml"):
            print "    reading namelist"
            self.nml = parse_nml(data_dir+"usedinput.nml")
        #
        # now get some constants
        #
        self.n_r     = len(self.x)
        self.n_t     = len(self.timesteps)
        self.n_m     = len(self.m_grid)
        #
        # reformat the variables
        #
        n_r = self.n_r
        n_m = self.n_m
        n_t = self.n_t
        if self.alpha      !=None: self.alpha      = self.alpha.reshape(-1,n_r)
        if self.alpha_dead !=None: self.alpha_dead = self.alpha_dead.reshape(-1,n_r)
        if self.nu         !=None: self.nu         = self.nu.reshape(n_t,n_r)
        if self.sigma_dead !=None: self.sigma_dead = self.sigma_dead.reshape(-1,n_r)
        if self.sigma_g    !=None: self.sigma_g    = self.sigma_g.reshape(-1,n_r)
        if self.sigma_d    !=None: self.sigma_d    = self.sigma_d.reshape(n_t*n_m,n_r)
        if self.T          !=None: self.T          = self.T.reshape(n_t,n_r)
        if self.v_gas      !=None: self.v_gas      = self.v_gas.reshape(n_t,n_r)
        if self.v_gas_dead !=None: self.v_gas_dead = self.v_gas_dead.reshape(n_t,n_r)
        if self.v_dust     !=None: self.v_dust     = self.v_dust.reshape(n_t*n_m,n_r)
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
        print "saving as",filename
        #
        # write data into file
        #
        if self.D_grain1!=None:         f.create_dataset('D_grain1',         compression=4, data=self.D_grain1)
        if self.T!=None:                f.create_dataset('T',                compression=4, data=self.T)
        if self.accretion_dust!=None:   f.create_dataset('accretion_dust',   compression=4, data=self.accretion_dust)
        if self.accretion_dust_e!=None: f.create_dataset('accretion_dust_e', compression=4, data=self.accretion_dust_e)
        if self.accretion_gas!=None:    f.create_dataset('accretion_gas',    compression=4, data=self.accretion_gas)
        if self.alpha!=None:            f.create_dataset('alpha',            compression=4, data=self.alpha)
        if self.alpha_dead!=None:       f.create_dataset('alpha_dead',       compression=4, data=self.alpha_dead)
        if self.d_evap!=None:           f.create_dataset('d_evap',           compression=4, data=self.d_evap)
        if self.data_dir!=None:         f.create_dataset('data_dir',                        data=self.data_dir)
        #if self.dust_flux!=None:         f.create_dataset('dust_flux',        compression=4, data=self.dust_flux)
        if self.dust_flux_o!=None:      f.create_dataset('dust_flux_o',      compression=4, data=self.dust_flux_o)
        if self.dust_flux_o_e!=None:    f.create_dataset('dust_flux_o_e',    compression=4, data=self.dust_flux_o_e)
        if self.fallen_disk_mass!=None: f.create_dataset('fallen_disk_mass', compression=4, data=self.fallen_disk_mass)
        if self.flim!=None:             f.create_dataset('flim',             compression=4, data=self.flim)
        if self.flim_dead!=None:        f.create_dataset('flim_dead',        compression=4, data=self.flim_dead)
        if self.gas_flux_o!=None:       f.create_dataset('gas_flux_o',       compression=4, data=self.gas_flux_o)
        if self.grainsizes!=None:       f.create_dataset('grainsizes',       compression=4, data=self.grainsizes)
        if self.m_dot_star!=None:       f.create_dataset('m_dot_star',       compression=4, data=self.m_dot_star)
        if self.m_grid!=None:           f.create_dataset('m_grid',           compression=4, data=self.m_grid)
        if self.m_star!=None:           f.create_dataset('m_star',           compression=4, data=self.m_star)
        if self.n_m!=None:              f.create_dataset('n_m',                             data=self.n_m)
        if self.n_r!=None:              f.create_dataset('n_r',                             data=self.n_r)
        if self.n_t!=None:              f.create_dataset('n_t',                             data=self.n_t)
        if self.nu!=None:               f.create_dataset('nu',               compression=4, data=self.nu)
        if self.peak_position!=None:    f.create_dataset('peak_position',    compression=4, data=self.peak_position)
        if self.r_centri!=None:         f.create_dataset('r_centri',         compression=4, data=self.r_centri)
        if self.r_min!=None:            f.create_dataset('r_min',            compression=4, data=self.r_min)
        if self.r_snow!=None:           f.create_dataset('r_snow',           compression=4, data=self.r_snow)
        if self.sig_dot_t!=None:        f.create_dataset('sig_dot_t',        compression=4, data=self.sig_dot_t)
        if self.sigma_coag!=None:       f.create_dataset('sigma_coag',       compression=4, data=self.sigma_coag)
        if self.sigma_d!=None:          f.create_dataset('sigma_d',          compression=4, data=self.sigma_d)
        if self.sigma_dead!=None:       f.create_dataset('sigma_dead',       compression=4, data=self.sigma_dead)
        if self.sigma_g!=None:          f.create_dataset('sigma_g',          compression=4, data=self.sigma_g)
        if self.steps!=None:            f.create_dataset('steps',            compression=4, data=self.steps)
        if self.timesteps!=None:        f.create_dataset('timesteps',        compression=4, data=self.timesteps)
        if self.v_dust!=None:           f.create_dataset('v_dust',           compression=4, data=self.v_dust)
        if self.v_gas!=None:            f.create_dataset('v_gas',            compression=4, data=self.v_gas)
        if self.v_gas_dead!=None:       f.create_dataset('v_gas_dead',       compression=4, data=self.v_gas_dead)
        if self.x!=None:                f.create_dataset('x',                compression=4, data=self.x)
        if self.x05!=None:              f.create_dataset('x05',              compression=4, data=self.x05)
        #
        # save the nml
        #
        if self.nml!=None:
            grp=f.create_group("nml")
            for key,val in self.nml.iteritems():
                grp.create_dataset(key, data=val)
        #
        # close file
        #
        f.close()
        print(' ... DONE')

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
# ============================================================================
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
            year          = d.year
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
                from IPython import embed
                embed()
            if any(sig_out<0):
                print('negative number detected in sig_out')
                from IPython import embed #@Reimport
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
                from IPython import embed #@Reimport
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
                from IPython import embed #@Reimport
                embed()
            if any(N<0.0):
                print('Error: negative number in N detected')
                from IPython import embed #@Reimport
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
                
def setup_diskev(sim_name,R,T,sig_g,alpha,inputvars,savedir='.',res=10):
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
        savetxt(savedir+os.sep+'T_input_'+    NAME+'.dat', T,     delimiter=' ')
        savetxt(savedir+os.sep+'alpha_input_'+NAME+'.dat', alpha, delimiter=' ')
