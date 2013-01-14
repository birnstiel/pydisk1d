def make_sim_interactive(d):
    """
    This function is to make a simulation result locally availabe in the interactive ipython/python environment.
    Note that in order to make it work, you cannot import it but you need to execute it like %run or execfile(...)
    """
    #
    # define the variables as global
    #
    global  grain1
    global  T
    global  accretion_dust
    global  accretion_dust_e
    global  accretion_gas
    global  alpha
    global  alpha_dead
    global  evap
    global  ta_dir
    global  st_flux_o
    global  st_flux_o_e
    global  fallen_disk_mass
    global  flim
    global  flim_dead
    global  gas_flux_o
    global  grainsizes
    global  m_dot_star
    global  m_grid
    global  m_star
    global  n_m
    global  n_r
    global  n_t
    global  nu
    global  peak_position
    global  r_centri
    global  r_min
    global  r_snow
    global  sig_dot_t
    global  sigma_coag
    global  sigma_d
    global  sigma_dead
    global  sigma_g
    global  steps
    global  timesteps
    global  v_dust
    global  v_gas
    global  v_gas_dead
    global  x
    global  x05
    global  nml
    #
    # set the variables
    #
    grain1              =    d.D_grain1            
    T                   =    d.T                   
    accretion_dust      =    d.accretion_dust      
    accretion_dust_e    =    d.accretion_dust_e    
    accretion_gas       =    d.accretion_gas       
    alpha               =    d.alpha               
    alpha_dead          =    d.alpha_dead          
    evap                =    d.d_evap              
    ta_dir              =    d.data_dir            
    st_flux_o           =    d.dust_flux_o         
    st_flux_o_e         =    d.dust_flux_o_e       
    fallen_disk_mass    =    d.fallen_disk_mass    
    flim                =    d.flim                
    flim_dead           =    d.flim_dead           
    gas_flux_o          =    d.gas_flux_o          
    grainsizes          =    d.grainsizes.squeeze()          
    m_dot_star          =    d.m_dot_star          
    m_grid              =    d.m_grid.squeeze()
    m_star              =    d.m_star              
    n_m                 =    int(d.n_m)                 
    n_r                 =    int(d.n_r)
    n_t                 =    int(d.n_t)                
    nu                  =    d.nu                  
    peak_position       =    d.peak_position       
    r_centri            =    d.r_centri            
    r_min               =    d.r_min               
    r_snow              =    d.r_snow              
    sig_dot_t           =    d.sig_dot_t           
    sigma_coag          =    d.sigma_coag          
    sigma_d             =    d.sigma_d
    sigma_dead          =    d.sigma_dead          
    sigma_g             =    d.sigma_g             
    steps               =    d.steps               
    timesteps           =    d.timesteps.squeeze()
    v_dust              =    d.v_dust              
    v_gas               =    d.v_gas               
    v_gas_dead          =    d.v_gas_dead          
    x                   =    d.x.squeeze()                  
    x05                 =    d.x05.squeeze()
    nml                 =    d.nml
    #
    # make the namelist variables also global
    #
    for name,val in nml.iteritems():
        exec name+'='+str(val) in globals()