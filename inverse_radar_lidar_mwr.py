#!/usr/bin/env python


import numpy as np
import ConfigParser
import sys
import pickle
from time import clock
from scipy.optimize import differential_evolution

from cloud_module import *
from mwr_module import *
from covariance_module import *    


def calc_beta_cal_arsl(lidarsetup,extcoeff,ra,z_ms,icld,cal,ext_co_air,ext_co_arsl,small=1,wide=1):
  """
  Construct input file for multiscatter and run multiscatter 
  Return (calibrated) attenuated lidar  backscatter profile
  Input: lidarsetup: a list of wavelength(m),instrument altitude(m),divergence(rad),FoV(rad),S[sr]
         extcoeff: extinction coeff inside cloud in 1/m (array) 
         ra: equivalent-area radius inside cloud in m (array)
         z_ms: range gates in meters  (array)
         icld: indices where cloud is present (index of the first range gate is zero)
         cal: calibration constant (dummy variable)
         ext_co_air: extinction coefficient of air in 1/m at z_ms (array)
         ext_co_arsl: attenuated backscatter due to aerosol
         small: code for 'fast' argument in 'run_multiscatter': 0(original),1(fast), or 2(eloranta)  
         wide: code for 'WA' argument in 'run_multiscatter': 0(no WA) or 1(with WA)
  Single scattering albedo of cloud:1
  Scattering asymmetry factor for the cloud:0.85
  Single scattering albedo of air:1
  """
  global clid
  nrange = z_ms.size
  wav = lidarsetup[0]
  alt = lidarsetup[1]
  div = lidarsetup[2]
  fov = lidarsetup[3]
  sfac_cld = lidarsetup[4]

  sp=' '
  newl='\n'

  tf = tempfile.NamedTemporaryFile(delete=False)   
  msfile = tf.name
  f = open(msfile,'w')
  f.write(str(nrange)+sp+str(wav)+sp+str(alt)+sp+str(div)+sp+str(fov)+newl)

  ra_ms = np.zeros(z_ms.size)+1e-8
  extco_ms = np.zeros(z_ms.size)
  cldfrac = np.zeros(z_ms.size)  

  s_arsl = 50.
  lidarratio = np.zeros(z_ms.size)
  lidarratio[icld] = sfac_cld
  lidarratio[0:ext_co_arsl.size-1] = s_arsl
  
  ra_ms[icld] = ra
  extco_ms[icld] = extcoeff
  extco_ms[0:ext_co_arsl.size]=ext_co_arsl[0:]  #klett reference point is cloud base

  cldfrac[icld[1:]] = 1.0

  extco_ms[extco_ms < 0.0] = 0.0    
  ra_ms[ra_ms <= 0.0] = 1.e-8       
  sfactor = sfactor_cld(0.0,z_ms,extco_ms,lidarratio) 

  for i in range(z_ms.size):   
    string = str(z_ms[i])+sp+str(extco_ms[i])+sp+str(ra_ms[i])+sp+str(sfactor[i])+sp+str(ext_co_air[i])+sp+'1.0'+sp+'0.85'+sp+'1.0'+sp+str(cldfrac[i])+sp+'0.0'+newl
    f.write(string)
  f.close()

  backscatter = run_multiscatter(msfile,fast=small,WA=wide,file=0)
  os.remove(msfile)

  tau_blwcld = np.zeros(ext_co_arsl.size)
  tau_blwcld[0] = (ext_co_arsl[0]+ext_co_air[0])*(z_ms[3]-z_ms[2])  
  tau_cum = tau_blwcld[0]

  for i in range(1,ext_co_arsl.size):
    tau_blwcld[i] = ((ext_co_arsl[i]+ext_co_air[i])*(z_ms[3]-z_ms[2]))+tau_cum  
    tau_cum = tau_blwcld[i]

  ext_tot_blwcld = (ext_co_arsl+s_arsl*ext_co_air[0:ext_co_arsl.size]*3/(8.*np.pi))
  cl_blwcld = s_arsl*beta_mean[0:ext_co_arsl.size]*np.exp(2*tau_blwcld)/ext_tot_blwcld
  clid = np.median(cl_blwcld)
  return backscatter*clid


def callback_func(xk,convergence):
  """ 
  Print out the progress of the optimization and allow for an early termination
    xk: state vector value at k-th iteration 
    convergence: fractional value of the convergence
  """
  global Niter,cost_value
  progf.write(str(Niter)+'  '+str(convergence)+'  '+str('%.2f'%(xk[0]))+'  '+str('%.2f'%(xk[1]))+'  '+str('%.2f'%(xk[2]))+'  '+str('%.2e'%(xk[3]))+'  '+str('%.2f'%(xk[4]))+'  '+str('%.2f'%(xk[5]))+'  '+str('%.2f'%(xk[6]))+'  '+str('%.2f'%(xk[7]))+'  '+str('%.2f'%(xk[8]))+'  '+str('%.2f'%(xk[9]))+'  '+str('%.2f'%(xk[10]))+'  '+str('%.2f'%(xk[11]))+'  '+str('%.2f'%(xk[12]))+'\n')
  Niter += 1
  
#  if (convergence > 0.05): return True   #early termination based on convergence level


def cost2(xv):
  """
  For the case where drizzle is not present below cloud base, allow the possibility that there is no drizzle at all
  Drizzle is derived using excess reflectivity; lwc_dzl within cloud is parametrized (Boers model)
  Cloud base is optimized(&smoothed)
  State vector:
  #        0       1        2       3     4      5         6           7         8       9       10        11       ,   12
  # x = [nu_cld,hhat_cld,alph_cld,Nad_cld,rcal,klettfac,cldtop_mod,cldbase_mod,nu_dzl,hhat_dzl,alph_dzl,lwcscale_dzl,weight] 

  """
  global cost_value
  global Nfeval,z_common,zcb,zct,dtop             
  global Ftb, Fbeta, Fradref, Fradref_dzl, Fradref_att               #observables
  global ext_cld_common,re_cld_common,lwc_cld_common,Nz_cld_common   #cloud
  global ext_dzl_common,re_dzl_common,lwc_dzl_common,Nz_dzl_common   #drizzle

  Nfeval = Nfeval+1
  zct = ctop+(xv[6]*gridres)                                 #zct<=ctop 
  zcb = cbase_mod_lo+(xv[7]*(cbase_mod_hi-cbase_mod_lo))             
  cdepth = zct-zcb
  Pcb = np.interp(zcb,tpqheight,presdata)
  Tcb = np.interp(zcb,tpqheight,tempdata)
  Gamma_l, rho_air = cloud_ad(Pcb,Tcb)

  #1st, compute LWC_ad and LWC_cld using lidar grid
  ind_cloud = [b for b,x in enumerate(height_beta) if (x>=zcb and x<=zct)] 
  zdata_cld = height_beta[ind_cloud]
  if (zdata_cld[-1] < zct): zdata_cld = np.append(zdata_cld,zct) 
  zdata_cld_cb = zdata_cld-zcb
  fz=subad_frac(zdata_cld_cb,cdepth,xv[2],xv[1])
  lwc_ad = rho_air*Gamma_l*zdata_cld_cb     #kg/m3 
  lwc_ad = lwc_ad*1000.                     #g/m3
  lwc_cld = fz*lwc_ad                       #g/m3; see eq. A2 of Boers et al.                
  Nz_cld,re_cld,ext_cld = calc_others(zdata_cld_cb,fz,xv[3],xv[0],Pcb,Tcb,mmod)  

  #2nd, construct weights for cloud base smoothing and exponential moving average on cloud LWC 
  nwidth = 2*(ind_cloud[0]-1-indl_cbase)+1    
  if (nwidth==1):    
    lwc_smooth= np.append(0,lwc_cld)
    Nz_smooth = np.append(Nz_cld[0],Nz_cld)
    z_smooth = np.append(height_beta[ind_cloud[0]-1],zdata_cld)
  else:              
    x=np.linspace(0,0.5*(nwidth-1),0.5*(nwidth+1))
    y = np.exp(-xv[12]*x)       
    weight = np.zeros(nwidth)  
    weight[int(0.5*(nwidth-1)):] = y[:]
    weight[range(int(0.5*(nwidth-1))-1,-1,-1)] = y[1:]

    lwc2 = np.zeros(lwc_cld.size+nwidth) #original lwc on extended grid 
    lwc2[nwidth:] = lwc_cld
    lwc_smooth = np.zeros(lwc_cld.size+0.5*(nwidth+1))   #include cbase
    extragates=0
    for i in range(nwidth+extragates):   
      lwc_smooth[i] = np.average(lwc2[i:i+nwidth],weights=weight)
    lwc_smooth[nwidth+extragates:] = lwc_cld[nwidth+extragates-int(0.5*(nwidth+1)):]  
  
    ind_extension = np.arange(int(0.5*(nwidth+1)),0,-1)    
    ind_temp = list(ind_cloud[0]-ind_extension)
    ind_smooth = ind_temp+ind_cloud
    z_smooth = height_beta[ind_smooth]      
    if (z_smooth[-1] < zct): z_smooth = np.append(z_smooth,zct)  
    ind_lo = int(0.5*(nwidth+1))
    Nz_smooth = np.zeros(Nz_cld.size+ind_lo)
    Nz_smooth[ind_lo:] = Nz_cld[:]
    Nz_smooth[0:ind_lo] = Nz_cld[0]         
  
  #3rd, compute other microphysical properties of cloud (ra, rn, ext)
  re_smooth = (1.5*lwc_smooth*(xv[0]+2)*(xv[0]+2)/(2*np.pi*Nz_smooth*1.e6*xv[0]*(xv[0]+1)))**(1./3)
  if (mmod == 0): re_smooth[-1] = re_smooth[-2]+(re_smooth[-2]-re_smooth[-3])  
  ra_smooth = re_smooth*np.sqrt(xv[0]*(xv[0]+1))/(xv[0]+2)
  rn_smooth = re_smooth/(xv[0]+2)
  ext_smooth = 2*np.pi*Nz_smooth*ra_smooth*ra_smooth
  
  #4th, interpolate lwc_smooth and Nz_smooth to radar grid and compute cloud reflectivity
  rn_interp = np.zeros(height_z.size)
  Nz_interp = np.zeros(height_z.size)
  indgrid = list(np.where(height_z>=z_smooth[0])[0])            
  rn_interp[indgrid] = np.interp(height_z[indgrid],z_smooth,rn_smooth)   
  Nz_interp[indgrid] = np.interp(height_z[indgrid],z_smooth,Nz_smooth) 
  Fradref = calc_radref_nu(rn_interp,Nz_interp,xv[0],xv[4])     
  maxpos_mod = np.where(Fradref == np.max(Fradref[-4:-1]))[0][0] 
  maxpos_obs = np.where(Z_mean == np.max(Z_mean[-4:-1]))[0][0]

  #5th, compute residual/excess reflectivity and smooth it to become drizzle reflectivity
  height_z2 = np.append(dbase,height_z)   
  residual = Z_mean-Fradref               
  residual[residual<0.0] = 0.0
  Fradref_dzl = np.zeros(residual.size)
  Fradref_dzl[0] = residual[0]  
  Fradref_dzl[-1] = residual[-1]  
  for i in range(1,residual.size-1):  
    Fradref_dzl[i] = np.mean(residual[i-1:i+2])    #mm6/m3
  ind_nonzero = np.array(np.where(Fradref_dzl > 0.0)[0])   
  diff_ind = ind_nonzero[1:]-ind_nonzero[0:-1]
  ind_diff_ind = list(np.where(diff_ind == 1)[0])
  condition1 = (len(ind_diff_ind) >= 2) 
  condition2 = (residual[0] > 0.0)      
  
  #6th, construct drizzle LWC and compute drizzle microphysical properties from Z_dzl and LWC_dzl (not smoothed)
  if (condition1 and condition2):                       #there is drizzle
    if (ind_nonzero[-1]+1==residual.size): dtop = zct   
    else: dtop = height_z[ind_nonzero[-1]+1]      
    ddepth = dtop-dbase              
    zdata_dzl_db = height_z2[height_z2<=dtop]-dbase  
    if (height_z2[-1] < dtop): zdata_dzl_db = np.append(zdata_dzl_db,dtop-dbase)  
    fz=subad_frac(zdata_dzl_db, ddepth,xv[10],xv[9])
    lwc_ad = rho_air*Gamma_l*zdata_dzl_db     #kg/m3 
    lwc_ad = lwc_ad*1000.                     #g/m3
    lwc_dzl = fz*lwc_ad*(10.**xv[11])
  
    k36_2 = xv[8]*(xv[8]+1)*(xv[8]+2)/(xv[8]+3)/(xv[8]+4)/(xv[8]+5)    
    nominator = np.pi*1.e6*Fradref_dzl[height_z<dtop]*1.e-18*(xv[8]+2)*(xv[8]+2)*(xv[8]+2)
    denominator = lwc_dzl[1:-1]*48.*(xv[8]+3)*(xv[8]+4)*(xv[8]+5)   
    re_dzl = (nominator/denominator)**(1./3.)   #m
    re_dzl_top = re_dzl[-1]+((zdata_dzl_db[-1]-zdata_dzl_db[-2])*(re_dzl[-1]-re_dzl[-2])/(zdata_dzl_db[-2]-zdata_dzl_db[-3]))
    re_dzl0 = re_dzl[0]+(re_dzl[0]-re_dzl[1])  

    re_dzl = np.append(np.append(re_dzl0,re_dzl),re_dzl_top)      
    ext_dzl = 1.5*lwc_dzl/1.e6/re_dzl
    rn_dzl = re_dzl/(xv[8]+2)
    ra_dzl = np.sqrt(rn_dzl*rn_dzl*xv[8]*(xv[8]+1))
    Nz_dzl = ext_dzl/(2.*np.pi*ra_dzl*ra_dzl)
    height_z3 = zdata_dzl_db+dbase

    #when there is no drizzle, drizzle size is irrelevant   
    re_dzl[lwc_dzl == 0.0] = 0.0   
    ra_dzl[lwc_dzl == 0.0] = 0.0   
    rn_dzl[lwc_dzl == 0.0] = 0.0   

    violation1 = (np.min(re_dzl[1:-1]) < re_dzl_lo)
    violation2 = (np.max(re_dzl[1:-1]) > re_dzl_hi)   
    radrefratio = Fradref_dzl[-2:]/Fradref[-2:]
    violation3 = (np.any(radrefratio > 1.0))   
    
    if (violation1 or violation2 or violation3): return 100.*cost_value    

    
    #7th, interpolate drizzle from radar to a common grid 

    height_extras = height_beta[(height_beta>dbase)&(height_beta<z_smooth[0])]
    z_common = np.sort(np.append(height_extras,np.append(dbase,z_smooth)))   
    #------cloud-------
    lwc_cld_common = np.zeros(z_common.size)
    ra_cld_common = np.zeros(z_common.size)
    re_cld_common = np.zeros(z_common.size)
    rn_cld_common = np.zeros(z_common.size)
    ext_cld_common = np.zeros(z_common.size)
    Nz_cld_common = np.zeros(z_common.size)
    lwc_cld_common = np.interp(z_common,z_smooth,lwc_smooth)     #dbase is not in z_smooth
    ra_cld_common = np.interp(z_common,z_smooth,ra_smooth)
    re_cld_common = np.interp(z_common,z_smooth,re_smooth)
    rn_cld_common = np.interp(z_common,z_smooth,rn_smooth)
    ext_cld_common = np.interp(z_common,z_smooth,ext_smooth)
    Nz_cld_common = np.interp(z_common,z_smooth,Nz_smooth)

    #------drizzle-------
    lwc_dzl_common = np.zeros(z_common.size)
    ra_dzl_common = np.zeros(z_common.size)
    re_dzl_common = np.zeros(z_common.size)
    rn_dzl_common = np.zeros(z_common.size)
    ext_dzl_common = np.zeros(z_common.size)
    Nz_dzl_common = np.zeros(z_common.size)
    indgrid = list(np.where((z_common>=dbase) & (z_common<=dtop))[0])
    lwc_dzl_common[indgrid] = np.interp(z_common[indgrid],height_z3,lwc_dzl)
    ra_dzl_common[indgrid] = np.interp(z_common[indgrid],height_z3,ra_dzl)
    re_dzl_common[indgrid] = np.interp(z_common[indgrid],height_z3,re_dzl)
    rn_dzl_common[indgrid] = np.interp(z_common[indgrid],height_z3,rn_dzl)
    ext_dzl_common[indgrid] = np.interp(z_common[indgrid],height_z3,ext_dzl)
    Nz_dzl_common[indgrid] = np.interp(z_common[indgrid],height_z3,Nz_dzl)

  else:   #no drizzle present!
    
    height_extras = height_beta[(height_beta>dbase)&(height_beta<z_smooth[0])]
    z_common = np.sort(np.append(height_extras,np.append(dbase,z_smooth)))   
    #------cloud-------
    lwc_cld_common = np.zeros(z_common.size)
    ra_cld_common = np.zeros(z_common.size)
    re_cld_common = np.zeros(z_common.size)
    rn_cld_common = np.zeros(z_common.size)
    ext_cld_common = np.zeros(z_common.size)
    Nz_cld_common = np.zeros(z_common.size)
    lwc_cld_common = np.interp(z_common,z_smooth,lwc_smooth)     
    ra_cld_common = np.interp(z_common,z_smooth,ra_smooth)
    re_cld_common = np.interp(z_common,z_smooth,re_smooth)
    rn_cld_common = np.interp(z_common,z_smooth,rn_smooth)
    ext_cld_common = np.interp(z_common,z_smooth,ext_smooth)
    Nz_cld_common = np.interp(z_common,z_smooth,Nz_smooth)

    lwc_dzl_common = np.zeros(z_common.size)
    ra_dzl_common = np.zeros(z_common.size)
    re_dzl_common = np.zeros(z_common.size)
    rn_dzl_common = np.zeros(z_common.size)
    ext_dzl_common = np.zeros(z_common.size)
    Nz_dzl_common = np.zeros(z_common.size)
    Fradref_dzl = Fradref_dzl*0.0

  #8th, combine cloud and rizzle parameters on the common grid 
  lwc_all_common = lwc_cld_common+lwc_dzl_common
  ext_all_common =ext_cld_common+ext_dzl_common 
  ra_all_common = (ra_cld_common*ext_cld_common+ra_dzl_common*ext_dzl_common)/(ext_cld_common+ext_dzl_common) 
  re_all_common = (re_cld_common*ext_cld_common+re_dzl_common*ext_dzl_common)/(ext_cld_common+ext_dzl_common)  
  rn_all_common = (rn_cld_common*ext_cld_common+rn_dzl_common*ext_dzl_common)/(ext_cld_common+ext_dzl_common)  
  ind0 = list(np.where(ext_all_common == 0.0)[0])
  re_all_common[ind0] = 0.0
  rn_all_common[ind0] = 0.0
  ra_all_common[ind0] = 0.0
  Nz_all_common = Nz_cld_common+Nz_dzl_common     

    
  #9th, compute Ftb    
  lwc30 = np.zeros(n30)
  z_com_ic = np.zeros(z_common.size)    
  lwc_com_ic = np.zeros(z_common.size)
  z_com_ic[:] = z_common[:]           
  lwc_com_ic[:] = lwc_all_common[:]   

  bad_elements = list(set(z_com_ic)-set(height_incld))  
  for element in bad_elements:   
    ind = np.where(element == z_com_ic)[0][0]
    z_com_ic = np.delete(z_com_ic,ind)    
    lwc_com_ic = np.delete(lwc_com_ic,ind)

  lwc30[n_blwcb:n_blwcb+len(ind_beta_incld)] = lwc_com_ic

  tau_cld = calc_tau_cloud(height30,temp30,lwc30,freq_brt,liq_abs)
  tau = tau_cld + tau_gas_total
  Ftb = calc_tb(temp30,tau,mu,freq_brt)

  
  #10th, Klett method to derive extinction coefficient due to aerosols
  indd = indl_dbase-1 if (dbase < cbase) else indl_cbase  
  z0 = height_beta[indd]           
  beta0 = beta_mean[indd]          
  ext_arsl0 = 10.**(xv[5])                 
  beta_air = ext_air/s_air
  beta_air0 = beta_air[indd]       
  ext_tot0 = ext_arsl0+s_arsl*beta_air0 

  transmission = np.zeros(indd+1)
  transmission[0] = 2.*(ext_air[0]-s_arsl*beta_air[0])*(height_beta[0]-0.0)  
  for i in range(1,indd+1):
    transmission[i] = transmission[i-1]+(2.*(ext_air[i]-s_arsl*beta_air[i])*(height_beta[i]-height_beta[i-1]))  
  ext_tot = np.zeros(indd+1)
  ext_tot[-1] = ext_tot0
  ext_arsl = np.zeros(indd+1)
  ext_arsl[-1] = ext_arsl0
  pz0 = beta0*np.exp(transmission[-1])   
  pz = np.zeros(indd+1)         
  pz[-1] = pz0
  pz_cum = 2.*pz0*(z0-height_beta[indd-1])     
  
  denominator1 = pz0/ext_tot0
  for i in range(indd-1,-1,-1):  
    pz[i] = beta_mean[i]*np.exp(transmission[i])
    if (i == 0): pz_cum = pz_cum+2.*pz[i]*(height_beta[i]-0.0) 
    else: pz_cum = pz_cum+2.*pz[i]*(height_beta[i]-height_beta[i-1]) 

    ext_tot[i] = pz[i]/(denominator1+pz_cum)
    ext_arsl[i] = ext_tot[i]-s_arsl*beta_air[i]

    
  #---clean up grid  
  indbeta = list(np.where((height_beta >= cbase) & (height_beta <= zct))[0])
  z_com_ic = np.zeros(z_common.size)
  ext_total = np.zeros(z_common.size)
  ra_total = np.zeros(z_common.size)
  z_com_ic[:] = z_common[:]
  ext_total = ext_all_common[:]
  ra_total = ra_all_common[:]
  bad_elements = list(set(z_com_ic)-set(height_beta[indbeta]))  
  for element in bad_elements:   
    ind = np.where(element == z_com_ic)[0][0]
    z_com_ic = np.delete(z_com_ic,ind)    
    ext_total = np.delete(ext_total,ind)
    ra_total = np.delete(ra_total,ind)

  #11th, compute lidar forward model and cost function
  Fbeta = calc_beta_cal_arsl([wav,alt,div,fov,ratio],ext_total,ra_total,height_beta,indbeta,xv[5],ext_air,ext_arsl,small=0,wide=0)

  ffx = np.zeros(ntb+nbeta+nrref)    
  ffx[0:ntb] = Ftb
  ffx[ntb:ntb+nbeta] = Fbeta[0:indl_200+1]

  lwc_heightz = np.interp(height_z,z_common,lwc_all_common) 
  Fradref_att = attenuation(radarfreq,temp_heightz,gridres,lwc_heightz,Fradref+Fradref_dzl)
  ffx[ntb+nbeta:ntb+nbeta+nrref] = Fradref_att 

  diff = yy[0:ntb+nbeta]-ffx[0:ntb+nbeta]
  tempfac = dot(cov_msr_inv[0:ntb+nbeta,0:ntb+nbeta],diff)
  trans = np.transpose(diff)
  cost_value = dot(trans,tempfac)

  diff = yy-ffx
  sumz = 0.0
  for i in range(ntb+nbeta,ntb+nbeta+nrref):
    for j in range(ntb+nbeta,ntb+nbeta+nrref):
       term = cov_msr_inv[i,j]*diff[i]*diff[j]
       sumz = sumz + term
 
  ind_new_cld = list(np.where((z_common >= cbase) & (z_common <= zct))[0]) 
  delta_re = re_dzl_common[ind_new_cld[1]:ind_new_cld[-1]+1]-re_dzl_common[ind_new_cld[0]:ind_new_cld[-1]]
  ind_pos = np.where(delta_re > 0.0)[0]
  npenalty3 = len(ind_pos)   

  cost_value = cost_value+sumz*(1+npenalty3)

  cff.write(str(Nfeval)+'   '+str(cost_value)+'\n')   

  return cost_value

  
  
def cost1(xv):
  """
  For the case where drizzle extends below cloud base
  Drizzle is derived using excess reflectivity; re_dzl within cloud is parametrised (exponential)
  Cloud base is optimized(&smoothed)
  State vector:
  #        0       1        2       3     4      5         6           7         8           9           10      ,    11       12
  #x = [nu_cld,hhat_cld,alph_cld,Nad_cld,rcal,klettfac,cldtop_mod,cldbase_mod,nu_dzl,ext_dzl_dbase,ext_dzl_cbase, ext_dzl_peak,weight] 
  #  x[9] = -3 to -0.001  (to force it to be smaller than ext_dzl_cbase; ext_dzl_dbase = x[9] * ext_dzl_cbase; ext_dzl_dbase means ext_dzl at one range gate above dbase)  
  #  x[10] = -9 to -6  (ext_dzl_cbase = 10.**x[10])
  #  x[11] = -4 to -2  (ext_dzl_peak = x[11]*np.max(ext_cld)

  """
  global cost_value
  global Nfeval,flag_nan,z_common,zcb,zct,dtop              
  global Ftb, Fbeta, Fradref, Fradref_dzl, Fradref_att               #observables
  global ext_cld_common,re_cld_common,lwc_cld_common,Nz_cld_common   #cloud
  global ext_dzl_common,re_dzl_common,lwc_dzl_common,Nz_dzl_common   #dzl

  if (np.isnan(xv[6]) and seed_de==8):
    flag_nan = 1
    return True    
  elif (np.isnan(xv[6]) and seed_de==10):
    flag_nan = 10
    return True    
   
  Nfeval = Nfeval+1
  zct = ctop+(xv[6]*gridres)                                   
  zcb = cbase_mod_lo+(xv[7]*(cbase_mod_hi-cbase_mod_lo))       
  cdepth = zct-zcb
  Pcb = np.interp(zcb,tpqheight,presdata)
  Tcb = np.interp(zcb,tpqheight,tempdata)
  Gamma_l, rho_air = cloud_ad(Pcb,Tcb)
 
  #1st, compute LWC_ad and LWC_cld using lidar grid
  ind_cloud = [b for b,x in enumerate(height_beta) if (x>=zcb and x<=zct)] 
  zdata_cld = height_beta[ind_cloud]
  if (zdata_cld[-1] < zct): zdata_cld = np.append(zdata_cld,zct) 
  zdata_cld_cb = zdata_cld-zcb
  fz=subad_frac(zdata_cld_cb,cdepth,xv[2],xv[1])
  lwc_ad = rho_air*Gamma_l*zdata_cld_cb     #kg/m3 
  lwc_ad = lwc_ad*1000.                     #g/m3
  lwc_cld = fz*lwc_ad                       #g/m3; see eq. A2 of Boers et al.                
  Nz_cld,re_cld,ext_cld = calc_others(zdata_cld_cb,fz,xv[3],xv[0],Pcb,Tcb,mmod)  

  #2nd, construct weights for cloud base smoothing,exponential moving average on cloud LWC around the cloud base
  nwidth = 2*(ind_cloud[0]-1-indl_cbase)+1    
  if (nwidth==1):    
    lwc_smooth= np.append(0,lwc_cld)
    Nz_smooth = np.append(Nz_cld[0],Nz_cld)
    z_smooth = np.append(height_beta[ind_cloud[0]-1],zdata_cld)
  else:              
    x=np.linspace(0,0.5*(nwidth-1),0.5*(nwidth+1))
    y = np.exp(-xv[12]*x)       
    weight = np.zeros(nwidth)  
    weight[int(0.5*(nwidth-1)):] = y[:]
    weight[range(int(0.5*(nwidth-1))-1,-1,-1)] = y[1:]

    lwc2 = np.zeros(lwc_cld.size+nwidth) 
    lwc2[nwidth:] = lwc_cld
    lwc_smooth = np.zeros(lwc_cld.size+0.5*(nwidth+1))   
    extragates=0
    for i in range(nwidth+extragates):   
      lwc_smooth[i] = np.average(lwc2[i:i+nwidth],weights=weight)
    lwc_smooth[nwidth+extragates:] = lwc_cld[nwidth+extragates-int(0.5*(nwidth+1)):]  
  
    ind_extension = np.arange(int(0.5*(nwidth+1)),0,-1)    
    ind_temp = list(ind_cloud[0]-ind_extension)
    ind_smooth = ind_temp+ind_cloud
    z_smooth = height_beta[ind_smooth]      
    if (z_smooth[-1] < zct): z_smooth = np.append(z_smooth,zct)  
    ind_lo = int(0.5*(nwidth+1))
    Nz_smooth = np.zeros(Nz_cld.size+ind_lo)
    Nz_smooth[ind_lo:] = Nz_cld[:]
    Nz_smooth[0:ind_lo] = Nz_cld[0]                   
    
  #3rd, compute other microphysical properties of cloud (ra, rn, ext)
  re_smooth = (1.5*lwc_smooth*(xv[0]+2)*(xv[0]+2)/(2*np.pi*Nz_smooth*1.e6*xv[0]*(xv[0]+1)))**(1./3)
  if (mmod == 0): re_smooth[-1] = re_smooth[-2]+(re_smooth[-2]-re_smooth[-3])  
  ra_smooth = re_smooth*np.sqrt(xv[0]*(xv[0]+1))/(xv[0]+2)
  rn_smooth = re_smooth/(xv[0]+2)
  ext_smooth = 2*np.pi*Nz_smooth*ra_smooth*ra_smooth

  
  #4th, interpolate lwc_smooth and Nz_smooth to radar grid and compute cloud reflectivity
  rn_interp = np.zeros(height_z.size)
  Nz_interp = np.zeros(height_z.size)
  indgrid = list(np.where(height_z>=z_smooth[0])[0])     
  rn_interp[indgrid] = np.interp(height_z[indgrid],z_smooth,rn_smooth)   
  Nz_interp[indgrid] = np.interp(height_z[indgrid],z_smooth,Nz_smooth) 
  Fradref = calc_radref_nu(rn_interp,Nz_interp,xv[0],xv[4])        
  maxpos_mod = np.where(Fradref == np.max(Fradref[-4:-1]))[0][0] 
  maxpos_obs = np.where(Z_mean == np.max(Z_mean[-4:-1]))[0][0]

  #5th, compute residual/excess reflectivity and smooth it to become drizzle reflectivity
  residual = Z_mean-Fradref   
  residual[residual<0.0] = 0.0
  Fradref_dzl = np.zeros(residual.size)
  Fradref_dzl[0:indr_cbase2] = residual[0:indr_cbase2]  
  Fradref_dzl[-1] = residual[-1]  
  for i in range(indr_cbase2,residual.size-1):  
    Fradref_dzl[i] = np.mean(residual[i-1:i+2])    #mm6/m3  
  ind_posres = np.where(Fradref_dzl > 0.0)[0][-1]   

  radrefratio = Fradref_dzl[-2:]/Fradref[-2:]
  
  
  #6th, construct drizzle using Fradref_dzl and parametrized re_dzl
  #first for the re profile above cloud base
  ext_dzl_cbase = 10.**(xv[10])
  nominator = 2*np.pi*((xv[8]+2)**3)*Z_cbase*1.e-18/xv[4]
  denominator= 64.*ext_dzl_cbase*(xv[8]+3)*(xv[8]+4)*(xv[8]+5)  
  re_dzl_cbase = (nominator/denominator)**(0.25)  
  ind150 = np.where(height_z-cbase <=np.min([150.,height_z[ind_posres-1]-cbase]))[0][-1]   
  ext150 = np.interp(height_z[ind150],z_smooth,ext_smooth)
  ext_dzl_peak = (10.**(xv[11]))*ext150 
  nominator = 2*np.pi*((xv[8]+2)**3)*Fradref_dzl[ind150]*1.e-18/xv[4]
  denominator= 64.*ext_dzl_peak*(xv[8]+3)*(xv[8]+4)*(xv[8]+5)  
  re_dzl_peak = (nominator/denominator)**(0.25)    

  violation1 = (maxpos_mod != maxpos_obs)   
  violation2 = (np.max(re_cld) > re_dzl_lo) 
  violation3 = (np.any(radrefratio > 1.0))   
  violation4 = (re_dzl_peak > re_dzl_cbase) 
  violation6 = (np.all(Fradref_dzl[indr_cbase2:] == 0.0))  
  if (violation1 or violation2 or violation3 or violation4 or violation6):
    cost_value = 100.*cost_value
    cff.write(str(Nfeval)+'  '+str(cost_value)+'  1st\n')   
    return cost_value

  if (ind_posres < Fradref.size-1):
    dtop = height_z[ind_posres+1]                  
    height_z2 = np.append(cbase,height_z[indr_cbase2:ind_posres+2])  
  else:
    dtop = zct                                    
    height_z2 = np.append(cbase,np.append(height_z[indr_cbase2:],dtop))  
  norm_height = (height_z2-cbase)/(dtop-cbase)     

  ind_Zpeak = np.where(Fradref_dzl == np.max(Fradref_dzl[indr_cbase2:]))[0][-1]  
  y_top = (height_z[ind150]-cbase)/(dtop-cbase)
  bslope = ((-1*np.log(re_dzl_peak/re_dzl_cbase))**2)/y_top
  re_dzl_incld = re_dzl_cbase*np.exp(-np.sqrt(bslope*norm_height))  
  
  ext_dzl_blw = (10.**(xv[9])) * ext_dzl_cbase   
  nominator = 2*np.pi*((xv[8]+2)**3)*Fradref_dzl[0]*1.e-18/xv[4]
  denominator= 64.*ext_dzl_blw*(xv[8]+3)*(xv[8]+4)*(xv[8]+5)  
  re_dzl_blw = (nominator/denominator)**(0.25)    
  
  norm_height = (np.append(dbase,height_z[0:indr_cbase1+1])-dbase)/(cbase-dbase)  
  y_base = (height_z[0]-dbase)/(cbase-dbase)   
  bslope = np.log(re_dzl_blw/re_dzl_cbase)/np.log(y_base)  
  re_dzl_blwcld = re_dzl_cbase*(norm_height**bslope)   
  re_dzl_blwcld[0] = 0.0
  
  re_dzl = np.append(re_dzl_blwcld,re_dzl_incld)   

  if (indr_cbase2-indr_cbase1 == 2): nominator = 2*np.pi*((xv[8]+2)**3)*Fradref_dzl[0:ind_posres+1]*1.e-18/xv[4]            
  elif (indr_cbase2-indr_cbase1 == 1): 
    radref_dzl = np.insert(Fradref_dzl[0:ind_posres+1],indr_cbase2,Z_cbase)  
    nominator = 2*np.pi*((xv[8]+2)**3)*radref_dzl*1.e-18/xv[4]        

  denominator= 64.*(re_dzl[1:-1]**4)*(xv[8]+3)*(xv[8]+4)*(xv[8]+5)    
  ext_dzl = np.append(np.append(0.0,nominator/denominator),0.0)       

  violation5 = (np.isnan(re_dzl).any() or np.max(re_dzl) > re_dzl_hi)   
  if (y_base > 0.5): violation7 = (re_dzl_blw < 0.0)     
  else: violation7 = (re_dzl_blw > re_dzl_cbase)       

  violation8 = (np.min(re_dzl[1:-1]) < re_dzl_lo)   
  if (violation7 or violation8 or violation5):
    cost_value = 100.*cost_value
    cff.write(str(Nfeval)+'  '+str(cost_value)+'  2nd\n') 
    return cost_value
  
  rn_dzl = re_dzl/(xv[8]+2)                       
  ra_dzl = np.sqrt(rn_dzl*rn_dzl*xv[8]*(xv[8]+1))
  lwc_dzl = 1.e6*re_dzl*ext_dzl/1.5               
  Nz_dzl = ext_dzl/(2.*np.pi*ra_dzl*ra_dzl)       
  Nz_dzl[ra_dzl == 0.0] = 0.0

  re_dzl[lwc_dzl == 0.0] = 0.0   #no drizzle, drizzle size is irrelevant
  ra_dzl[lwc_dzl == 0.0] = 0.0   
  rn_dzl[lwc_dzl == 0.0] = 0.0   
  
 
  #7th, combine drizzle and cloud into a common grid
  #------drizzle-------
  z_common = np.append(height_beta[indl_dbase:indl_cbase],z_smooth) #cloud base and cloud top are already in z_smooth
  if (z_common[0]>dbase):z_common = np.append(dbase,z_common) #cloud base, drizzle base, cloud top are in z_common ; drizzle top not
  if (dtop != zct and dtop not in z_common):
    ztemp = np.append(z_common,dtop)
    z_common = np.sort(ztemp)

  lwc_dzl_common = np.zeros(z_common.size)
  ra_dzl_common = np.zeros(z_common.size)
  re_dzl_common = np.zeros(z_common.size)
  rn_dzl_common = np.zeros(z_common.size)
  ext_dzl_common = np.zeros(z_common.size)
  Nz_dzl_common = np.zeros(z_common.size)
  indgrid = list(np.where(z_common<=dtop)[0])

  height_z3 = np.append(np.append(dbase,height_z[0:indr_cbase1+1]),height_z2)  #height_z3 includes dbase,cbase,dtop
  lwc_dzl_common[indgrid] = np.interp(z_common[indgrid],height_z3,lwc_dzl)

  ra_dzl_common[indgrid] = np.interp(z_common[indgrid],height_z3,ra_dzl)
  re_dzl_common[indgrid] = np.interp(z_common[indgrid],height_z3,re_dzl)
  rn_dzl_common[indgrid] = np.interp(z_common[indgrid],height_z3,rn_dzl)
  ext_dzl_common[indgrid] = np.interp(z_common[indgrid],height_z3,ext_dzl)  
  Nz_dzl_common[indgrid] = np.interp(z_common[indgrid],height_z3,Nz_dzl)

  #------cloud-------
  lwc_cld_common = np.zeros(z_common.size)
  ra_cld_common = np.zeros(z_common.size)
  re_cld_common = np.zeros(z_common.size)
  rn_cld_common = np.zeros(z_common.size)
  ext_cld_common = np.zeros(z_common.size)
  Nz_cld_common = np.zeros(z_common.size)
  indgrid = list(np.where((z_common>=cbase) & (z_common<=zct))[0])
  lwc_cld_common[indgrid] = np.interp(z_common[indgrid],z_smooth,lwc_smooth)
  ra_cld_common[indgrid] = np.interp(z_common[indgrid],z_smooth,ra_smooth)
  re_cld_common[indgrid] = np.interp(z_common[indgrid],z_smooth,re_smooth)
  rn_cld_common[indgrid] = np.interp(z_common[indgrid],z_smooth,rn_smooth)
  ext_cld_common[indgrid] = np.interp(z_common[indgrid],z_smooth,ext_smooth)
  Nz_cld_common[indgrid] = np.interp(z_common[indgrid],z_smooth,Nz_smooth)
  
  #-------combined cloud and drizzle properties
  lwc_all_common = lwc_cld_common+lwc_dzl_common
  ext_all_common = ext_cld_common+ext_dzl_common 
  ra_all_common = (ra_cld_common*ext_cld_common+ra_dzl_common*ext_dzl_common)/(ext_cld_common+ext_dzl_common) 
  re_all_common = (re_cld_common*ext_cld_common+re_dzl_common*ext_dzl_common)/(ext_cld_common+ext_dzl_common)  
  rn_all_common = (rn_cld_common*ext_cld_common+rn_dzl_common*ext_dzl_common)/(ext_cld_common+ext_dzl_common)  
  ind0 = list(np.where(ext_all_common == 0.0)[0])
  re_all_common[ind0] = 0.0
  rn_all_common[ind0] = 0.0
  ra_all_common[ind0] = 0.0
  Nz_all_common = Nz_cld_common+Nz_dzl_common     

  #8th, compute Ftb    
  lwc30 = np.zeros(n30)
  z_com_ic = np.zeros(z_common.size)    
  lwc_com_ic = np.zeros(z_common.size)
  z_com_ic[:] = z_common[:]           
  lwc_com_ic[:] = lwc_all_common[:]   

  bad_elements = list(set(z_com_ic)-set(height_incld))  
  for element in bad_elements:   
    ind = np.where(element == z_com_ic)[0][0]
    z_com_ic = np.delete(z_com_ic,ind)   
    lwc_com_ic = np.delete(lwc_com_ic,ind)

  lwc30[n_blwcb:n_blwcb+len(ind_beta_incld)] = lwc_com_ic

  tau_cld = calc_tau_cloud(height30,temp30,lwc30,freq_brt,liq_abs)
  tau = tau_cld + tau_gas_total
  Ftb = calc_tb(temp30,tau,mu,freq_brt)

  
  #9th, Klett method to infer aerosol extinction coefficient
  #(for region between ground and cloud top where neither cloud nor drizzle is present)                                   
  z0 = height_beta[indl_dbase-1]      
  beta0 = beta_mean[indl_dbase-1]     
  ext_arsl0 = 10.**(xv[5])            
  beta_air = ext_air/s_air
  beta_air0 = beta_air[indl_dbase-1]  
  ext_tot0 = ext_arsl0+s_arsl*beta_air0 

  transmission = np.zeros(indl_dbase)
  transmission[0] = 2.*(ext_air[0]-s_arsl*beta_air[0])*(height_beta[0]-0.0)  
  for i in range(1,indl_dbase):
    transmission[i] = transmission[i-1]+(2.*(ext_air[i]-s_arsl*beta_air[i])*(height_beta[i]-height_beta[i-1])) 
  ext_tot = np.zeros(indl_dbase)
  ext_tot[-1] = ext_tot0
  ext_arsl = np.zeros(indl_dbase)
  ext_arsl[-1] = ext_arsl0
  pz0 = beta0*np.exp(transmission[-1])   
  pz = np.zeros(indl_dbase)         
  pz[-1] = pz0
  pz_cum = 2.*pz0*(z0-height_beta[indl_dbase-2])    
  
  denominator1 = pz0/ext_tot0
  for i in range(indl_dbase-2,-1,-1):  
    pz[i] = beta_mean[i]*np.exp(transmission[i])
    if (i == 0): pz_cum = pz_cum+2.*pz[i]*(height_beta[i]-0.0)   
    else: pz_cum = pz_cum+2.*pz[i]*(height_beta[i]-height_beta[i-1])  

    ext_tot[i] = pz[i]/(denominator1+pz_cum)
    ext_arsl[i] = ext_tot[i]-s_arsl*beta_air[i]

  #---now delete elements in z_common that are not present in height_beta (e.g. drizzle base, cloud top, drizzle top)       
  indbeta = list(np.where((height_beta >= dbase) & (height_beta <= zct))[0])
  z_com_ic = np.zeros(z_common.size)
  ext_total = np.zeros(z_common.size)
  ra_total = np.zeros(z_common.size)
  z_com_ic[:] = z_common[:]
  ext_total = ext_all_common[:]
  ra_total = ra_all_common[:]
  bad_elements = list(set(z_com_ic)-set(height_beta[indbeta]))  
  for element in bad_elements:   
    ind = np.where(element == z_com_ic)[0][0]
    z_com_ic = np.delete(z_com_ic,ind)    
    ext_total = np.delete(ext_total,ind)
    ra_total = np.delete(ra_total,ind)

  Fbeta = calc_beta_cal_arsl([wav,alt,div,fov,ratio],ext_total,ra_total,height_beta,indbeta,xv[5],ext_air,ext_arsl,small=0,wide=0)

#11th, compute the cost value
  ffx = np.zeros(ntb+nbeta+nrref)         
  extras = np.zeros(Fradref_dzl.size)
  extras[-2:] = Fradref_dzl[-2:]
  
  ffx[0:ntb] = Ftb
  ffx[ntb:ntb+nbeta] = Fbeta[0:indl_200+1]

  #with attenuation  
  lwc_heightz = np.interp(height_z,z_common,lwc_all_common) 
  Fradref_att = attenuation(radarfreq,temp_heightz,gridres,lwc_heightz,Fradref+Fradref_dzl)
  ffx[ntb+nbeta:ntb+nbeta+nrref] = Fradref_att + extras

  diff = yy-ffx
  tempfac = dot(cov_msr_inv,diff)
  trans = np.transpose(diff)
  cost_value = dot(trans,tempfac)
  
  cff.write(str(Nfeval)+'   '+str(cost_value)+'\n')   
 
  return cost_value



if __name__ == "__main__":
  theta = 0.0                    #MWR points to zenith
  s_air = 8.*np.pi/3.            #lidar ratio for air molecules
  s_arsl = 50.                   #lidar ratio for aerosols
      
  re_dzl_lo = 13.e-6             #lower threshold of drizzle droplet effective radius [m]
  re_dzl_hi = 150.e-6            #upper threshold of drizzle droplet effective radius [m]

  if (len(sys.argv) == 1):
    print "--------- INCOMPLETE ARGUMENTS: provide the profile number to invert ----------"
    exit()
    
  elif (len(sys.argv) == 2):     #invert only one profile
    atnumber = int(sys.argv[1])
    endcount = atnumber
    
  elif (len(sys.argv) > 2):      #invert profiles from 'atnumber' until 'endcount'
    atnumber = int(sys.argv[1])
    endcount = int(sys.argv[2])


#---------------------Read configfile-------------------------
  config = ConfigParser.ConfigParser()
  config.read('configfile')

  infilename = config.get('options','infile')
  tpqfilename = config.get('options','tpqfile')
  infix = config.get('options','infix')
  calc_error = int(config.get('options','calc_error'))
  
  mmod = int(config.get('mixmodel','mmod'))

  wav = float(config.get('lidar','wavelength'))        #m 
  alt = float(config.get('lidar','altitude'))          #m
  div = float(config.get('lidar','divergence'))        #rad
  fov = float(config.get('lidar','fieldview'))         #rad
  ratio = float(config.get('lidar','lidarratio'))      #sr

  radarfreq = float(config.get('radar','frequency'))   #GHz
  
  liq_abs = config.get('MWR','liq_abs')
  gas_abs = config.get('MWR','gas_abs')

  max_it = int(config.get('DE','max_it'))
  recomb = float(config.get('DE','recomb'))
  popsizef = int(config.get('DE','popsizef'))
  b_str = config.get('DE','mutationf')
  b_list = [float(n) for n in b_str.split(',')] 
  mutationf = tuple(b_list)
  tolerance = float(config.get('DE','tolerance'))
  
  b_str = config.get('boundaries1','lb1')
  b_list = [float(n) for n in b_str.split(',')]
  lb1 = np.array(b_list)

  b_str = config.get('boundaries1','ub1')
  b_list = [float(n) for n in b_str.split(',')]
  ub1 = np.array(b_list)

  b_str = config.get('boundaries2','lb2')
  b_list = [float(n) for n in b_str.split(',')]
  lb2 = np.array(b_list)

  b_str = config.get('boundaries2','ub2')
  b_list = [float(n) for n in b_str.split(',')]
  ub2 = np.array(b_list)
  
  
#---------------------------Load data------------------------------
  count = 0  

  #prepare file containing radar, lidar and MWR data 
  infile = open(infilename,'r')
   
  #prepare file containing T,P,Q data (up to at least 30 km)
  tpqfile = open(tpqfilename,'r')
  tpqheight = pickle.load(tpqfile)

  sys.stderr.write("Start: count "+str(atnumber)+"\n")
  sys.stderr.write("End: count "+str(endcount)+"\n")

  while 1:
    try:    
      along_track,height_z,Z_mean,Z_err,height_beta,beta_mean,beta_err,freq_brt,tb_mean,tb_err,cbase_mean = pickle.load(infile)  
      along_track_tpq,tempdata,presdata,humdata = pickle.load(tpqfile)
      temp_heightz = np.interp(height_z,tpqheight,tempdata)
      gridres = height_z[1]-height_z[0]
      
      if (count < atnumber):
         count += 1
         continue
      elif (count > endcount):
        print ("The end")
        sys.stderr.write("The end\n")        
        infile.close()
        tpqfile.close()
        exit()
       
      print '\n============Count: ',count, '================='

      if (height_beta[0] >= height_z[0]):    
        sys.stderr.write("--------- ERROR: There is no beta data below the first radar gate; count "+str(count)+" ----------\n")

        count += 1
        continue     

      cffile = 'cf'+infix+str(count)             #cost function values
      sgnfile = 'sgn'+infix+str(count)           #signal (observation and model)
      progfile = 'prog'+infix+str(count)         #progress (convergence fraction and state vector elements)
      retfile = 'ret'+infix+str(count)           #cloud and drizzle microphysical properties 
      
      #-----estimate the cloud base height of the mean beta profile-----
      ind_max =list(np.where(beta_mean == np.max(beta_mean))[0])[0] 
      ind_search = range(ind_max-10,ind_max) 
      beta1 = beta_mean[0:-1]
      beta2 = beta_mean[1:]
      ind_ratio=list(np.where(beta2/beta1 > 1.5)[0])
      ind_intersect = sorted(list(set(ind_ratio)&set(ind_search)))
      flag_cbase = 0    
      for i in range(len(ind_intersect)):
        indl_cbase = ind_intersect[i]             
        beta1 = beta_mean[indl_cbase:ind_max]                                                                                    
        beta2 = beta_mean[indl_cbase+1:ind_max+1]                                                                                
        if (np.all(beta2/beta1 > 1.0)):                                                                                        
          cbase = height_beta[indl_cbase]
          flag_cbase = 1
          break
        else:
          continue     

      if (flag_cbase == 0):
        sys.stderr.write("--------- ERROR: cloud base is not found; count "+str(count)+" ----------\n")
        count += 1
        continue   #the next profile  
        
      if (ind_max-indl_cbase < 3):
        cbase_mod_lo = cbase
        cbase_mod_hi = height_beta[indl_cbase+1]
      elif (ind_max-indl_cbase >= 3):
        cbase_mod_lo = height_beta[indl_cbase+1]
        cbase_mod_hi = height_beta[ind_max-1]

      indr_cbase2 = np.where(height_z > cbase)[0][0]     #index of the lowest radar gate located above cbase
       
      dbase = height_z[0]-gridres
      ctop = height_z[-1] + gridres                      #estimate of cloud top

      indl_ctop = np.where(height_beta < ctop)[0][-1]       #index of highest lidar gate below ctop
      indl_dbase = np.where(height_beta > dbase)[0][0]      #index of first lidar gate above drizzle base
      indl_200 = np.where(height_beta < cbase+200.)[0][-1]  #index of the highest lidar gate within 200m from the cloud base


      #-----construct TPQ profile-----
      indices = list(np.where(tpqheight<=np.min([dbase,cbase]))[0])  
      height_blwcb = tpqheight[indices]                        
      temp_blwcb = tempdata[indices]  
      pres_blwcb = presdata[indices]  
      hum_blwcb =  humdata[indices]   
      n_blwcb = height_blwcb.size
      
      indices = list(np.where(tpqheight>=ctop)[0])      
      height_abvct = tpqheight[indices]    
      temp_abvct = tempdata[indices]  
      pres_abvct = presdata[indices]  
      hum_abvct =  humdata[indices]   
      
      #interpolate TPQ for within the cloud/drizzle
      ind_beta_incld = list(np.where((height_beta>np.min([dbase,cbase])) & (height_beta<=height_z[-1]))[0])  
      height_incld = height_beta[ind_beta_incld]
      temp_incld = np.interp(height_incld,tpqheight,tempdata)
      pres_incld = np.interp(height_incld,tpqheight,presdata)
      hum_incld = np.interp(height_incld,tpqheight,humdata)

      #stitch TPQ profiles together (below+within+above up to 1km+between 1 and 30km)
      height30 = np.append(np.append(height_blwcb,height_incld),height_abvct)
      pres30 = np.append(np.append(pres_blwcb,pres_incld),pres_abvct)
      temp30 = np.append(np.append(temp_blwcb,temp_incld),temp_abvct)
      hum30 = np.append(np.append(hum_blwcb,hum_incld),hum_abvct)
      n30 = height30.size

      presdata_extair = np.interp(height_beta,tpqheight,presdata) 
      tempdata_extair = np.interp(height_beta,tpqheight,tempdata) 
      ext_air = ext_coeff_air(wav=wav,pressure=presdata_extair,temperature=tempdata_extair)

      #compute the optical depth due to gas
      tau_gas = calc_tau_gas(height30,temp30,pres30,hum30,freq_brt,gas_abs)
      tau_gas_total = tau_gas[0]
      
      theta = theta*np.pi/180.0                                  #convert to radians      
      mu = np.cos(theta) + 0.025*np.exp(-11.*np.cos(theta))      #Rozenberg approximation for 1/airmass

      cov_msr = cov_mat_msr(tb_err,beta_err[0:indl_200+1],Z_err,beta_mean[0:indl_200+1],Z_mean)
      cov_msr_inv = invert_lu(cov_msr)                  

      ntb = tb_mean.size
      nbeta = indl_200+1
      nrref = Z_mean.size  

      #build the measurement vector
      yy = np.zeros(ntb+nbeta+nrref)                    
      yy[0:ntb] = tb_mean
      yy[ntb:ntb+nbeta] = beta_mean[0:indl_200+1]
      yy[ntb+nbeta:ntb+nbeta+nrref] = Z_mean


      #run the optimization
      cff = open(cffile,'w')
      cff.write('# Nfeval     total_cost_value\n')

      Nfeval=0
      Niter = 0
      flag_nan = 0
        
      cost_value = 1.e6     #random large number      
      seed_de = 8

      progf = open(progfile,'w')
      progf.write('#along track count = '+str(atnumber)+'\n')

      #now split into 2 categories
      if (height_z[0] >= cbase):     #this inversion script is only for profiles WITHOUT drizzle signal below cloud base  ==> cost2
        print " ******** There is no radar signal below cloud base; count ", count

        bounds2 = []
        for iix in range(lb2.size): bounds2 = bounds2+[tuple([lb2[iix],ub2[iix]])]
        
        progf.write('#seed(DE): '+str(seed_de)+'; maxiter(DE): '+str(max_it)+'\n')
        progf.write('#recombination(DE): '+str(recomb)+'; popsize(DE): '+str(popsizef)+'\n')
        progf.write('#mutation(DE): '+str(mutationf)+'; tolerance(DE): '+str(tolerance)+'\n')
        progf.write('#DE upper bound: '+str(ub2[0:4])+'\n')
        progf.write('#                '+str(ub2[4:8])+'\n')
        progf.write('#                '+str(ub2[8:])+'\n')
        progf.write('#DE lower bound: '+str(lb2[0:4])+'\n')
        progf.write('#                '+str(lb2[4:8])+'\n')
        progf.write('#                '+str(lb2[8:])+'\n')
        progf.write('#mixing model: '+str(mmod)+'\n')
        progf.write('#Radar grid resolution [m]: '+str(gridres)+'\n')
        progf.write('#Absorption model [liquid,gas]: ['+liq_abs+','+gas_abs+']\n')
        progf.write('#Zenith angle (degrees): '+str(theta)+'\n')
        progf.write('iter  frac.conv.  x[0]  x[1]  x[2]  x[3]  x[4]  x[5]  x[6]  x[7]  x[8]  x[9]  x[10]  x[11]  x[12]\n')
      
        c0 = clock()
        res = differential_evolution(cost2, bounds2, seed=seed_de, tol=tolerance, maxiter=max_it, polish=1,init='latinhypercube',callback=callback_func,mutation=mutationf,popsize=popsizef, recombination=recomb)
        c1 = clock()
        cff.close()

        xres=res.x

        retf = open(retfile,'w')
        retf.write('#time[hrs] = '+str(along_track/3600.)+'\n')   #time in hrs
        retf.write('#-----cloud parameters-----\n')       
        retf.write('#boundaries   = '+str(cbase)+' , '+str(zcb)+' ; '+str(zct)+' [m]\n')
        retf.write('#nu           = '+str(xres[0])+'\n')
        retf.write('#hhat         = '+str(xres[1])+'\n')
        retf.write('#alpha        = '+str(xres[2])+'\n')
        retf.write('#Nad          = '+str(xres[3])+' [1/m3]\n')
        retf.write('#rcal         = '+str(xres[4])+'\n')
        retf.write('#lcal         = '+str(clid)+'  ;  ext_dbase: '+str(10.**xres[5])+'  ;  '+str(xres[5])+'\n')
        retf.write('#cldtop       = '+str(ctop+(xres[6]*gridres))+'  ;  '+str(xres[6])+'\n')
        retf.write('#cbasemod     = '+str(cbase_mod_lo+(xres[7]*(cbase_mod_hi-cbase_mod_lo)))+'  ;  '+str(xres[7])+'\n')      
        retf.write('#weight_cbase = '+str(xres[12])+'\n')
        retf.write('#-----drizzle parameters-----\n')       
        retf.write('#boundaries   = '+str(dbase)+' ; '+str(dtop)+' [m]\n')   
        retf.write('#nu           = '+str(xres[8])+'\n')
        retf.write('#hhat         = '+str(xres[9])+'\n')
        retf.write('#alpha        = '+str(xres[10])+'\n')
        retf.write('#lwc_scale    = '+str(10.**xres[11])+'  ;  '+str(xres[11])+'\n')
        retf.write('#===============================================================\n')
              
        
      else: # (height_z[0] < cbase)  ==>     cost1
        indr_cbase1 = np.where(height_z < cbase)[0][-1]  #index of the highest radar gate located below cbase  
        if (indr_cbase2-indr_cbase1 == 1):
          Z_cbase = np.interp(cbase, [height_z[indr_cbase1],height_z[indr_cbase2]], [Z_mean[indr_cbase1],Z_mean[indr_cbase2]]) #Z at cbase: interpolation
        elif (indr_cbase2-indr_cbase1 == 2):
          Z_cbase = Z_mean[indr_cbase2-1]

        bounds1 = []
        for iix in range(lb1.size): bounds1 = bounds1+[tuple([lb1[iix],ub1[iix]])]
        
        progf.write('#seed(DE): '+str(seed_de)+'; maxiter(DE): '+str(max_it)+'\n')
        progf.write('#recombination(DE): '+str(recomb)+'; popsize(DE): '+str(popsizef)+'\n')
        progf.write('#mutation(DE): '+str(mutationf)+'; tolerance(DE): '+str(tolerance)+'\n')
        progf.write('#DE upper bound: '+str(ub1[0:4])+'\n')
        progf.write('#                '+str(ub1[4:8])+'\n')
        progf.write('#                '+str(ub1[8:])+'\n')
        progf.write('#DE lower bound: '+str(lb1[0:4])+'\n')
        progf.write('#                '+str(lb1[4:8])+'\n')
        progf.write('#                '+str(lb1[8:])+'\n')
        progf.write('#mixing model: '+str(mmod)+'\n')
        progf.write('#Radar grid resolution [m]: '+str(gridres)+'\n')
        progf.write('#Absorption model [liquid,gas]: ['+liq_abs+','+gas_abs+']\n')
        progf.write('#Zenith angle (degrees): '+str(theta)+'\n')
        progf.write('iter  frac.conv.  x[0]  x[1]  x[2]  x[3]  x[4]  x[5]  x[6]  x[7]  x[8]  x[9]  x[10]  x[11]  x[12]\n')
     
        c0 = clock()
        res = differential_evolution(cost1, bounds1, seed=seed_de, tol=tolerance, maxiter=max_it, polish=1,init='latinhypercube',callback=callback_func,mutation=mutationf,popsize=popsizef, recombination=recomb)
        c1 = clock()

        if (flag_nan == 1):   #due to the early return because of nan values in the state vector; repeat the optimization with a different random seed (some variables need resetting)
          Nfeval = 0
          Niter = 0
          cost_value = 1.e6
          seed_de = 10
          flag_nan = 0    #reset
          print '-----------RESTART-------------'
          progf.write('\n-----------random seed : '+str(seed_de)+'--------------\n')
          progf.write('iter  frac.conv.  x[0]  x[1]  x[2]  x[3]  x[4]  x[5]  x[6]  x[7]  x[8]  x[9]  x[10]  x[11]\n')
          cff.write('\n-----------restart; random seed: '+str(seed_de)+'--------------\n')
          cff.write('# Nfeval    total_cost_value\n')
          c0 = clock()
          res = differential_evolution(cost1, bounds1, seed=seed_de, tol=tolerance, maxiter=max_it, polish=1,init='latinhypercube',callback=callback_func,mutation=mutationf,popsize=popsizef, recombination=recomb)
          
          c1 = clock()

          if (flag_nan==10): 
            sys.stderr.write("--------- ERROR: The inversion failed; count "+str(count)+" ----------\n")
            cff.close()
            count+=1
            continue            

        cff.close()  

        xres=res.x
  
        retf = open(retfile,'w')
        retf.write('#time[hrs] = '+str(along_track/3600.)+'\n')   #time in hrs
        retf.write('#-----cloud parameters-----\n')       
        retf.write('#boundaries   = '+str(cbase)+' , '+str(zcb)+' ; '+str(zct)+' [m]\n')
        retf.write('#nu           = '+str(xres[0])+'\n')
        retf.write('#hhat         = '+str(xres[1])+'\n')
        retf.write('#alpha        = '+str(xres[2])+'\n')
        retf.write('#Nad          = '+str(xres[3])+' [1/m3]\n')
        retf.write('#rcal         = '+str(xres[4])+'\n')
        retf.write('#lcal         = '+str(clid)+'  ;  ext_dbase: '+str(10.**xres[5])+'  ;  '+str(xres[5])+'\n')
        retf.write('#cldtop       = '+str(ctop+(xres[6]*gridres))+'  ;  '+str(xres[6])+'\n')
        retf.write('#cbasemod     = '+str(cbase_mod_lo+(xres[7]*(cbase_mod_hi-cbase_mod_lo)))+'  ;  '+str(xres[7])+'\n')      
        retf.write('#weight_cbase = '+str(xres[12])+'\n')
        retf.write('#-----drizzle parameters-----\n')       
        retf.write('#boundaries   = '+str(dbase)+' ; '+str(dtop)+' [m]\n')   
        retf.write('#nu           = '+str(xres[8])+'\n')
        retf.write('#ext_dzl_blw  = '+str((10.**xres[9])*(10.**xres[10]))+'  ;  '+str(xres[9])+'\n')
        retf.write('#ext_dzl_cbase= '+str(10.**xres[10])+'  ;  '+str(xres[10])+'\n')
        retf.write('#ext_dzl_peak = '+str((10.**xres[11])*np.max(ext_cld_common))+'  ;  '+str(xres[11])+'\n')
        retf.write('#===============================================================\n')

 
      #-------------Printing out retrieved profiles--------------
      retf.write('#Col 1: Height [m]\n')
      retf.write('#Col 2: LWC [g/m3]\n')
      retf.write('#Col 3: r_e [m]\n')
      retf.write('#Col 4: extcoeff [1/m]\n')
      retf.write('#Col 5: N [1/m3]\n')
      retf.write('#========================= cloud ===============================\n')
      for i in range(z_common.size):
         retf.write(str(z_common[i])+'  '+str(lwc_cld_common[i])+'  '+str(re_cld_common[i])+'  '+str(ext_cld_common[i])+'  '+str(Nz_cld_common[i])+'\n')  
      retf.write('#==================================================================\n')
      retf.write('88.  88.  88.  88.  88\n')         
      retf.write('#========================= drizzle ================================\n')
      for i in range(z_common.size):
        retf.write(str(z_common[i])+'  '+str(lwc_dzl_common[i])+'  '+str(re_dzl_common[i])+'  '+str(ext_dzl_common[i])+'  '+str(Nz_dzl_common[i])+'\n')
      retf.write('#==================================================================\n')
      retf.write('#'+res.message+'\n')
      retf.write('#Current function value: '+str(res.fun)+'\n')
      retf.write('#Iterations: '+str(res.nit)+'\n')
      retf.write('#Function evaluations: '+str(res.nfev)+'\n')
      retf.write('#Elapsed CPU time: '+str((c1-c0)/60.)+' minutes on '+system()+'\n')
      retf.close()    

      #-------------Printing out signals--------------
      sgnf = open(sgnfile,'w')
      sgnf.write ('#This logs the signal at the last iteration\n')
      sgnf.write ('#Height[m]     Z_fm_cld[mm6/m3]     Z_fm_dzl[mm6/m3]      Z_data[mm6/m3]      Z_err[mm6/m3]      Z_fm_tot_att[mm6/m3]\n')
      for i in range(nrref):
        sgnf.write(str(height_z[i])+'  '+str(Fradref[i])+'  '+str(Fradref_dzl[i])+'  '+str(Z_mean[i])+'  '+str(Z_err[i])+'  '+str(Fradref_att[i])+'\n')
      sgnf.write('#=======================================\n')
      sgnf.write ('#Frequency[GHz]   T_B_fm[K]     placeholder     T_B_data[K]     T_B_err[K]     placeholder\n')
      for i in range(ntb):
        sgnf.write(str(freq_brt[i])+'  '+str(Ftb[i])+'  '+str(0.0)+'  '+str(tb_mean[i])+'  '+str(tb_err[i])+'  '+str(0.0)+'\n')
      sgnf.write('#=======================================\n')
      sgnf.write ('#Height[m]     beta_fm[1/m/sr]      placeholder        beta_data[1/m/sr]       beta_err[1/m/sr]      placeholder\n')
      for i in range(height_beta.size):
        sgnf.write(str(height_beta[i])+'  '+str(Fbeta[i])+'  '+str(0.0)+'  '+str(beta_mean[i])+'  '+str(beta_err[i])+'  '+str(0.0)+'\n')
      sgnf.close()

      count += 1  
      
    except (EOFError): 
      print "End of File"
      infile.close()
      tpqfile.close()
