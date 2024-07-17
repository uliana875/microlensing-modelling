import MulensModel as mm
import os
import numpy as np
import pandas as pd
import time
import scipy.optimize as op
from fitting import chi2_fun
from scipy.stats import gaussian_kde
import arviz as az


def create_event(t_0, u_0, t_E, datasets):
    params = dict()
    params['t_0'] = t_0
    params['u_0'] = u_0
    params['t_E'] = t_E
    my_model = mm.Model(params)
    my_event = mm.Event(datasets=(datasets), model=my_model)
    return my_event

def create_event_parallax(parameters, datasets, coords,t0_par=None, fix_blends=None):
    t_0, u_0, t_E, pi_E_N, pi_E_E = parameters
    params = dict()
    if t0_par is None: params['t_0_par'] = t_0
    else: params['t_0_par'] = t0_par
    params['t_0'] = t_0
    params['u_0'] = u_0
    params['t_E'] = t_E
    params['pi_E_N'] = pi_E_N
    params['pi_E_E'] = pi_E_E
    my_model = mm.Model(params, coords=coords)
    my_event = mm.Event(datasets=(datasets), model=my_model, fix_blend_flux=fix_blends)
    return my_event


def create_event_parallax_FS(parameters, datasets, coords, 
                             magnification_methods, t0_par=None, fix_blends=None):
    t_0, u_0, t_E, pi_E_N, pi_E_E, rho = parameters
    params = dict()
    if t0_par is None: params['t_0_par'] = t_0
    else: params['t_0_par'] = t0_par
    params['t_0'] = t_0
    params['u_0'] = u_0
    params['t_E'] = t_E
    params['pi_E_N'] = pi_E_N
    params['pi_E_E'] = pi_E_E
    params['rho'] = rho
    my_model = mm.Model(params, coords=coords)
    my_model.set_magnification_methods(magnification_methods)
    my_event = mm.Event(datasets=(datasets), model=my_model, fix_blend_flux=fix_blends)
    return my_event



def get_params(my_event):
    params_dict = my_event.model.parameters.__dict__['parameters']
    params = dict()
    for key, value in params_dict.items():
        if key == 't_E':
            params[key] = params_dict[key].value
        else:
            params[key] = params_dict[key]
    return params

def beep(duration, freq):
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
    
def get_samples(sampler, n_burn, my_event):
    keys = list(my_event.model.parameters.__dict__['parameters'].keys())
    n_dim = len(keys)
    if 't_0_par' in keys: n_dim -= 1
    samples = sampler.chain[:, n_burn:, :]
    blobs = np.array(sampler.blobs)
    blobs = np.transpose(blobs, axes=(1, 0, 2))[:, n_burn:, :]
    samples_with_blobs = np.dstack((samples, blobs))
    samples = samples_with_blobs.reshape((-1,n_dim+2*len(my_event.datasets)))
 
    return samples
    

def get_datacut(samples, my_event, rho_lims=None, u0_lims=None, t0_lims=None, tE_lims=None, piEN_lims=None, piEE_lims=None):
    params_names = list(my_event.model.parameters.__dict__['parameters'].keys())
    if 't_0_par' in params_names: params_names.remove('t_0_par')
    flux_names = []
    n = len(params_names)
    size_ax1 = np.shape(samples)[1]
    for i in range(n,size_ax1):
        #filter_name = my_event.datasets[i-size].plot_properties['label']
        flux_names += [str(i)]
        
    col_names = params_names + flux_names
    
    samples_params = []
    size_ax0 = len(samples[:,0])
    for i in range(size_ax1):
        samples_params.append(samples[:,i].reshape(size_ax0,1))
    datacut0 = np.hstack(samples_params)
    df_datacut = pd.DataFrame(datacut0, columns=col_names)
    
    if rho_lims is not None:
        df_datacut = df_datacut.drop(df_datacut[(df_datacut['rho']<=rho_lims[0]) | (df_datacut['rho']>=rho_lims[1])].index, axis=0)
           
    if u0_lims is not None:
        df_datacut = df_datacut.drop(df_datacut[(df_datacut['u_0']<=u0_lims[0]) | (df_datacut['u_0']>=u0_lims[1])].index, axis=0)
        
    if t0_lims is not None:
        df_datacut = df_datacut.drop(df_datacut[(df_datacut['t_0']<=t0_lims[0]) | (df_datacut['t_0']>=t0_lims[1])].index, axis=0)
        
    if tE_lims is not None:
        df_datacut = df_datacut.drop(df_datacut[(df_datacut['t_E']<=tE_lims[0]) | (df_datacut['t_E']>=tE_lims[1])].index, axis=0)
        
    if piEN_lims is not None:
        df_datacut = df_datacut.drop(df_datacut[(df_datacut['pi_E_N']<=piEN_lims[0]) | (df_datacut['pi_E_N']>=piEN_lims[1])].index, axis=0)
        
    if piEE_lims is not None:
        df_datacut = df_datacut.drop(df_datacut[(df_datacut['pi_E_E']<=piEE_lims[0]) | (df_datacut['pi_E_E']>=piEE_lims[1])].index, axis=0)
    
        
    for i in range(n,size_ax1,2):
        fs = df_datacut[str(i)]
        fb = df_datacut[str(i+1)]
        F = fs+fb
        if len(F[F<0])==0:
            df_datacut[str(i)] = 22-2.5*np.log10(np.array(F, dtype=float))
        else:
            df_datacut[str(i)] = np.random.normal(size=len(F))
            
        df_datacut[str(i+1)] = fs/F
    
    datacut = df_datacut[params_names].to_numpy(dtype=float)
    datacut_flux = df_datacut[flux_names].to_numpy(dtype=float)
    
    # if rho_lims is not None:
    #     inds = np.where((datacut[:,5]>rho_lims[0]) & (datacut[:,5]<rho_lims[1]))[0]
    #     datacut = datacut[inds,:]
    #     datacut_flux = datacut_flux[inds,:]
            
           
    # if u0_lims is not None:
    #     inds = np.where((datacut[:,1]>u0_lims[0]) & (datacut[:,1]<u0_lims[1]))[0]
    #     datacut = datacut[inds,:]
    #     datacut_flux = datacut_flux[inds,:]
    
    return datacut, datacut_flux
    


def datacut_for_DLC(datacut,datacut_flux,gaia_index):
    # gaia index is the index in datacut_flux
    return np.hstack((datacut[:,:5], datacut_flux[:,gaia_index:gaia_index+2]))

def extract_params(data_cut):
    mc_result = []
    
    # parameters for mode solution
    for i in range(0, len(data_cut[0,:])):
        sol = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(data_cut, [16, 50, 84], axis=0))))[i]
        mc_result.append(np.array([sol[0],sol[1],sol[2]]))
        
    return np.array(mc_result)


def chi2_loop(initial_params,delta_t, rhos, datasets, coords, magnification_method='finite_source_uniform_Gould94_direct', fix_blends=None, do_beep=True, printting=True):
    params = dict()
    params['t_0_par'] = initial_params[0]
    params['t_0'], params['u_0'], params['t_E'], params['pi_E_N'], params['pi_E_E'] = initial_params
    maglim1, maglim2 = params['t_0']-delta_t, params['t_0']+delta_t
    magnification_methods = [maglim1, magnification_method, maglim2] 
    
    chi2s = []
    u0s = []
    piees, piens = [],[]
    
    print('Executing the loop...')
    
    start_t = time.time()
    for rho in rhos:
        params['rho'] = rho
        if printting: print(params['rho'])
        my_model = mm.Model(params,coords=coords)
        
        my_model.set_magnification_methods(magnification_methods)
        my_event = mm.Event(datasets=(datasets), model=my_model, fix_blend_flux=fix_blends)

        parameters_to_fit = ["t_0", "u_0", "t_E", "pi_E_N", "pi_E_E"]

        result = op.minimize(chi2_fun, x0=initial_params, args=(parameters_to_fit, my_event))
        
        chi2 = chi2_fun(result.x, parameters_to_fit, my_event)
        u0s.append(my_event.model.parameters.u_0)
        piees.append(my_event.model.parameters.pi_E_E)
        piens.append(my_event.model.parameters.pi_E_N)
        chi2s.append(chi2)
    
    print('Done')
    ex_time = time.time()-start_t
    print(str(round(ex_time/60,2))+' min')
    
    if do_beep: beep(1, 620)
    
    return np.array(chi2s), np.array(u0s), np.array(piens), np.array(piees)
    


def get_mode_value(data):
    kde = gaussian_kde(data)

    # Generate a range of values to evaluate the PDF
    x_vals = np.linspace(min(data), max(data), 500)

    # Evaluate the PDF at each point
    pdf_values = kde(x_vals)

    # Find the mode (peak) of the PDF
    mode_index = np.argmax(pdf_values)
    param_value = x_vals[mode_index]
    
    min_param, max_param = az.hdi(data, hdi_prob=.68)
    
    return np.array([param_value, max_param-param_value, param_value-min_param])


def print_flux_values(datacut_flux, df=None, filter_names=None, do_round=True):
    if filter_names is None:
    	names=df['Filter'].unique()
    else:
    	names=filter_names
    for i in range(len(datacut_flux[1,:])):
        if i%2==0: 
            print(names[int(i//2)])
            
        m = np.round(np.median(datacut_flux[:,i]), 4)
        
        if (i%2==1) and (m>=1.) and (m<1.1):
            print(m,' !')
        elif (i%2==1) and (m>=1.1):
            print(m,' !!')
        else: print(m)
        
        if (i%2==1):
            print('\n')
