import MulensModel as mm
import matplotlib.pyplot as plt
import numpy as np
import corner
from main import extract_params, get_mode_value
import matplotlib
from fitting import quick_fit
from scipy.stats import gaussian_kde
from matplotlib import container
import sys
sys.path.insert(0, '/home/dell/microlensing/phot_modelling')
from fitting import quick_fit_parallax

def final_plot_FS_with_peak2(datacut, coords, datasets,magnification_methods, xlims_peak, ylims_peak, t0_par,  order_markers,
                             xlims=None, ylims_res=None, u0_hdi=False, rho=None,plot_grid=True,axes_list=[0.18, 0.6, 0.28, 0.25], 
                             gaia_filter=0, in_guess=None, fix_blends=None, pspl_color='indigo'):
    
    best = extract_params(datacut)[:,0]
    
    if u0_hdi:
        kde = gaussian_kde(datacut[:,1])

        # Generate a range of values to evaluate the PDF
        x_vals = np.linspace(min(datacut[:,1]), max(datacut[:,1]), 500)

        # Evaluate the PDF at each point
        pdf_values = kde(x_vals)

        # Find the mode (peak) of the PDF
        mode_index = np.argmax(pdf_values)
        param_value = x_vals[mode_index]
        
        best[1] = param_value
        
    if rho is not None:
        best[5] = rho
    
    params1 = dict()
    params1['t_0_par'] = t0_par
    params1['t_0'], params1['u_0'], params1['t_E'], params1['pi_E_N'], params1['pi_E_E'], params1['rho'] = best
    my_model_final = mm.Model(params1, coords = coords)
    
    my_model_final.set_magnification_methods(magnification_methods)
    my_event_final = mm.Event(datasets=(datasets), model=my_model_final, fix_blend_flux=fix_blends) 
    
    ###
    
    my_model = mm.Model({'t_0': best[0],'t_0_par': best[0], 'u_0': best[1], 't_E': best[2], 
                         'pi_E_N': best[3], 'pi_E_E': best[4]}, coords = coords)
    my_event = mm.Event(datasets=(datasets), model=my_model, fix_blend_flux=fix_blends) 
    quick_fit_parallax(my_event, in_guess=in_guess)
    
    if xlims is None: xlim1, xlim2 = default_limits(my_event_final)
    else: xlim1, xlim2 = np.array(xlims)
    
    
    fig = plt.figure(figsize=(6,3.6))

    
    ''' Main plot '''
  
    axes = fig.add_axes([0., 0.7, 0.92, 1.])
    my_event_final.plot_data(subtract_2450000=True,capsize=2)
    my_event_final.plot_model(color='black', t_start=xlim1+2450000., t_stop=xlim2+2450000., lw=2, subtract_2450000=True, label="FSPL", dt=0.1, zorder=70)
    my_event.plot_model(color=pspl_color, ls='--', t_start=xlim1+2450000., t_stop=xlim2+2450000.,lw=2, subtract_2450000=True, label="PSPL", zorder=60)
    
    plt.xlim(xlim1,xlim2)
    handles, labels = axes.get_legend_handles_labels()
    order_lines = [0,1]
    order_markers= order_markers
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    legend_lines = axes.legend([handles[idx] for idx in order_lines],[labels[idx] for idx in order_lines], 
                loc='upper left',frameon=False,fontsize=11)
    legend_markers = axes.legend([handles[idx] for idx in order_markers],[labels[idx] for idx in order_markers], 
                loc='upper right',frameon=False,fontsize=11)
    plt.gca().add_artist(legend_lines)
    #plt.legend(loc='upper right')
    
    if plot_grid:
        plt.grid()
        
    plt.xlabel(None)
    plt.xticks(color='w', fontsize=1)
    #plt.subplots_adjust(hspace=0.1)
    
    
    ''' Residuals '''
    axes = fig.add_axes([0., 0.18, 0.92, 0.45])
    #axes = ax[0,1
    my_event_final.plot_residuals(subtract_2450000=True,capsize=2, capthick=1)
    
    
    (source_flux1, blend_flux1) = my_event.get_flux_for_dataset(gaia_filter)
    (source_flux2, blend_flux2) = my_event_final.get_flux_for_dataset(gaia_filter)
    xmodel = np.linspace(xlim1+2450000, xlim2+2450000, num=400)
    ymodel1=my_model.get_lc(xmodel, source_flux=source_flux1[0], blend_flux=blend_flux1)
    ymodel2=my_event_final.model.get_lc(xmodel, source_flux=source_flux2[0], blend_flux=blend_flux2)
    difmodel = ymodel2-ymodel1
    plt.plot(xmodel-2450000, difmodel, ls='--',color='indigo',zorder=10, lw=2)
    
    plt.xlim(xlim1,xlim2)
    plt.xlabel(None)
       
    if ylims_res is None: ylim1_res, ylim2_res = -0.08,0.08
    else: ylim1_res, ylim2_res = np.array(ylims_res)
    
    plt.ylim(ylim1_res, ylim2_res)
    
    if plot_grid:
        plt.grid(zorder=0)
        
    
    ''' Small plot '''
    axes = fig.add_axes([1.0, 0.7, 0.64, 1.])
    #axes = ax[1,0]
    if plot_grid:
        plt.grid()
    
    my_event_final.plot_data(subtract_2450000=True,capsize=2., markersize=8)
    my_event_final.plot_model(color='black', t_start=xlim1+2450000., t_stop=xlim2+2450000., dt=0.02, lw=2, subtract_2450000=True, zorder=70)    
    my_event.plot_model(color='indigo', ls='--', t_start=xlim1+2450000., t_stop=xlim2+2450000., dt=0.02, subtract_2450000=True, zorder=60,lw=2)
    
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    
    
    plt.xlim(xlims_peak[0], xlims_peak[1])
    plt.ylim(ylims_peak[0], ylims_peak[1])
    
    plt.xlabel(None)
    plt.ylabel(None)
    plt.xticks(color='w', fontsize=1)
   
        
    
    ''' Residuals for the small plot '''
    axes = fig.add_axes([1.0, 0.18, 0.64, 0.45])
    #axes = ax[1,1]
    my_event_final.plot_residuals(subtract_2450000=True,capsize=2, capthick=1)
    plt.plot(xmodel-2450000, difmodel, ls='--',color='indigo',zorder=10, lw=2)
    plt.xlim(xlims_peak[0], xlims_peak[1])
    plt.ylim(ylim1_res, ylim2_res)
    #plt.show()
    plt.xlabel(None)
    plt.ylabel(None)
    plt.yticks(color='w', fontsize=1)
    
    #fig.set_xlabel('t')
    
    fig.text(0.9, 0.04, 'Time - 2450000', ha='center')
    
    return my_event, my_event_final


def default_limits(my_event):
    t_0 = my_event.model.parameters.t_0
    t_E = my_event.model.parameters.t_E
    xlim1 = t_0 - 2450000 - 3*t_E
    xlim2 = t_0 - 2450000 + 3*t_E
    return xlim1,xlim2

def plot_event(my_event, xlims=None, ylims=None, grid=True, dt=None):
    
    if xlims==None:
        xlim1, xlim2 = default_limits(my_event)
    else:
        xlim1, xlim2 = xlims  
    
    
              
    fig, (ax1) = plt.subplots(1, 1)
    my_event.plot_data(subtract_2450000=True)#, axis=ax1)
    my_event.plot_model(subtract_2450000=True, t_start=xlim1+2450000, t_stop=xlim2+2450000, dt=dt)#, axis=ax1)
    
    handles, labels = ax1.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    ax1.legend(handles, labels)
    plt.grid(grid)
    
    plt.xlim(xlim1,xlim2)
    if ylims!=None:
    	  ylim1, ylim2 = ylims  
    	  plt.ylim(ylim1,ylim2)
    plt.show()
    
    
def chi2_fun(theta, parameters_to_fit, event):
    for (parameter, value) in zip(parameters_to_fit, theta):
        setattr(event.model.parameters, parameter, value)
    return event.get_chi2()

    
def corner_plot(datacut, flux_labels=None):#, log_rho=False, log_u0=False, rho_lims=None, u0_lims=None):
    
    size = np.shape(datacut)[1]

    if flux_labels is None:
        if size==5:
            labels=["$t_0$", "$u_0$", "$t_E$", '$\pi_{EN}$','$\pi_{EE}$']
        elif size==6:
            labels=["$t_0$", "$u_0$", "$t_E$", '$\pi_{EN}$','$\pi_{EE}$', r'$\rho$']
        else:
            raise ValueError('Wrong size of a datacut')
    else:
        labels = flux_labels
               
    
    corner.corner(datacut, bins=60, color='k', fontsize=15, show_titles=True, verbos=True, title_fmt='.4f',
                        labels=labels,
                               quantiles=[0.16,0.5,0.84],
                        plot_datapoints=True, levels=(1-np.exp(-0.5), 1-np.exp(-2.), 1-np.exp(-9./2.)),
                        fill_contours=True)
    

def get_flux_labels(df):
    labels_flux = []
    for f in df['Filter'].unique():
        labels_flux += ['$G_{'+f+'}$', '$f_{s,'+f+'}$']
    return labels_flux

def basic_corner(datacut):
    corner.corner(datacut, bins=60, color='k', fontsize=15, show_titles=True, verbos=True, title_fmt='.4f',
                               quantiles=[0.16,0.5,0.84],
                        plot_datapoints=True, levels=(1-np.exp(-0.5), 1-np.exp(-2.), 1-np.exp(-9./2.)),
                        fill_contours=True)
    plt.show()
    
    
def final_plot(datacut, coords, datasets, xlims=None, ylims_res=None, u0_hdi=False, rho=None, grid=True, gaia_filter=0):
        
    best = extract_params(datacut)[:,0]
    
    if u0_hdi:
        kde = gaussian_kde(datacut[:,1])

        # Generate a range of values to evaluate the PDF
        x_vals = np.linspace(min(datacut[:,1]), max(datacut[:,1]), 500)

        # Evaluate the PDF at each point
        pdf_values = kde(x_vals)

        # Find the mode (peak) of the PDF
        mode_index = np.argmax(pdf_values)
        param_value = x_vals[mode_index]
        
        best[1] = param_value
        
    
    params = dict()

    params['t_0'], params['u_0'], params['t_E'] = best

    my_model_final = mm.Model(params, coords = coords)
    my_event_final = mm.Event(datasets=(datasets), model=my_model_final) 
    
    my_model = mm.Model({'t_0': best[0], 'u_0': best[1], 't_E': best[2]}, coords = coords)
    my_event = mm.Event(datasets=(datasets), model=my_model) 
    quick_fit(my_event)
    
    if xlims is None: xlim1, xlim2 = default_limits(my_event_final)
    else: xlim1, xlim2 = np.array(xlims)
        
    plt.figure(figsize=(10, 6))
    grid = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axes = plt.subplot(grid[0])
    my_event_final.plot_data(subtract_2450000=True,capsize=2)
    my_event_final.plot_model(color='black', t_start=xlim1+2450000., t_stop=xlim2+2450000., lw=2, subtract_2450000=True, label="$PSPL+\pi$", dt=0.02)
    my_event.plot_model(color='magenta', ls='--', t_start=xlim1+2450000., t_stop=xlim2+2450000., subtract_2450000=True, label="PSPL",dt=0.02)
    plt.xlabel(None)
    plt.grid()


    plt.xlim(xlim1,xlim2)
    #plt.ylim(14,10)

    plt.legend(loc='upper left')

    axes = plt.subplot(grid[1])
    my_event_final.plot_residuals(subtract_2450000=True,alpha=1,zorder=5,capsize=2)

    (source_flux1, blend_flux1) = my_event.get_ref_fluxes(gaia_filter)
    (source_flux2, blend_flux2) = my_event_final.get_ref_fluxes(gaia_filter)
    xmodel = np.linspace(xlim1+2450000, xlim2+2450000, num=3000)
    ymodel1=my_model.get_lc(xmodel, source_flux=source_flux1, blend_flux=blend_flux1)
    ymodel2=my_event_final.model.get_lc(xmodel, source_flux=source_flux2, blend_flux=blend_flux2)
    difmodel = ymodel2-ymodel1#
    plt.plot(xmodel-2450000, difmodel, ls='--',color='magenta',zorder=10)

    plt.xlim(xlim1, xlim2)
    
    if ylims_res is None: ylim1_res, ylim2_res = -0.08,0.08
    else: ylim1_res, ylim2_res = np.array(ylims_res)
    
    plt.ylim(ylim1_res, ylim2_res)
    plt.grid(zorder=0)

    plt.show()
    return my_event, my_event_final
    

def final_plot_parallax(datacut, coords, datasets, t0_par,xlims=None, ylims_res=None,u0_hdi=False, grid=True, gaia_filter=0):
    
    best = extract_params(datacut)[:,0]
    
    if u0_hdi:
        kde = gaussian_kde(datacut[:,1])

        # Generate a range of values to evaluate the PDF
        x_vals = np.linspace(min(datacut[:,1]), max(datacut[:,1]), 500)

        # Evaluate the PDF at each point
        pdf_values = kde(x_vals)

        # Find the mode (peak) of the PDF
        mode_index = np.argmax(pdf_values)
        param_value = x_vals[mode_index]
        
        best[1] = param_value
    
    params = dict()
    params['t_0_par'] = t0_par

    params['t_0'], params['u_0'], params['t_E'], params['pi_E_N'], params['pi_E_E'] = best

    my_model_final = mm.Model(params, coords = coords)
    my_event_final = mm.Event(datasets=(datasets), model=my_model_final) 
    
    my_model = mm.Model({'t_0': best[0], 'u_0': best[1], 't_E': best[2]}, coords = coords)
    my_event = mm.Event(datasets=(datasets), model=my_model) 
    quick_fit(my_event)
    
    if xlims is None: xlim1, xlim2 = default_limits(my_event_final)
    else: xlim1, xlim2 = np.array(xlims)
        
    plt.figure(figsize=(10, 6))
    grid = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axes = plt.subplot(grid[0])
    my_event_final.plot_data(subtract_2450000=True,capsize=2)
    my_event_final.plot_model(color='black', t_start=xlim1+2450000., t_stop=xlim2+2450000., lw=2, subtract_2450000=True, label="$PSPL+\pi$", dt=0.02)
    my_event.plot_model(color='magenta', ls='--', t_start=xlim1+2450000., t_stop=xlim2+2450000., subtract_2450000=True, label="PSPL",dt=0.02)
    plt.xlabel(None)
    plt.grid()


    plt.xlim(xlim1,xlim2)
    #plt.ylim(14,10)

    plt.legend(loc='upper left')

    axes = plt.subplot(grid[1])
    my_event_final.plot_residuals(subtract_2450000=True,alpha=1,zorder=5,capsize=2)


    (source_flux1, blend_flux1) = my_event.get_ref_fluxes(gaia_filter)
    (source_flux2, blend_flux2) = my_event_final.get_ref_fluxes(gaia_filter)
    xmodel = np.linspace(xlim1+2450000, xlim2+2450000, num=3000)
    ymodel1=my_model.get_lc(xmodel, source_flux=source_flux1, blend_flux=blend_flux1)
    ymodel2=my_event_final.model.get_lc(xmodel, source_flux=source_flux2, blend_flux=blend_flux2)
    difmodel = ymodel2-ymodel1#
    plt.plot(xmodel-2450000, difmodel, ls='--',color='magenta',zorder=10)

    plt.xlim(xlim1, xlim2)
    
    if ylims_res is None: ylim1_res, ylim2_res = -0.08,0.08
    else: ylim1_res, ylim2_res = np.array(ylims_res)
    plt.ylim(ylim1_res, ylim2_res)
    plt.grid(zorder=0)

    plt.show()
    
    return my_event, my_event_final
    
def final_plot_FS(datacut, coords, datasets,magnification_methods, t0_par, xlims=None, ylims_res=None, 
                  u0_hdi=False, rho=None,grid=True, gaia_filter='Gaia', in_guess=None, dt=0.1):
    
    best = extract_params(datacut)[:,0]
    
    if u0_hdi:
        kde = gaussian_kde(datacut[:,1])

        # Generate a range of values to evaluate the PDF
        x_vals = np.linspace(min(datacut[:,1]), max(datacut[:,1]), 500)

        # Evaluate the PDF at each point
        pdf_values = kde(x_vals)

        # Find the mode (peak) of the PDF
        mode_index = np.argmax(pdf_values)
        param_value = x_vals[mode_index]
        
        best[1] = param_value
        
    if rho is not None:
        best[5] = rho
    
    params1 = dict()
    params1['t_0_par'] = t0_par
    params1['t_0'], params1['u_0'], params1['t_E'], params1['pi_E_N'], params1['pi_E_E'], params1['rho'] = best
    my_model_final = mm.Model(params1, coords = coords)
    
    my_model_final.set_magnification_methods(magnification_methods)
    my_event_final = mm.Event(datasets=(datasets), model=my_model_final) 
    
    my_model = mm.Model({'t_0': best[0], 'u_0': best[1], 't_E': best[2]}, coords = coords)
    my_event = mm.Event(datasets=(datasets), model=my_model) 
    quick_fit(my_event, in_guess=in_guess)
    
    if xlims is None: xlim1, xlim2 = default_limits(my_event_final)
    else: xlim1, xlim2 = np.array(xlims)
    
    plt.figure(figsize=(10, 6))
    grid = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axes = plt.subplot(grid[0])
    my_event_final.plot_data(subtract_2450000=True,capsize=2)
    my_event_final.plot_model(color='black', t_start=xlim1+2450000., t_stop=xlim2+2450000., lw=2, subtract_2450000=True, label="$\pi+FS$", dt=dt)
    my_event.plot_model(color='magenta', ls='--', t_start=xlim1+2450000., t_stop=xlim2+2450000., subtract_2450000=True, label="PSPL")
    
    
        
    plt.xlim(xlim1,xlim2)
    plt.legend(loc='upper left')
    plt.grid()
    
    axes = plt.subplot(grid[1])
    my_event_final.plot_residuals(subtract_2450000=True,alpha=1,zorder=5,capsize=2)
    
    
    (source_flux1, blend_flux1) = my_event.get_ref_fluxes(gaia_filter)
    (source_flux2, blend_flux2) = my_event_final.get_ref_fluxes(gaia_filter)
    xmodel = np.linspace(xlim1+2450000, xlim2+2450000, num=200)
    ymodel1=my_model.get_lc(xmodel, source_flux=source_flux1, blend_flux=blend_flux1)
    ymodel2=my_event_final.model.get_lc(xmodel, source_flux=source_flux2, blend_flux=blend_flux2)
    difmodel = ymodel2-ymodel1
    plt.plot(xmodel-2450000, difmodel, ls='--',color='magenta',zorder=10)
    
    plt.xlim(xlim1,xlim2)
    
    if ylims_res is None: ylim1_res, ylim2_res = -0.08,0.08
    else: ylim1_res, ylim2_res = np.array(ylims_res)
    
    plt.ylim(ylim1_res, ylim2_res)
    plt.grid(zorder=0)
    
    plt.show()
    
    return my_event, my_event_final


def final_plot_FS_with_peak(datacut, coords, datasets,magnification_methods, t0_par, xlims_peak, ylims_peak, xlims=None, ylims_res=None, 
                            u0_hdi=False, rho=None,plot_grid=True,axes_list=[0.18, 0.6, 0.28, 0.25], gaia_filter=0, in_guess=None,
                            pspl_color='indigo'):
    
    best = extract_params(datacut)[:,0]
    
    if u0_hdi:
        kde = gaussian_kde(datacut[:,1])

        # Generate a range of values to evaluate the PDF
        x_vals = np.linspace(min(datacut[:,1]), max(datacut[:,1]), 500)

        # Evaluate the PDF at each point
        pdf_values = kde(x_vals)

        # Find the mode (peak) of the PDF
        mode_index = np.argmax(pdf_values)
        param_value = x_vals[mode_index]
        
        best[1] = param_value
        
    if rho is not None:
        best[5] = rho
    
    params1 = dict()
    params1['t_0_par'] = t0_par
    params1['t_0'], params1['u_0'], params1['t_E'], params1['pi_E_N'], params1['pi_E_E'], params1['rho'] = best
    my_model_final = mm.Model(params1, coords = coords)
    
    my_model_final.set_magnification_methods(magnification_methods)
    my_event_final = mm.Event(datasets=(datasets), model=my_model_final) 
    
    my_model = mm.Model({'t_0': best[0],'t_0_par': best[0], 'u_0': best[1], 't_E': best[2], 
                         'pi_E_N': best[3], 'pi_E_E': best[4]}, coords = coords)
    my_event = mm.Event(datasets=(datasets), model=my_model) 
    quick_fit_parallax(my_event, in_guess=in_guess)
    
    if xlims is None: xlim1, xlim2 = default_limits(my_event_final)
    else: xlim1, xlim2 = np.array(xlims)
    
    
    fig = plt.figure(figsize=(10,6))
    
    #plt.subplots_adjust(wspace=0)
    #ax1 = fig.axes[0]
    
    ''' Main plot '''
    grid = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    #grid.tight_layout(figure=fig,pad=0.1)
    axes = plt.subplot(grid[0])
    my_event_final.plot_data(subtract_2450000=True,capsize=2)
    my_event_final.plot_model(color='black', t_start=xlim1+2450000., t_stop=xlim2+2450000., lw=2, subtract_2450000=True, label="FSPL", dt=0.1, zorder=70)
    my_event.plot_model(color=pspl_color, ls='--', t_start=xlim1+2450000., t_stop=xlim2+2450000.,lw=2, subtract_2450000=True, label="PSPL", zorder=60)
    
    plt.xlim(xlim1,xlim2)
    handles, labels = axes.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    axes.legend(handles, labels, loc='upper right',frameon=False)
    #plt.legend(loc='upper right')
    
    if plot_grid:
        plt.grid()
        
    plt.xlabel(None)
    plt.xticks(color='w', fontsize=1)
    plt.subplots_adjust(hspace=0.1)
    
    ''' Small plot '''
    axes = fig.add_axes(axes_list)
    if plot_grid:
        plt.grid()
    
    my_event_final.plot_data(subtract_2450000=True,capsize=2.)
    my_event_final.plot_model(color='black', t_start=xlim1+2450000., t_stop=xlim2+2450000., dt=0.02, lw=2, subtract_2450000=True, zorder=70)    
    my_event.plot_model(color='indigo', ls='--', t_start=xlim1+2450000., t_stop=xlim2+2450000., dt=0.02, subtract_2450000=True, zorder=60,lw=2)
    
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    
    
    plt.xlim(xlims_peak[0], xlims_peak[1])
    plt.ylim(ylims_peak[0], ylims_peak[1])
    
    plt.xlabel(None)
    plt.ylabel(None)
      
    
    
    ''' Residuals '''
    axes = plt.subplot(grid[1])
    my_event_final.plot_residuals(subtract_2450000=True,capsize=2, capthick=1)
    
    
    (source_flux1, blend_flux1) = my_event.get_flux_for_dataset(gaia_filter)
    (source_flux2, blend_flux2) = my_event_final.get_flux_for_dataset(gaia_filter)
    xmodel = np.linspace(xlim1+2450000, xlim2+2450000, num=200)
    ymodel1=my_model.get_lc(xmodel, source_flux=source_flux1[0], blend_flux=blend_flux1)
    ymodel2=my_event_final.model.get_lc(xmodel, source_flux=source_flux2[0], blend_flux=blend_flux2)
    difmodel = ymodel2-ymodel1
    plt.plot(xmodel-2450000, difmodel, ls='--',color='indigo',zorder=10, lw=2)
    
    plt.xlim(xlim1,xlim2)
    
    if ylims_res is None: ylim1_res, ylim2_res = -0.08,0.08
    else: ylim1_res, ylim2_res = np.array(ylims_res)
    
    plt.ylim(ylim1_res, ylim2_res)
    
    if plot_grid:
        plt.grid(zorder=0)
    
    #plt.show()
    
    return my_event, my_event_final


def final_plot_FS_with_peak_plus_minus(datacut_plus, datacut_minus, coords, datasets,magnification_methods, xlims_peak, ylims_peak, xlims=None, ylims_res=None, 
                            u0_hdi=False, rho=None,plot_grid=True,axes_list=[0.18, 0.6, 0.28, 0.25], gaia_filter=0, in_guess=None):
    
    best_plus = extract_params(datacut_plus)[:,0]
    best_minus = extract_params(datacut_minus)[:,0]
    
    if u0_hdi:
        best_plus[1] = get_mode_value(datacut_plus[:,1])[0]
        best_minus[1] = get_mode_value(datacut_minus[:,1])[0]
        
    if rho is not None:
        best_plus[5] = rho
        best_minus[5] = rho
    
    params1 = dict()
    params1['t_0_par'] = best_plus[0]
    params1['t_0'], params1['u_0'], params1['t_E'], params1['pi_E_N'], params1['pi_E_E'], params1['rho'] = best_plus
    my_model_plus = mm.Model(params1, coords = coords)
    
    my_model_plus.set_magnification_methods(magnification_methods)
    my_event_plus = mm.Event(datasets=(datasets), model=my_model_plus) 
    
    params2 = dict()
    params2['t_0_par'] = best_minus[0]
    params2['t_0'], params2['u_0'], params2['t_E'], params2['pi_E_N'], params2['pi_E_E'], params2['rho'] = best_minus
    my_model_minus = mm.Model(params2, coords = coords)
    
    my_model_minus.set_magnification_methods(magnification_methods)
    my_event_minus = mm.Event(datasets=(datasets), model=my_model_minus) 
    
    my_model = mm.Model({'t_0': best_plus[0], 'u_0': best_plus[1], 't_E': best_plus[2]}, coords = coords)
    my_event = mm.Event(datasets=(datasets), model=my_model) 
    quick_fit(my_event, in_guess=in_guess)
    
    if xlims is None: xlim1, xlim2 = default_limits(my_event_plus)
    else: xlim1, xlim2 = np.array(xlims)
    
    
    fig = plt.figure(figsize=(10,6))
    
    ''' Main plot '''
    grid = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    
    axes = plt.subplot(grid[0])
    my_event_plus.plot_data(subtract_2450000=True,capsize=2)
    my_event_plus.plot_model(color='black', t_start=xlim1+2450000., t_stop=xlim2+2450000., lw=2, 
                             subtract_2450000=True, label="$(FSPL+\pi)+$", dt=0.1, zorder=50)
    my_event_minus.plot_model(color='blue', t_start=xlim1+2450000., t_stop=xlim2+2450000., lw=2, 
                              subtract_2450000=True, label="$(FSPL+\pi)-$", dt=0.1, zorder=50)
    my_event.plot_model(color='magenta', ls='--', t_start=xlim1+2450000., t_stop=xlim2+2450000., 
                        subtract_2450000=True, label="PSPL", zorder=50)
    
    plt.xlim(xlim1,xlim2)
    plt.legend(loc='upper right')
    
    if plot_grid:
        plt.grid()
    
    
    ''' Small plot '''
    axes = fig.add_axes(axes_list)
    if plot_grid:
        plt.grid()
    
    my_event_plus.plot_data(subtract_2450000=True,capsize=2)
    my_event_plus.plot_model(color='black', t_start=xlim1+2450000., t_stop=xlim2+2450000., dt=0.02, lw=2, 
                             subtract_2450000=True, zorder=50)
    my_event_minus.plot_model(color='blue', t_start=xlim1+2450000., t_stop=xlim2+2450000., dt=0.02, lw=2, 
                              subtract_2450000=True, zorder=50)
    
    
    my_event.plot_model(color='magenta', ls='--', t_start=xlim1+2450000., t_stop=xlim2+2450000., dt=0.02, subtract_2450000=True)
    
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    
    
    plt.xlim(xlims_peak[0], xlims_peak[1])
    plt.ylim(ylims_peak[0], ylims_peak[1])
    
    plt.xlabel(None)
    plt.ylabel(None)
      
    
    
    ''' Residuals '''
    axes = plt.subplot(grid[1])
    my_event_plus.plot_residuals(subtract_2450000=True,zorder=5,capsize=2)
    plt.axhline(0, color='magenta', zorder=0, ls='--')
    
    (source_flux1, blend_flux1) = my_event.get_flux_for_dataset(gaia_filter)
    (source_flux2, blend_flux2) = my_event_plus.get_flux_for_dataset(gaia_filter)
    (source_flux3, blend_flux3) = my_event_minus.get_flux_for_dataset(gaia_filter)
    xmodel = np.linspace(xlim1+2450000, xlim2+2450000, num=200)
    ymodel1=my_model.get_lc(xmodel, source_flux=source_flux1[0], blend_flux=blend_flux1)
    ymodel2=my_event_plus.model.get_lc(xmodel, source_flux=source_flux2[0], blend_flux=blend_flux2)
    ymodel3=my_event_minus.model.get_lc(xmodel, source_flux=source_flux3[0], blend_flux=blend_flux3)
    difmodel_plus = ymodel2-ymodel1
    difmodel_minus = ymodel3-ymodel1
    plt.plot(xmodel-2450000, difmodel_plus, ls='-',color='black',zorder=10)
    plt.plot(xmodel-2450000, difmodel_minus, ls='-',color='blue',zorder=10)
 
    plt.xlim(xlim1,xlim2)
    
    if ylims_res is None: ylim1_res, ylim2_res = -0.08,0.08
    else: ylim1_res, ylim2_res = np.array(ylims_res)
    
    plt.ylim(ylim1_res, ylim2_res)
    
    if plot_grid:
        plt.grid(zorder=0)
    

    return my_event, my_event_plus, my_event_minus
    
    
def get_diff_model(event1, event2, gaia_filter=0, xlims=None, n=3000):
    (source_flux1, blend_flux1) = event1.get_flux_for_dataset(gaia_filter)
    (source_flux2, blend_flux2) = event2.get_flux_for_dataset(gaia_filter)
    
    if xlims is None: xlim1, xlim2 = default_limits(event1)
    else: xlim1, xlim2 = np.array(xlims)
    
    xmodel = np.linspace(xlim1+2450000, xlim2+2450000, n)
    ymodel1=event1.model.get_lc(xmodel, source_flux=source_flux1[0], blend_flux=blend_flux1)
    ymodel2=event2.model.get_lc(xmodel, source_flux=source_flux2[0], blend_flux=blend_flux2)
    difmodel = ymodel2-ymodel1
    x = xmodel-2450000
    return x,difmodel, ymodel1, ymodel2
    
        
def plot_diff_model(event1, event2, gaia_filter=0, xlims=None, ylims_res = None, grid=True, n=3000, color='black', absolute=False):
    x, difmodel, _, _ = get_diff_model(event1, event2, gaia_filter,xlims,n)
    if absolute: plt.plot(x, np.abs(difmodel), ls='-',color=color,zorder=10)
    else: plt.plot(x, difmodel, ls='-',color=color,zorder=10) 
    plt.plot(x, np.zeros(len(x)),ls='--',color='magenta')
    plt.ylim(ylims_res)#[0], ylims_res[1])
    if grid: plt.grid()

