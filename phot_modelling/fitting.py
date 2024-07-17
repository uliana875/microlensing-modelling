import scipy.optimize as op
import numpy as np
    
def chi2_fun(theta, parameters_to_fit, event):
    for (parameter, value) in zip(parameters_to_fit, theta):
        setattr(event.model.parameters, parameter, value)
    return event.get_chi2()

def quick_fit(my_event,print_result=False, in_guess=None):

        parameters_to_fit = ["t_0", "u_0", "t_E"]
        params = my_event.model.parameters
        
        if in_guess is None: initial_guess = [params.t_0, params.u_0, params.t_E]
        else: initial_guess = in_guess
        
        result = op.minimize(chi2_fun, x0=initial_guess, args=(parameters_to_fit, my_event))
        
        (fit_t_0, fit_u_0, fit_t_E) = result.x
        
        # Save the best-fit parameters
        chi2 = chi2_fun(result.x, parameters_to_fit, my_event)

        if print_result:
            print('scipy.optimize.minimize result:')
            print(result)
            
        return result.x, chi2
   
def quick_fit_parallax(my_event, in_guess=None, print_result=False):

        parameters_to_fit = ["t_0", "u_0", "t_E", "pi_E_N", "pi_E_E"]
        params = my_event.model.parameters
        
        if in_guess is None:
            in_guess = [params.t_0, params.u_0, params.t_E, params.pi_E_N, params.pi_E_E]
            
        
        result = op.minimize(chi2_fun, x0=in_guess, args=(parameters_to_fit, my_event))
    
        (fit_t_0, fit_u_0, fit_t_E, fit_piEN, fit_piEE) = result.x
        
        chi2 = chi2_fun(result.x, parameters_to_fit, my_event)
    
        if print_result:
            print('scipy.optimize.minimize result:')
            print(result)
        
        return result.x, chi2
    
    
def get_fluxes(event):
    """
    Taken from example_10
    For given Event instance extract all the fluxes and return them in
    a list. Odd elements are source fluxes and even ones are blending fluxes.
    These fluxes are in units used by MulensModel, where 1 corresponds to
    22 mag.
    """
    fluxes = []
    for dataset in event.datasets:
        (data_source_flux, data_blend_flux) = event.get_flux_for_dataset(
            dataset)
        fluxes.append(data_source_flux[0])
        fluxes.append(data_blend_flux)

    return fluxes


'''Point source'''

def ln_like_PSPL(theta, event, parameters_to_fit):
    """ likelihood function """
    for key, val in enumerate(parameters_to_fit):
        setattr(event.model.parameters, val, theta[key])

    chi2 = event.get_chi2()
    return -0.5 * chi2


def ln_prior_PSPL(theta, parameters_to_fit):
    """priors - we only reject obviously wrong models"""
    if theta[parameters_to_fit.index("t_E")] < 0.:
        return -np.inf
    return 0.0


def ln_prob_PSPL(theta, event, parameters_to_fit):
    
    #combines likelihood and priors; returns ln(prob) and a list of fluxes
    
    fluxes = [None] * 2 * len(event.datasets)

    ln_prior_ = ln_prior_PSPL(theta, parameters_to_fit)
    if not np.isfinite(ln_prior_):
        return (-np.inf, fluxes)
    ln_like_ = ln_like_PSPL(theta, event, parameters_to_fit)

    # In the cases that source fluxes are negative we want to return
    # these as if they were not in priors.
    if np.isnan(ln_like_):
 
        return (-np.inf, fluxes)

    ln_prob_ = ln_prior_ + ln_like_
    fluxes = get_fluxes(event)
    return (ln_prob_, fluxes)


'''Finite source'''

def ln_like_FS(theta, event, parameters_to_fit):
    """ likelihood function """
    for key, val in enumerate(parameters_to_fit):
        #if ((val == 'rho') and (theta[key]<=0)):
         #   setattr(event.model.parameters, val, np.power(10,theta[key]))
        #else: 
        setattr(event.model.parameters, val, theta[key])

    chi2 = event.get_chi2()
    return -0.5 * chi2


def ln_prior_FS(theta, parameters_to_fit):
    """priors - we only reject obviously wrong models"""
    if ((theta[parameters_to_fit.index("t_E")] <= 0.) or (theta[parameters_to_fit.index("rho")] <= 0.)): # !!!!
        return -np.inf
    return 0.0



def ln_prob_FS(theta, event, parameters_to_fit):
    
   # combines likelihood and priors; returns ln(prob) and a list of fluxes
    
    fluxes = [None] * 2 * len(event.datasets)

    ln_prior_ = ln_prior_FS(theta, parameters_to_fit)
    if not np.isfinite(ln_prior_):
        return (-np.inf, fluxes)
    ln_like_ = ln_like_FS(theta, event, parameters_to_fit)

    # In the cases that source fluxes are negative we want to return
    # these as if they were not in priors.
    if np.isnan(ln_like_):
 
        return (-np.inf, fluxes)

    ln_prob_ = ln_prior_ + ln_like_
    fluxes = get_fluxes(event)
    return (ln_prob_, fluxes)



#log rho
'''def ln_like_FS(theta, event, parameters_to_fit):
    """ likelihood function """
    for key, val in enumerate(parameters_to_fit):
        if ((val == 'rho')):
            #print(theta[key])
            setattr(event.model.parameters, val, np.power(10,-theta[key]))
            #print(event.model.parameters.rho)
        else: setattr(event.model.parameters, val, theta[key])

    chi2 = event.get_chi2()
    return -0.5 * chi2


def ln_prior_FS(theta, parameters_to_fit):
    """priors - we only reject obviously wrong models"""
    if ((theta[parameters_to_fit.index("t_E")] <= 0.) or 
        (theta[parameters_to_fit.index("rho")] <= 0.25) or
        (theta[parameters_to_fit.index("rho")] >= 10.)): # !!!!
        return -np.inf
    else:
        return 0.0



def ln_prob_FS(theta, event, parameters_to_fit):
    
    #combines likelihood and priors; returns ln(prob) and a list of fluxes
    
    fluxes = [None] * 2 * len(event.datasets)

    ln_prior_ = ln_prior_FS(theta, parameters_to_fit)
    if not np.isfinite(ln_prior_):
        return (-np.inf, fluxes)
    
    else:
        ln_like_ = ln_like_FS(theta, event, parameters_to_fit)
        #print(event.model.parameters)
    
        # In the cases that source fluxes are negative we want to return
        # these as if they were not in priors.
        if np.isnan(ln_like_):
     
            return (-np.inf, fluxes)
        else:
            ln_prob_ = ln_prior_ + ln_like_
            fluxes = get_fluxes(event)
            return (ln_prob_, fluxes)
'''