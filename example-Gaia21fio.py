import pandas as pd
import sys
sys.path.insert(0, '/home/dell/microlensing/microlensing-modelling/phot_modelling')
from all_functions import *
import emcee
#%%
# Coordinates and light curve data from BHTOM (http://bhtom.space/)
coords = '11:34:9.792   -65:27:43.020'
Name = "Gaia21fio"
df = pd.read_csv('/home/dell/microlensing/microlensing-modelling/data/target_'+Name+'_photometry.csv',delimiter=';')

# Drop limits (points with error=-1)
df = df.drop(df[df['Error']<0].index, axis=0)

# Show the filters with the number of data points
show_filters(df)

# Some of the filters have to few points, so they can't be used in the modelling
# Some other, such as ATLAS, V(GaiaSP) and I(GaiaSP), have poor quality data, so they are also not used here
filters_to_drop = ['2MASS(K)', '2MASS(J)', '2MASS(H)',
                   'DECAPS(i)', 'DECAPS(g)', 'DECAPS(r)', 'DECAPS(z)',
                   'ATLAS(o)', 'ATLAS(c)', 'g(GaiaSP)', 'r(GaiaSP)',
                   'V(GaiaSP)', 'I(GaiaSP)']
df = drop_filter(df, filters_to_drop)

# If we want, we can change the filters' names
rename_dict = {'GSA(G)': 'Gaia'}
df = rename_filter(df, rename_dict)

# Generate MulensModel datasets
datasets = generate_datasets(df)
#%%
# Initialize the event, here we try to guess the curve parameters
my_event_parallax = create_event_parallax([2459543, 0.2, 90, 0, 0], datasets, coords)
plot_event(my_event_parallax)

# Fit parameters ueing scipy.optimize.minimize
params, chi2 = quick_fit_parallax(my_event_parallax, print_result=True)

#%%
# Now we have an event with better estimated parameters
my_event_parallax = create_event_parallax(params, datasets, coords)
plot_event(my_event_parallax)
#%%
# Initializations for emcee
parameters_to_fit = ["t_0", "u_0", "t_E", "pi_E_N", "pi_E_E"]

sigmas = [0.05,0.001,0.75,0.0175,0.005]
n_dim = len(parameters_to_fit)
n_walkers = 300
n_steps = 1000

params = get_params(my_event_parallax)
start_1 = [params[p] for p in parameters_to_fit]
start = [start_1 + np.random.randn(n_dim) * sigmas
         for i in range(n_walkers)]
#%%
# Run emcee (this takes some time)
sampler = emcee.EnsembleSampler(
    n_walkers, n_dim, ln_prob_PSPL, 
    args=(my_event_parallax, parameters_to_fit))
sampler.run_mcmc(start, n_steps,progress=True)

# Sound after emcee is finished, useful if the process is long
beep(1,440)
#%%
# Plot posterior distribution
samples = get_samples(sampler, 50, my_event_parallax)
datacut,datacut_flux = get_datacut(samples, my_event_parallax)
corner_plot(datacut)

#%%
# Plot baseline magnitude and blending distribution
flux_labels = get_flux_labels(df)
corner_plot(datacut_flux,flux_labels)
#%%
# final plot
my_event_fit, my_event_final = final_plot_parallax(datacut, coords, datasets, t0_par=params['t_0_par'])
#%%
# save samples
#np.save('/home/dell/microlensing/microlensing-modelling/data/posterior.npy', np.hstack((datacut,datacut_flux)))
