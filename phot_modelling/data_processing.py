import numpy as np
import MulensModel as mm
import matplotlib.pyplot as plt

def generate_datasets(df,MJD_to_JD=True):
   df = df.drop(['Observer','Facility'],axis=1)

   time_index=0

   if MJD_to_JD:
       df['JD'] = df['MJD'] + 2400000.5 
       time_index=-1
       
   datasets = []
   filters = df['Filter'].unique()

   for f in filters:
       dataset = df[df['Filter']==f]
       dataset = dataset.drop(['Filter'],axis=1).to_numpy()
       dataset = np.array(dataset, dtype=float)
       
       Data = mm.MulensData(
           data_list = (dataset[:,time_index], dataset[:,1], dataset[:,2]),
           phot_fmt = 'mag',
           add_2450000 = False,
           #ephemerides_file = gaia_ephem,
           plot_properties={'label': f, 'marker' : 'o'})
            
       datasets.append(Data)
       
   return datasets

def scale_data(df, datasets, my_event_final, make_plot=False):
    filters = df['Filter'].unique()
    #n = len(filters)
    scaling_factors = []
    #dof = len(df)-6
    chi2 = my_event_final.get_chi2()
    for i in range(len(datasets)):
        
        part_chi2 = my_event_final.get_chi2_for_dataset(i)
        
        dof = len(df[df['Filter']==filters[i]])-6-2
        sf = np.sqrt(part_chi2/(dof))#/np.sqrt(part_chi2/chi2) # to ma być równe 1
        print(datasets[i])
        #print(chi2/part_chi2)
        print(round(sf,20))
        print('\n')
        scaling_factors.append(sf)
    
    df2 = df.copy()
    
    scale_mapping = {f: sf for (f,sf) in zip(filters,scaling_factors)}
    df2['Error'] = df2.apply(lambda row: row['Error'] * scale_mapping.get(row['Filter'], 1), axis=1)
    
    if make_plot:
        plt.plot(df['Error'], label='original')
        plt.plot(df2['Error'], label='scaled')
        plt.plot(df2['Error']-df['Error'], label='difference')
        plt.legend()
    
    datasets2 = generate_datasets(df2)
    
    return datasets2, df2

def cut_points(df, filterr, time_lower, time_upper, mag_lower, mag_upper):
    if 'JD' not in df.columns:
        df['JD'] = df['MJD'] + 2400000.5 
    df = df.drop(df[(df['JD']>time_lower) & (df['JD']<time_upper) & 
                    (df['Magnitude']>mag_lower) &  (df['Magnitude']<mag_upper) &
                    (df['Filter']==filterr)].index, axis=0)
    return df

def show_filters(df):
    for f in df['Filter'].unique():
        print(f,': ',len(df[df['Filter']==f]))
        
def drop_filter(df, Filters):
    for f in Filters:
    	df = df.drop(df[df['Filter']==f].index,axis=0)
    return df

def rename_filter(df,rename_dict):
    initial_names = list(rename_dict.keys())
    final_names = list(rename_dict.values())
    df['Filter'] = df['Filter'].replace(initial_names, final_names)
    return df

def get_data_table_data(df):
    for observer in df['Observer'].unique():
        df_reduced = df[df['Observer']==observer]
        print(observer,' ',df_reduced['Filter'].unique(),' ',len(df_reduced) , round(df_reduced['MJD'].min(),2),
              ' ', round(df_reduced['MJD'].max(),2))
        
