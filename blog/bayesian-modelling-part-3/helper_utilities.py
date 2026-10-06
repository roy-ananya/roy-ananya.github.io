### pandas, numpy
import yaml
import pandas as pd
import numpy as np

### Visualising Hierarchical Relationships
import matplotlib.pyplot as plt
import networkx as nx
##### No need to use - # import pygraphviz as pgv
from networkx.drawing.nx_agraph import graphviz_layout

### Probabilistic Programming 
import pystan
import pickle
import arviz as az


configurations_filepath = '<path>'

### Revenue Mean/SD
revenue_location_scales_device = {
    'iphone' : [2000, 400],
    'ipad':    [500, 100],
    'samsung': [1200, 200],
    'huawei':  [900, 100],
    'nokia':   [500, 100]
}

country_weights = {
    'US': 0.2,
    'UK': 0.15,
    'India':0.35,
    'China':0.23,
    'Russia':0.11
}

### Install Probabilities
installs_location_scales_device = {
    'iphone' : 0.55,
    'ipad':    0.2,
    'samsung': 0.75,
    'huawei':  0.30,
    'nokia':   0.2
}

nodes_os     = ['ios','android']
nodes_device = ['iphone','ipad','samsung','huawei','nokia']
nodes_software_versions = ['i1.0.1','i1.0.2','i1.0.3','i10.0.1','i10.0.2', 's1.1','s1.2','s1.3','s1.4','h2.2.1','h2.3','n1.1']
nodes_list = ['os'] + nodes_os + nodes_device + nodes_software_versions
nodes_colors = [0] + [6]*len(nodes_os) + [14]*len(nodes_device) + [19]*len(nodes_software_versions)
os_edges = [('os','android'),('os','ios')]
device_edges = [('ios','iphone'),('ios','ipad'),('android','samsung'),('android','huawei'),('android','nokia')]
software_v_edges = [("iphone","i1.0.1"), ("iphone","i1.0.2"), ("iphone","i1.0.3"),
                   ("ipad","i10.0.1"),("ipad","i10.0.2"),
                   ("samsung","s1.1"), ("samsung","s1.2"), ("samsung","s1.3"), ("samsung","s1.4"),
                   ("huawei","h2.2.1"),("huawei","h2.3"),("nokia","n1.1")]
edges_list = os_edges + device_edges + software_v_edges

### Generating a simulated dataset
def get_configs(config_filepath):
    '''
    Gets the configurations from the yaml config file
    Args
        config_filepath: filepath of the configs yaml file
    Returns
        configs: dictionary of configurations
    '''
    with open(config_filepath, 'r') as conf:
        try:  
            configs = yaml.safe_load(conf)
            return configs
        except yaml.YAMLError as exception:
            print(exception)
                                                                                                                        
def generate_simulated_data(configs,
                            num_total_samples = 10000):
    '''
    Generates simulated data using the configuration specs loaded from the configs YAML file
    Args
        configs: dictionary of configs from YAML
        num_total_samples: Total size of dataset, number of total observations
    Returns
        sim_df: Simulated dataset
    '''
    np.random.seed(12345678)
    sim_df = pd.DataFrame()
    sim_df['country'] = np.random.choice(configs['sim_df_elements']['countries']['names'], num_total_samples, 
                                      p=configs['sim_df_elements']['countries']['probs'])
    sim_df['os'] = np.random.choice(configs['sim_df_elements']['os']['names'], num_total_samples, 
                                      p=configs['sim_df_elements']['os']['probs'])
    sim_df['device_type'] = None
    sim_df['software_version'] = None
    sim_df['discount'] = np.nan
    sim_df['install'] = np.nan
    sim_df['revenue'] = np.nan

    for os in configs['sim_df_elements']['os_device_versions'].keys():  
        device_types = [device_type for device_type in configs['sim_df_elements']['os_device_versions'][os].keys() if device_type != 'prob']
        device_probs = [configs['sim_df_elements']['os_device_versions'][os][dev]['prob']/configs['sim_df_elements']['os_device_versions'][os]['prob'] \
                    for dev in device_types]

        sim_df.loc[sim_df['os']==os, 'device_type'] = np.random.choice(a=device_types, 
                                                                      size=sim_df[sim_df['os']==os].shape[0],replace=True,
                                                                      p=device_probs)
        for device in device_types:
            software_versions = [soft_version for soft_version in configs['sim_df_elements']['os_device_versions'][os][device].keys() \
                                 if soft_version != 'prob']
            software_probs = [
                configs['sim_df_elements']['os_device_versions'][os][device][software]/configs['sim_df_elements']['os_device_versions'][os][device]['prob'] \
                for software in software_versions ]
    #             print(device)
    #             print(software_versions)
    #             print([configs['sim_df_elements']['os_device_versions'][os][device][software]/configs['sim_df_elements']['os_device_versions'][os][device]['prob'] for software in software_versions ])
            sim_df.loc[sim_df['device_type']==device, 'software_version'] = np.random.choice(a=software_versions, 
                                                                          size=sim_df[sim_df['device_type']==device].shape[0],replace=True,
                                                                          p=software_probs)
        
    ### Adding Revenues
    for dev in revenue_location_scales_device.keys():
        df_shape = sim_df[sim_df['device_type']==dev].shape[0]
        sim_df.loc[sim_df['device_type']==dev, 'revenue'] = np.random.normal(revenue_location_scales_device[dev][0], 
                                             revenue_location_scales_device[dev][1],
                                             df_shape)

    ### Adding Discounts based on revenue
    sim_df['discount'] = sim_df[['country','revenue']].apply(lambda s: country_weights[str(s['country'])]*s['revenue'], axis=1)

    ### Adding Number of Installs
    for dev in installs_location_scales_device.keys():
        df_shape = sim_df[sim_df['device_type']==dev].shape[0]
        sim_df.loc[sim_df['device_type']==dev, 'install'] = np.random.binomial(n=1, p=installs_location_scales_device[dev], size = df_shape)

    return sim_df

def map_to_index(df, 
                 columns = ['os','device_type','software_version'], 
                 nodes = [nodes_os,nodes_device,nodes_software_versions]):
    for column, node in zip(columns, nodes):
        nodes_idx = [i+1 for i in range(len(node))] # all indices in stan start from 1, not 0
        tmp       = pd.DataFrame(list(zip(node, nodes_idx)), columns=[column,f'{column}_index'])
        df        = pd.merge(df, tmp, how='left', left_on = column, right_on=column)
    df['id_index'] = list(range(1,df.shape[0]+1))
    return df

def convert_string_cols_numeric(df, 
                                columns_to_convert = ['country','os','device_type','software_version']): 
    for col_to_convert in columns_to_convert:
        df[col_to_convert+'_index'] = None
        mapping = dict(zip(df[col_to_convert].unique(), range(1,1+df[col_to_convert].nunique())))
        df[col_to_convert+'_index'] = df[col_to_convert].apply(lambda s: mapping.get(s) if s in mapping else s)
    df['id_index'] = list(range(1,df.shape[0]+1))
    return df

def get_parent(df, parent_col='os', child_col='device_type'):
    '''
    Get parent for each column
    '''
    tmp = df.groupby([child_col, f'{child_col}_index']).agg({f'{parent_col}_index':'unique', parent_col:['unique']}).reset_index()
    tmp.columns = [child_col, f'{child_col}_index', 'parent_id', 'parent']
    tmp = tmp.sort_values(by=[f'{child_col}_index'])
    tmp['parent_id'] = tmp['parent_id'].apply(lambda s: s[0])
    return tmp

def get_num_children(df, parent_col='os', child_col='device_type'):
    '''
    Getting the number of children for each parent type
    '''
    tmp = df.groupby([parent_col, f'{parent_col}_index']).agg({f'{child_col}_index':['nunique']}).reset_index()
    tmp.columns = [parent_col,f'{parent_col}_index','num_children']
    tmp = tmp.sort_values(by=[f'{parent_col}_index'])
    return tmp


### Visualising Hierarchical Relationships
class HierarchicalGraphViz:
    '''
    Visualising the hierarchical graphical structure
    '''
    def __init__(self, 
                 nodes_list,
                 edges_list,
                 nodes_colors,
                 graph_size = (12,8) ):        
        self.G = nx.DiGraph()
        self.nodes_list   = nodes_list
        self.edges_list   = edges_list
        self.nodes_colors = nodes_colors
        self.graph_size   = graph_size
        ### Adding Nodes in the Graph
        self.G.add_nodes_from(nodes_list)
        ### Adding edges in the Graph
        self.add_edges()
        ### Creating graph aesthetics
        self.nodes_list_colors = dict(list(zip(self.nodes_list, self.nodes_colors)))
        self.node_size = [self.resize_node_graph(node) for node in self.nodes_list]
        self.visualise_graph()
        
    def visualise_graph(self):
        '''
        Visualising the Hierarchical Graph
        '''
        ### PLotting Graphs
        # plt.title('Hierarchy of Operating System-Device and Software Versions')
        pos=graphviz_layout(self.G, prog='dot')
        plt.figure(3,figsize=self.graph_size) 
        nx.draw_networkx(self.G, pos, 
                         with_labels = True, 
                         nodelist    = self.nodes_list ,
                         node_color  = self.nodes_colors, 
                         node_size   = self.node_size, #,node_size=[300,500,500,1000,1000,1000,1000,1000]
                         arrows=True,
                         alpha=0.6)
    
    def add_edges(self):
        '''
        Add multiple edges in the graph from the edge_list
        Args
            edges_list: List of edges in the graph
        '''
        for (edge1,edge2) in self.edges_list:
            self.G.add_edge(edge1,edge2)
        
    def resize_node_graph(self, node_string):
        '''
        Resizing the node sizes given the length of the string
        Args
        '''
        if len(node_string) <=4:
            return 1000
        elif len(node_string) >4 and len(node_string)<=6:
            return 1500
        else:
            return 2000

class ModelEvaluator():
    def __init__(
                self, 
                fit_object, 
                training_data,
                output_filepath='D:/ROY/Data/AppInstallsData/Output/',
                params_of_interest=None, 
                model_filepath='D:/ROY/Data/AppInstallsData/Output/model_fit_medium_v1.pkl'):
        self.fit = fit_object
        self.training_data = training_data
        self.output_filepath = output_filepath
        self.params_of_interest = params_of_interest

        ### Generate and save evaluation plots
        self.tr_plot_mu      = self.generate_trace_plots(['mu_os','mu_device','mu_sw_version'], saved_plot_name = "traceplots_mu.png")
        self.tr_plot_sigma   = self.generate_trace_plots(['sigma_root','sigma_os','sigma_device','sigma_sw_version'], saved_plot_name = "traceplots_sigma.png")
        self.tr_plot_sd      = self.generate_trace_plots(['sd_root','sd_os','sd_device'], saved_plot_name = "traceplots_sd.png")
        self.tr_plot_beta    = self.generate_trace_plots(['beta_country'], saved_plot_name = "traceplots_beta_country.png")

        self.dist_plot_mu    = self.generate_dist_confidence_intervals(params_to_plot=['mu_root','mu_os','mu_device','mu_sw_version'], saved_plot_name = "dist_ci_mu.png")
        self.dist_plot_sigma = self.generate_dist_confidence_intervals(params_to_plot=['sigma_root','sigma_os','sigma_device','sigma_sw_version'],
                                                                       saved_plot_name = "dist_ci_sigma.png")
        self.dist_plot_sd    = self.generate_dist_confidence_intervals(params_to_plot=['sd_root','sd_os','sd_device'], saved_plot_name = "dist_ci_sd.png")
        self.dist_plot_beta  = self.generate_dist_confidence_intervals(params_to_plot=['beta_country'], saved_plot_name = "dist_ci_beta_country.png")
        ### Generate Summary 
        self.generate_model_summary()
        
    def generate_mappings_codes(self):
        self.country_mapping          = self.training_data[['country','country_index']].drop_duplicates(['country'])#.set_index('country_index').to_dict()['country']
        self.os_mapping               = self.training_data[['os','os_index']].drop_duplicates(['os'])#.set_index('os_index').to_dict()['os']
        self.device_mapping           = self.training_data[['device_type','device_type_index']].drop_duplicates(['device_type'])#.set_index('device_type_index').to_dict()['device_type']
        self.software_version_mapping = self.training_data[['software_version','software_version_index']].drop_duplicates(['software_version'])#.set_index('software_version_index').to_dict()['software_version']

    def replace_model_summary_index(self):
        ### Generate Mapping of codes for country, os, device and software version to replace in the summary dataset
        self.generate_mappings_codes()
        ### Re-create Summary index
        summary_index_df = pd.DataFrame(self.summary.index, columns = ['param_name'])
        summary_index_df['param_value'] = summary_index_df['param_name'].apply(lambda s: s.split(']')[0].split('[')[-1])

        summary_index_mapped = summary_index_df[summary_index_df['param_name'].str.contains('root')]
        summary_index_mapped['param_name_re'] = summary_index_mapped['param_name']

        tmp   = summary_index_df[summary_index_df['param_name'].str.contains('os')]
        tmp['param_value'] = tmp['param_value'].apply(lambda x: int(x)+1)
        tmp = pd.merge(tmp, self.os_mapping, left_on=['param_value'], right_on=['os_index'],how='left')
        tmp['param_name_re'] = tmp[['param_name','param_value','os']].apply(lambda s: str(s['param_name']).replace(str(s['param_value']-1),str(s['os'])),axis=1)
        summary_index_mapped = pd.concat([summary_index_mapped,tmp[['param_name','param_value','param_name_re']] ], axis=0)

        tmp   = summary_index_df[summary_index_df['param_name'].str.contains('device')]
        tmp['param_value'] = tmp['param_value'].apply(lambda x: int(x)+1)
        tmp = pd.merge(tmp, self.device_mapping, left_on=['param_value'], right_on=['device_type_index'],how='left')
        tmp['param_name_re'] = tmp[['param_name','param_value','device_type']].apply(lambda s: str(s['param_name']).replace(str(s['param_value']-1),str(s['device_type'])),axis=1)
        summary_index_mapped = pd.concat([summary_index_mapped,tmp[['param_name','param_value','param_name_re']] ], axis=0)

        tmp   = summary_index_df[summary_index_df['param_name'].str.contains('sw_version')]
        tmp['param_value'] = tmp['param_value'].apply(lambda x: int(x)+1)
        tmp = pd.merge(tmp, self.software_version_mapping, left_on=['param_value'], right_on=['software_version_index'],how='left')
        tmp['param_name_re'] = tmp[['param_name','param_value','software_version']].apply(lambda s: str(s['param_name']).replace(str(s['param_value']-1),str(s['software_version'])),axis=1)
        summary_index_mapped = pd.concat([summary_index_mapped,tmp[['param_name','param_value','param_name_re']] ], axis=0)

        tmp   = summary_index_df[summary_index_df['param_name'].str.contains('country')]
        tmp['param_value'] = tmp['param_value'].apply(lambda x: int(x)+1)
        tmp = pd.merge(tmp, self.country_mapping, left_on=['param_value'], right_on=['country_index'],how='left')
        tmp['param_name_re'] = tmp[['param_name','param_value','country']].apply(lambda s: str(s['param_name']).replace(str(s['param_value']-1),str(s['country'])),axis=1)
        summary_index_mapped = pd.concat([summary_index_mapped,tmp[['param_name','param_value','param_name_re']] ], axis=0)

        self.summary_index_mapping = summary_index_mapped

        self.summary.index = summary_index_mapped['param_name_re']
        


    def load_model(self):
        with open(self.model_filepath, "rb") as f:
            data_dict = pickle.load(f)
        self.fit = data_dict['fit']
        self.fit = data_dict[1]

    def generate_model_summary(self):
        self.summary = az.summary(self.fit)
        if self.params_of_interest is not None:
            summary_params = [s for s in list(self.summary.index) for par in self.params_of_interest if par in s ]
        else:
            summary_params =  list(self.summary.index)
        self.summary = self.summary.filter(items = summary_params , axis=0)
        self.replace_model_summary_index()
        print(self.summary.head())

        self.summary.to_csv(self.output_filepath + 'summary.csv') # warning: dont drop the index since it has the parameter names !
        self.stan_summary = self.fit.stansummary()
        # print(self.stan_summary)

        with open(self.output_filepath + "model_summary.pkl", "wb") as f:
            pickle.dump({'summary' : self.summary, 'stan_summary' : self.stan_summary }, f, protocol=-1)

        self.draws_conv_stats = self.fit.to_dataframe()
        with open(self.output_filepath + "model_summary.pkl", "wb") as f:
            pickle.dump({'summary' : self.summary, 'stan_summary' : self.stan_summary , 'draws_conv_stats': self.draws_conv_stats}, f, protocol=-1)

        print("Number of Divergences - ")
        print(self.draws_conv_stats[ ['divergent__','energy__'] ].sum())
        print(self.draws_conv_stats[ ['accept_stat__'] ].mean())
        print("Means and Standard Deviations of sample draws - ")
        self.draws_conv_stats[ [col for col in self.draws_conv_stats.columns if 'mu_device' in col ] ].mean()
        self.draws_conv_stats[ [col for col in self.draws_conv_stats.columns if 'mu_device' in col ] ].std()

    def generate_trace_plots(self, trace_plot_vars, saved_plot_name = "traceplots.png"):
        dimensions = {
            'mu_os_dim_0':   [0,1],
            'sd_os_dim_0':   [0,1],
            'sigma_os_dim_0':[0,1],
            "mu_device_dim_0":    [0,1,2,3,4],
            "sd_device_dim_0":    [0,1,2,3,4],
            "sigma_device_dim_0": [0,1,2,3,4],
            "mu_sw_version_dim_0":    [0,1,2,3,4,5,6,7,8,9,10,11],
            "sd_sw_version_dim_0":    [0,1,2,3,4,5,6,7,8,9,10,11],
            "sigma_sw_version_dim_0": [0,1,2,3,4,5,6,7,8,9,10,11],
            'beta_country_dim_0':     [0,1,2,3,4]
        }
        coordinates = {
            'mu_os_dim_0':   ['ios','android'],
            'sd_os_dim_0':   ['ios','android'],
            'sigma_os_dim_0':['ios','android'],
            'mu_device_dim_0':    ['iphone','ipad','samsung','huawei','nokia'],
            "sd_device_dim_0":    ['iphone','ipad','samsung','huawei','nokia'],
            "sigma_device_dim_0": ['iphone','ipad','samsung','huawei','nokia'],
            'mu_sw_version_dim_0': ['i1.0.1','i1.0.2','i1.0.3','i10.0.1','i10.0.2','s1.1','s1.2','s1.3','s1.4','h2.2.1','h2.3','n1.1'],
            'sd_sw_version_dim_0': ['i1.0.1','i1.0.2','i1.0.3','i10.0.1','i10.0.2','s1.1','s1.2','s1.3','s1.4','h2.2.1','h2.3','n1.1'],
            'sigma_sw_version_dim_0': ['i1.0.1','i1.0.2','i1.0.3','i10.0.1','i10.0.2','s1.1','s1.2','s1.3','s1.4','h2.2.1','h2.3','n1.1'],
            'beta_country_dim_0': ['UK','India','US','China','Russia']
        }
        fit_data = az.convert_to_dataset(self.fit, coords=coordinates, dims = dimensions)
        ax = az.plot_trace(fit_data, var_names=trace_plot_vars, legend = True)
        fig = ax.ravel()[0].figure
        fig.savefig(self.output_filepath + saved_plot_name, format='png')
        return ax

    def generate_dist_confidence_intervals(self, params_to_plot, saved_plot_name = "dist_ci.png"):
        
        dimensions = {
            'mu_os_dim_0':   [0,1],
            'sd_os_dim_0':   [0,1],
            'sigma_os_dim_0':[0,1],
            "mu_device_dim_0":    [0,1,2,3,4],
            "sd_device_dim_0":    [0,1,2,3,4],
            "sigma_device_dim_0": [0,1,2,3,4],
            "mu_sw_version_dim_0":    [0,1,2,3,4,5,6,7,8,9,10,11],
            "sd_sw_version_dim_0":    [0,1,2,3,4,5,6,7,8,9,10,11],
            "sigma_sw_version_dim_0": [0,1,2,3,4,5,6,7,8,9,10,11],
            'beta_country_dim_0':     [0,1,2,3,4]
        }
        coordinates = {
            'mu_os_dim_0':   ['ios','android'],
            'sd_os_dim_0':   ['ios','android'],
            'sigma_os_dim_0':['ios','android'],
            'mu_device_dim_0':    ['iphone','ipad','samsung','huawei','nokia'],
            "sd_device_dim_0":    ['iphone','ipad','samsung','huawei','nokia'],
            "sigma_device_dim_0": ['iphone','ipad','samsung','huawei','nokia'],
            'mu_sw_version_dim_0': ['i1.0.1','i1.0.2','i1.0.3','i10.0.1','i10.0.2','s1.1','s1.2','s1.3','s1.4','h2.2.1','h2.3','n1.1'],
            'sd_sw_version_dim_0': ['i1.0.1','i1.0.2','i1.0.3','i10.0.1','i10.0.2','s1.1','s1.2','s1.3','s1.4','h2.2.1','h2.3','n1.1'],
            'sigma_sw_version_dim_0': ['i1.0.1','i1.0.2','i1.0.3','i10.0.1','i10.0.2','s1.1','s1.2','s1.3','s1.4','h2.2.1','h2.3','n1.1'],
            'beta_country_dim_0': ['UK','India','US','China','Russia']
        }
        
       
        # coords dict[str, iterable] -> A dictionary containing the values that are used as index. The key is the name of the dimension, the values are the index values.
        # dims dict[str, List(str)] -> A mapping from variables to a list of coordinate names for the variable
        fit_data = az.convert_to_dataset(self.fit, coords=coordinates, dims = dimensions)
        # print(fit_data)
        print(f"Dimensions = {fit_data.dims}")
        print(f"Coordinates = {fit_data.coords}")
        axes = az.plot_forest(
            fit_data,
            kind="forestplot",
            var_names= params_to_plot ,
            combined=True,
            ridgeplot_overlap=1.5,
            textsize=8,
            # colors="blue",
            figsize=(12, 8)
        )
        fig = axes.ravel()[0].figure
        fig.savefig(self.output_filepath + saved_plot_name, format='png')
        return axes

