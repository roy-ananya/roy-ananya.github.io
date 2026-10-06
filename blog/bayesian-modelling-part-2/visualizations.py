import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pygraphviz as pgv
from networkx.drawing.nx_agraph import graphviz_layout
from scipy import stats

def plot_bayesian_prior_posterior(true_mean=5.0, true_std=2.0, prior_mean=0.0, prior_std=3.0, 
                          known_data_std=2.0, seed=42):
    """
    Simple visualization of Bayesian updating for normal distribution .
    
    Parameters:
    - true_mean, true_std: True parameters of data generating process
    - prior_mean, prior_std: Prior belief about the mean
    - known_data_std: Assumed known standard deviation of data
    - seed: Random seed for reproducibility
    """
    
    np.random.seed(seed)
    
    # Generate data samples
    data_10 = np.random.normal(true_mean, true_std, 10)
    data_100 = np.random.normal(true_mean, true_std, 100)
    data_500 = np.random.normal(true_mean, true_std, 500)
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Bayesian Learning: How Posterior Beliefs Converge to Truth with More Data', fontsize=16, fontweight='bold')
    
    # Define x range for plotting
    x = np.linspace(-6, 12, 1000)
    
    # Sample sizes to plot
    scenarios = [
        (0, None, "Prior Only"),
        (10, data_10, "10 samples"),
        (100, data_100, "100 samples"), 
        (500, data_500, "500 samples")
    ]
    
    for i, (n_samples, data, title) in enumerate(scenarios):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        # Prior distribution (same for all plots)
        prior = stats.norm(prior_mean, prior_std)
        ax.plot(x, prior.pdf(x), 'b--', linewidth=2, label='Prior', alpha=0.7)
        
        if n_samples == 0:
            # No data case - posterior same as prior
            ax.fill_between(x, prior.pdf(x), alpha=0.3, color='blue')
        else:
            # Calculate posterior parameters using conjugate formulas
            sample_mean = np.mean(data)
            
            # Posterior parameters for Normal-Normal conjugacy (known variance)
            prior_precision = 1 / (prior_std**2)
            data_precision = n_samples / (known_data_std**2)
            
            posterior_precision = prior_precision + data_precision
            posterior_variance = 1 / posterior_precision
            posterior_mean = (prior_precision * prior_mean + data_precision * sample_mean) / posterior_precision
            
            # Plot posterior
            posterior = stats.norm(posterior_mean, np.sqrt(posterior_variance))
            ax.plot(x, posterior.pdf(x), 'r-', linewidth=3, label='Posterior')
            ax.fill_between(x, posterior.pdf(x), alpha=0.3, color='red')
            
            # Add vertical lines for means
            ax.axvline(sample_mean, color='orange', linestyle=':', linewidth=2, 
                      label=f'Sample mean = {sample_mean:.2f}')
            ax.axvline(posterior_mean, color='red', linestyle='-', linewidth=1, alpha=0.8,
                      label=f'Posterior mean = {posterior_mean:.2f}')
        
        # Add true mean line
        ax.axvline(true_mean, color='green', linestyle=':', linewidth=2, 
                  label=f'True mean = {true_mean}')
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-6, 12)
    
    plt.tight_layout()
    plt.show()

def plot_uncertainty_evolution():
    """
    Shows how uncertainty (credible intervals) shrinks with more data.
    """
    # Parameters for Beta-Binomial
    prior_alpha, prior_beta = 2, 2
    
    np.random.seed(42)
    true_prob = 0.65
    max_games = 100
    results = np.random.binomial(1, true_prob, max_games)
    
    # Calculate posterior parameters for different sample sizes
    sample_sizes = range(0, max_games + 1, 2)
    posterior_means = []
    credible_intervals = []
    
    for n in sample_sizes:
        if n == 0:
            wins = 0
        else:
            wins = np.sum(results[:n])
        
        losses = n - wins
        post_alpha = prior_alpha + wins
        post_beta = prior_beta + losses
        
        # Posterior mean
        post_mean = post_alpha / (post_alpha + post_beta)
        posterior_means.append(post_mean)
        
        # 95% credible interval
        post_dist = stats.beta(post_alpha, post_beta)
        ci_lower = post_dist.ppf(0.025)
        ci_upper = post_dist.ppf(0.975)
        credible_intervals.append((ci_lower, ci_upper))
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Evolution of posterior mean and credible intervals
    ci_lower = [ci[0] for ci in credible_intervals]
    ci_upper = [ci[1] for ci in credible_intervals]
    
    ax1.plot(sample_sizes, posterior_means, 'r-', linewidth=2, label='Posterior Mean')
    ax1.fill_between(sample_sizes, ci_lower, ci_upper, alpha=0.3, color='red', 
                     label='95% Credible Interval')
    ax1.axhline(true_prob, color='green', linestyle=':', linewidth=2, 
                label=f'True Probability = {true_prob}')
    
    ax1.set_xlabel('Number of Games')
    ax1.set_ylabel('Win Probability')
    ax1.set_title('Convergence of Posterior Mean and Shrinking Uncertainty', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Width of credible interval over time
    ci_widths = [ci[1] - ci[0] for ci in credible_intervals]
    
    ax2.plot(sample_sizes, ci_widths, 'purple', linewidth=2, marker='o', markersize=3)
    ax2.set_xlabel('Number of Games')
    ax2.set_ylabel('Width of 95% Credible Interval')
    ax2.set_title('Uncertainty Reduction with More Data', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


class HierarchicalGraphViz:
    '''
    Visualising the hierarchical graphical structure
    '''
    def __init__(self, 
                 nodes_list,
                 edges_list,
                 nodes_colors):        
        self.G = nx.DiGraph()
        self.nodes_list = nodes_list
        self.edges_list = edges_list
        self.nodes_colors = nodes_colors
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
        plt.figure(3,figsize=(12,8)) 
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
        
# nodes_os =['ios','android']#['androi d','ios']
# nodes_device = ['iphone','ipad','samsung','huawei','nokia']
# nodes_software_versions = ['i1.0.1','i1.0.2','i1.0.3','i10.0.1','i10.0.2', 's1.1','s1.2','s1.3','s1.4','h2.2.1','h2.3','n1.1']
# nodes_list = ['os'] + nodes_os + nodes_device + nodes_software_versions
# nodes_colors = [0] + [6]*len(nodes_os) + [14]*len(nodes_device) + [19]*len(nodes_software_versions)
# os_edges = [('os','android'),('os','ios')]
# device_edges = [('ios','iphone'),('ios','ipad'),('android','samsung'),('android','huawei'),('android','nokia')]
# software_v_edges = [("iphone","i1.0.1"), ("iphone","i1.0.2"), ("iphone","i1.0.3"),
#                    ("ipad","i10.0.1"),("ipad","i10.0.2"),
#                    ("samsung","s1.1"), ("samsung","s1.2"), ("samsung","s1.3"), ("samsung","s1.4"),
#                    ("huawei","h2.2.1"),("huawei","h2.3"),("nokia","n1.1")]
# edges_list = os_edges + device_edges + software_v_edges

# HierarchicalGraphViz(nodes_list, edges_list, nodes_colors) 