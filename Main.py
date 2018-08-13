import networkx as nx
import matplotlib.pyplot as plt
from animate_network import AnimateGraph as Ani
from simulate_network import Simulation


def main():
    #graph_type options are: Simulation.di_watts_strogatz_graph and Simulation.gnp_random_graph
    #if graph_type is di_watts, probability needs to be an int, but if it's gnp it needs to be a float
    s = Simulation(graph_type=Simulation.di_watts_strogatz_graph, size=200, min_con=.05, probability=5, confirmation_bias=False, filename="simulation_results_without_bias_di_frames1000")
    G = s.graph_type(s, seed=None)
    fig, ax = plt.subplots()   # fig is the graph layout where the nodes and edges go.
    # pos = nx.spring_layout(G)
    pos = nx.circular_layout(G)
    myfile = open(s.filename + ".txt", "w")           #opens user-named.txt file to write the table in
    myfile.write("graph_type=Simulation.di_watts_strogatz_graph, size=200, min_con=.05, probability=5, confirmation_bias=False, frames=1000\n")
    A = Ani(s, G, pos, fig, myfile, frames=1000)
    if s.confirmation_bias is True:                 #checks to see if s.confirmation_bias is true, the default is true.
        A.animate_with_bias()                       #It does so here so that it won't be repeatedly checked and waste time
    else:                                           #if confirmation_bias is True it runs an animate method specially made for that instance
        A.animate_without_bias()                    #if confirmation_bias is False it runs an animate method, specially made for that instance
    myfile.close()

main()
