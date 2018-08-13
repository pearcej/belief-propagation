import matplotlib.pyplot as plt
import matplotlib.animation as animation
from simulate_network import *
from collections import defaultdict


class AnimateGraph:
    def __init__(self, simulation, graph, pos, fig, myfile, frames=200, interval=500, count=0):
        self.simulation = simulation    #instance of the Simulation class
        self.graph = graph         #graph to animate
        self.pos = pos
        self.fig = fig
        self.myfile = myfile
        self.frames = frames    #how many frames the animation will have
        self.interval = interval    #how fast the animation will go
        self.count = count      #keeps track of the time stamp
        self.bs = defaultdict(list)

    def draw_graph(self):
        bs_color = []
        edge_list = []
        self.fig.clf()
        for n in self.graph.nodes():
            node_bs = self.graph.nodes[n]['belief_strength']
            bs_color.append(node_bs)
            for pre in self.graph.predecessors(n):
                edge_weight= self.graph.edges[pre, n]['weight']
                edge_list.append(edge_weight)
        nx.draw_networkx(self.graph, pos=self.pos, node_color=bs_color, cmap='plasma')

    def update_with_bias(self, useless_variable):
        self.bs = self.simulation.analyze_belief_strength_with_bias(self.graph)
        self.simulation.update_belief_strength(self.graph, self.bs)
        self.simulation.update_edge_weight(self.graph)
        #self.simulation.new_friends(self.graph)
        self.simulation.cut_ppl_off(self.graph)
        self.myfile.write("timestamp = " + str(self.count) + "\n")
        self.simulation.write_table(self.graph, self.myfile)
        self.draw_graph()
        self.count += 1
        return self.count

    def update_without_bias(self, useless_variable):
        self.bs = self.simulation.analyze_belief_strength_without_bias(self.graph)
        self.simulation.update_belief_strength(self.graph, self.bs)
        self.simulation.update_edge_weight(self.graph)
        self.simulation.new_friends(self.graph)
        self.simulation.cut_ppl_off(self.graph)
        self.myfile.write("timestamp = " + str(self.count) + "\n")
        self.simulation.write_table(self.graph, self.myfile)
        self.draw_graph()
        self.count += 1
        return self.count

    def animate_with_bias(self):
        self.draw_graph()
        ani = animation.FuncAnimation(self.fig, self.update_with_bias, frames=self.frames, interval=self.interval, repeat=False )#animation function that updates the figure and iterates while calling update()
        plt.show()

    def animate_without_bias(self):
        self.draw_graph()
        ani = animation.FuncAnimation(self.fig, self.update_without_bias, frames=self.frames, interval=self.interval, repeat=False )#animation function that updates the figure and iterates while calling update()
        plt.show()
