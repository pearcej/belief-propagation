import networkx as nx
import random
import itertools
from networkx import complete_graph


class Simulation:
    def __init__(self, graph_type, size=20, min_con=.2, probability=None, confirmation_bias=True, filename=None):
        self.graph_type = graph_type
        self.size = size
        self.min_con = min_con
        self.probability = probability          #if graph_type is di_watts, this needs to be an int, but if it's gnp it needs to be a float
        self.confirmation_bias = confirmation_bias
        self.filename=filename



    def di_watts_strogatz_graph(self, seed=None):
        """Return a Watts–Strogatz small-world graph as a DiGraph. THIS IS TAKEN FROM THE ORIGINAL
        WATTS_STROGATZ_GRAPH IN NETWORKX, BUT WE MODIFIED IT SO IT WOULD WORK FOR OUR PURPOSES.

        Parameters
        ----------
        self.size : int
            The number of nodes
        self.min_con : int
            Each node is joined with its `k` nearest neighbors in a ring
            topology.
        self.probability : int
            The probability of rewiring each edge
        seed : int, optional
            Seed for random number generator (default=None)

        See Also
        --------
        newman_watts_strogatz_graph()
        connected_watts_strogatz_graph()

        Notes
        -----
        First create a ring over $n$ nodes [1]_.  Then each node in the ring is joined
        to its $k$ nearest neighbors (or $k - 1$ neighbors if $k$ is odd).
        Then shortcuts are created by replacing some edges as follows: for each
        edge $(u, v)$ in the underlying "$n$-ring with $k$ nearest neighbors"
        with probability $p$ replace it with a new edge $(u, w)$ with uniformly
        random choice of existing node $w$.

        In contrast with :func:`newman_watts_strogatz_graph`, the random rewiring
        does not increase the number of edges. The rewired graph is not guaranteed
        to be connected as in :func:`connected_watts_strogatz_graph`.

        References
        ----------
        .. [1] Duncan J. Watts and Steven H. Strogatz,
           Collective dynamics of small-world networks,
           Nature, 393, pp. 440--442, 1998.
        """
        if self.min_con >= self.size:
            raise nx.NetworkXError("k>=n, choose smaller k or larger n")
        if seed is not None:
            random.seed(seed)

        G = nx.DiGraph()
        nodes = list(range(self.size))  # nodes are labeled 0 to n-1
        for i in nodes:
            G.add_node(i, belief_strength=random.randint(-100, 100), uncertainty=random.uniform(0,1), probability=random.uniform(0,1))
        # connect each node to k/2 neighbors
        for j in range(1, (int(self.min_con * self.size) // 2 + 1)):
            targets = nodes[j:] + nodes[0:j]  # first j nodes are now last in list
            G.add_edges_from(zip(nodes, targets))
        # rewire edges from each node
        # loop over all nodes in order (label) and neighbors in order (distance)
        # no self loops or multiple edges allowed
        for j in range(1, (int(self.min_con * self.size) // 2 + 1)):  # outer loop is neighbors
            targets = nodes[j:] + nodes[0:j]  # first j nodes are now last in list
            # inner loop in node order
            for u, v in zip(nodes, targets):
                if random.random() < self.probability:
                    w = random.choice(nodes)
                    # Enforce no self-loops or multiple edges
                    while w == u or G.has_edge(u, w):
                        w = random.choice(nodes)
                        if G.degree(u) >= self.size - 1:
                            break  # skip this rewiring
                    else:
                        G.remove_edge(u, v)
                        G.add_edge(u, w, weight=random.uniform(-1, 1))
        return G

    def gnp_random_graph(self, seed=None, directed=True):
        """ Return a gnp_random_graph small-world graph as a MultiDiGraph. THIS IS TAKEN FROM THE ORIGINAL
        GNP_RANDOM_GRAPH IN NETWORKX, BUT WE MODIFIED IT SO IT WOULD WORK FOR OUR PURPOSES.

        Returns a $G_{n,p}$ random graph, also known as an Erdős-Rényi graph
        or a binomial graph.

        The $G_{n,p}$ model chooses each of the possible edges with probability $p$.

        The functions :func:`binomial_graph` and :func:`erdos_renyi_graph` are
        aliases of this function.

        Parameters
        ----------
        self.size int
            The number of nodes.
        self.probability : float
            Probability for edge creation.
        seed : int, optional
            Seed for random number generator (default=None).
        directed : bool, optional (default=False)
            If True, this function returns a directed graph.

        See Also
        --------
        fast_gnp_random_graph

        Notes
        -----
        This algorithm [2]_ runs in $O(n^2)$ time.  For sparse graphs (that is, for
        small values of $p$), :func:`fast_gnp_random_graph` is a faster algorithm.

        References
        ----------
        .. [1] P. Erdős and A. Rényi, On Random Graphs, Publ. Math. 6, 290 (1959).
        .. [2] E. N. Gilbert, Random Graphs, Ann. Math. Stat., 30, 1141 (1959).
        """
        if directed:
            edges = itertools.permutations(range(self.size), 2)
            G = nx.DiGraph()
        else:
            edges = itertools.combinations(range(self.size), 2)
            G = nx.Graph()
        nodes = list(range(self.size))  # nodes are labeled 0 to n-1
        for i in nodes:
            G.add_node(i, belief_strength=random.randint(-100, 100), uncertainty=random.uniform(0, 1), probability=random.uniform(0, 1))
        if self.probability <= 0:
            return G
        if self.probability >= 1:
            return complete_graph(self.size, create_using=G)

        if seed is not None:
            random.seed(seed)

        for e in edges:
            if random.random() < self.probability:
                G.add_edge(*e, weight= random.uniform(-1,1))
        return G

    def analyze_belief_strength_without_bias(self, G):
        """Calculates a new belief strength for a node based off of the predecessing node's belief strengths and their
        influence while also accounting for the node's own uncertainty in belief strength.
        :param G = a networkX graph
        :return a python dictionary containing nodes as keys and updated belief strength values as values.
        """
        n = []
        nbs_list = []
        for node in G.nodes:    #cycles through the nodes of the graph to mine the attributes
            n.append(node)      #appends each node to a list that will be put into a dictionary
            pbs_list = []
            og_bs = G.nodes[node]['belief_strength']    #mines the numerical value for a nodes belief strength, from a pre-set node attribute
            unc = G.nodes[node]['uncertainty']           #mines the numerical value for a nodes belief uncertainty, from a pre-set node attribute
            prob = G.nodes[node]['probability']
            for pre in G.predecessors(node):
                ew = G.edges[pre, node]['weight']       #mines the numerical value of an edge's weight, from a pre-set edge attribute
                pre_bs = G.nodes[pre]['belief_strength']    #mines the numerical value for a predecessors belief strength, from a pre-set node attribute
                x = ew * pre_bs                 #determines how much a node values its neighbor's opinion.
                pbs_list.append(x)     #puts all values for predecessor belief strangths in a list
            if len(pbs_list) == 0:
                nbs = og_bs
                nbs = int(nbs)
            else:
                apbs = sum(pbs_list)/len(pbs_list) #calculates the average predecessor belief strength value for a node
                nbs = min(og_bs + (0.1*prob*unc*apbs), 100)  # average predecessor's belief strength is added to the original belief strength.
                nbs = max(nbs, -100)
                nbs = int(nbs)
            nbs_list.append(nbs)         #the new belief strengths are appended to a list that will be put into adictionary
        change = dict(zip(n, nbs_list))     #creates a dictionary from two lists which stores the nodes as keys and their new belief strengths as values
        print(change)
        return change                           #this will be used to update the list in a different function

    def analyze_belief_strength_with_bias(self, G):
        """Calculates a new belief strength for a node based off of the predecessing node's belief strengths and their
        influence while also accounting for the node's own uncertainty in belief strength.
        :param G = a networkX graph
        :return a python dictionary containing nodes as keys and updated belief strength values as values.
        """
        n = []
        nbs_list = []
        for node in G.nodes:    #cycles through the nodes of the graph to mine the attributes
            n.append(node)      #appends each node to a list that will be put into a dictionary
            pbs_list = []
            og_bs = G.nodes[node]['belief_strength']    #mines the numerical value for a nodes belief strength, from a pre-set node attribute
            unc = G.nodes[node]['uncertainty']           #mines the numerical value for a nodes belief uncertainty, from a pre-set node attribute
            prob = G.nodes[node]['probability']
            for pre in G.predecessors(node):
                ew = G.edges[pre, node]['weight']       #mines the numerical value of an edge's weight, from a pre-set edge attribute
                pre_bs = G.nodes[pre]['belief_strength']    #mines the numerical value for a predecessors belief strength, from a pre-set node attribute
                x = ew * pre_bs                 #determines how much a node values its neighbor's opinion.
                pbs_list.append(x)     #puts all values for predecessor belief strangths in a list
            if len(pbs_list) == 0:
                nbs = og_bs
                nbs = int(nbs)
            else:
                apbs = sum(pbs_list)/len(pbs_list) #calculates the average predecessor belief strength value for a node
                if apbs*og_bs > 0:
                    if apbs > 0:
                        nbs = min(og_bs + (0.1*prob*unc*apbs), 100)
                    else:
                        nbs = max(og_bs + (0.1*prob*unc*apbs), -100)
                    nbs = int(nbs)
                else:
                    nbs = og_bs
                    nbs = int(nbs)
            nbs_list.append(nbs)         #the new belief strengths are appended to a list that will be put into adictionary
        change = dict(zip(n, nbs_list))     #creates a dictionary from two lists which stores the nodes as keys and their new belief strengths as values
        print(change)
        return change                           #this will be used to update the list in a different function

    def update_belief_strength(self, G, change):
        """Updates original graph by assigning new belief strengths to their respective nodes in the graph.
        :param G = a networkX graph
        :param change = a dictionary containing a new belief strength for each node
        :return an updated networkx graph containing an updated belief strength as a node attribute.
        """
        for node in change:                                     #loops through the dictionary
            G.nodes[node]['belief_strength'] = change[node]     #assigns a new value for belief strength to a node.
        return G

    def new_friends(self, G):
        """Creates new edges using the built in function nx.preferential_attachment to make the network dynamic. Only
        adds edges a percentage of the time, which depends on how high the preferential attachment value is.
        :param G= a networkx digraph
        :return G
        """
        H = G.to_undirected()                       #creates an undirected copy of the original graph
        n = nx.preferential_attachment(H)           #uses the preferential_attachment method from networkx to create friends
        for u, v, p in n:
            chance = random.randint(0, 100)         #chance is a randomly generated number between 0 and 100
            if p >= len(G.edges) and chance >= 90:  #creates a new relationship (edge) between two nodes if their preferential
                    G.add_edge(u, v, weight=random.uniform(-1, 1))   #attachment number is higher than the total number of edges and
            else:                                                    #chance is greater than 90.
                continue
        return G

    def update_edge_weight(self, G):
        """Updates edge weight based on similarity of belief strength.
        :param G = a networkx Digraph
        :return G
        """
        for node in G.nodes:
            n1 = G.nodes[node]['belief_strength']   #gets a node's belief strength
            for pre in G.predecessors(node):
                n2 = G.nodes[pre]['belief_strength']    #gets the node's predecessors' belief strength
                dif = abs(n1-n2)
                if n1*n2> 0:
                    G.edges[pre, node]['weight'] += (dif/2000)      #clean
                else:
                    G.edges[pre, node]['weight'] -= (dif/2000)
        return G

    def cut_ppl_off(self, G):
        """Removes edges if the weight reach below a certain point, and re-adjusts high edge weights to one to keep the
        edge weights within their parameters.
        :param G= a networkx Digraph
        :return G
        """
        for pre, node in list(G.edges):
            ew = G.edges[pre, node]['weight']
            if ew <= -.95:
                G.remove_edge(pre, node)
            elif ew >= 1:
                G.edges[pre, node]['weight'] = 1.0
            else:
                continue
        return G

    def node_table(self, G, myfile):
        """
        Prints out the node and its belief strength.
        :param G:
        :param myfile:
        :return:
        """
        layout = "{0}{1:>6}{2:>6}{3:>6}{4:>6}{5:>6}{6:>6}{7:>6}"
        header = layout.format("Node", "\t", "Belief Strength", "\t", "Self-uncertainty", "\t", "Probability", "\n")
        myfile.write(header)
        for node in G.nodes:
            data = (layout.format(node, '\t', G.nodes[node]['belief_strength'], '\t', "{0:.2f}".format(G.nodes[node]['uncertainty']), "\t", "{0:.2f}".format(G.nodes[node]['probability']), "\n"))
            myfile.write(data)
        return myfile

    def edge_table(self, G, myfile):
        """
        Prints out each node's neighbor, and the edge weight in between them.
        :param G:
        :param myfile:
        :return:
        """
        layout = "{0}{1:>6}{2:>6}{3:>6}"
        header = layout.format("Neighbor", "\t", "Edge Weight", "\n")
        myfile.write(header)
        for pre, node in list(G.edges):
            data = layout.format((pre, node), '\t', "{0:.2f}".format(G.edges[pre, node]['weight']), "\n")
            myfile.write(data)
        return myfile

    def write_table(self, G, myfile):
        """
        Writes the table in a .txt file
        :return:
        """
        self.node_table(G, myfile)
        self.edge_table(G, myfile)
        return myfile
