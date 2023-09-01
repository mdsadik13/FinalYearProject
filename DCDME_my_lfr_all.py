import networkx as nx
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from collections import defaultdict
import time
import datetime
from sklearn import metrics
import math
import matplotlib.pyplot as plt  # For drawing pictures
import numpy as np
from numpy import linalg as LA
import openpyxl


# global comm_list
# comm_list=[]


def str_to_int(x):
    return [[int(v) for v in line.split()] for line in x]

def node_addition(G, addnodes, communitys):  # The community format entered is {node: community name}
    change_comm = set()  # Community labels that may be found to be changed in the storage structure
    processed_edges = set()  # For processed edges, you need to delete the processed edges from the added edges.

    for u in addnodes:
        neighbors_u = G.neighbors(u)
        neig_comm = set()  # Neighbor's community label
        pc = set()
        for v in neighbors_u:
            if v in communitys:
                neig_comm.add(communitys[v])
            pc.add((u, v))
            pc.add((v, u))  # There is one side in the undirectional map, and it is convenient to add it twice.
        if len(neig_comm) > 1:  # It shows that this joining node is not within the community.
            change_comm = change_comm | neig_comm
            lab = max(communitys.values()) + 1
            communitys.setdefault(u, lab)  # Assign a community tag to u
            change_comm.add(lab)
        else:
            if len(neig_comm) == 1:  # Explain that the node is within the community, or only connected to one community.
                communitys.setdefault(u, list(neig_comm)[0])  # Add nodes to the community
                processed_edges = processed_edges | pc
            else:
                communitys.setdefault(u,
                                      max(communitys.values()) + 1)  # The new node is not connected to other nodes, and a new community label is assigned.

    return change_comm, processed_edges, communitys  # Return to communities that may change, processed edges and the latest community structure.


def node_deletion(G, delnodes, communitys):  # tested, correct
    change_comm = set()  # Community labels that may be found to be changed in the storage structure
    processed_edges = set()  # For the processed side, you need to delete the processed side from the deleted side.
    for u in delnodes:
        neighbors_u = G.neighbors(u)
        neig_comm = set()  # Neighbor's community label
        for v in neighbors_u:
            if v in communitys:
                neig_comm.add(communitys[v])
            processed_edges.add((u, v))
            processed_edges.add((v, u))
        del communitys[u]  # Delete nodes and communities
        change_comm = change_comm | neig_comm
    return change_comm, processed_edges, communitys  # Return to communities that may change, processed edges and the latest community structure.


def edge_addition(addedges, communitys):  # If the joining side is within the community, it will not cause changes in the community, it will not be dealt with, otherwise it will be marked.
    change_comm = set()  # Community labels that may be found to be changed in the storage structure
    #    print addedges
    #    print communitys
    for item in addedges:
        neig_comm = set()  # Neighbor's community label
        neig_comm.add(communitys[item[0]])  # Determine the community where the nodes at both ends are located.
        neig_comm.add(communitys[item[1]])
        if len(neig_comm) > 1:  # It means that this joining is not within the community.
            change_comm = change_comm | neig_comm
    return change_comm  # Return to communities that may change,


def edge_deletion(deledges, communitys):  # If the deleted edge may cause community change within the community, it will not change outside the community.
    change_comm = set()  # Community labels that may be found to be changed in the storage structure
    for item in deledges:
        neig_comm = set()  # Neighbor's community label
        neig_comm.add(communitys[item[0]])  # Determine the community where the nodes at both ends are located.
        neig_comm.add(communitys[item[1]])
        if len(neig_comm) == 1:  # It means that this joining is not within the community.
            change_comm = change_comm | neig_comm

    return change_comm  # Return to communities that may change


def getchangegraph(all_change_comm, newcomm, Gt):
    Gte = nx.Graph()
    com_key = newcomm.keys()
    for v in Gt.nodes():
        if v not in com_key or newcomm[v] in all_change_comm:
            Gte.add_node(v)
            neig_v = Gt.neighbors(v)
            for u in neig_v:
                if u not in com_key or newcomm[u] in all_change_comm:
                    Gte.add_edge(v, u)
                    Gte.add_node(u)

    return Gte


# cdme Community testing****************************************************************

nodecount_comm = defaultdict(int)  #Dictionary which automatically give 0 to new keys


def CDME(G):
    deg = G.degree()#degree of each node in the grapg

    #Academic adar index
    def AA(NA, NB):
        comm_nodes = list(NA & NB)
        sim = 0
        for node in comm_nodes:
            degnode = deg[node]
            if deg[node] == 1:
                degnode = 1.1
            sim = sim + (1.0 / math.log(degnode))
        return sim

    # Compute the jaccard similarity coefficient of two node
    def simjkd(u, v):
        set_v = set(G.neighbors(v))
        set_v.add(v)
        set_u = set(G.neighbors(u))
        set_u.add(u)
        jac = len(set_v & set_u) * 1.0 / len(set_v | set_u)
        return jac

    # Initialize communities of each node
    # In the first stage, a node is a community.i.e node 1 is 1st community
    node_community = dict(zip(G.nodes(), G.nodes()))
    # Compute the core groups
    # The second stage, calculate the core grouping
    st = {}  # storge the AA

    # compute the AA for all pairs of node
    for node in G.nodes():
        Nv = sorted(G.neighbors(node))
        for u in Nv:
            Nu = G.neighbors(u)
            keys = str(node) + '_' + str(u)
            st.setdefault(keys, AA(set(Nv), set(Nu)))
    
    print('AAindex,done')
    for node in G.nodes():
        # The degree of current node
        deg_node = deg[node]
        flag = True
        maxsimdeg = 0#
        selected = node
        if deg_node == 1:#if the node has 1 neighbour than giving the node same community
            node_community[node] = node_community[list(G.neighbors(node))[0]]
        else:
            for neig in G.neighbors(node):
                deg_neig = deg[neig]#Degree of neigbour
                if flag is True and deg_node <= deg_neig:#Checcking if the neigbour has the more degree or not
                    flag = False
                    break

            if flag is False:#That means one neighbour is having more degree than node 
                for neig in sorted(G.neighbors(node)):
                    deg_neig = deg[neig]
                    # Compute the Jaccard similarity coefficient
                    # nodesim =  simjkd(node, neig)
                    # Use the AAindex
                    keys = str(node) + '_' + str(neig)
                    nodesim = st[keys]#Getting node similarities
                    # Compute the node attraction
                    nodesimdeg = deg_neig * nodesim#Calculating the node similarity * degree
                    if nodesimdeg > maxsimdeg:
                        selected = neig#Getting the neighbour with heighest nodesim image
                        maxsimdeg = nodesimdeg#Calculating maximum nodesimdeg
                    node_community[node] = node_community[selected]
    old_persum = -(2 ** 63 - 1)
    old_netw_per = -(2 ** 63 - 1)

    persum = old_persum + 1
    netw_per = old_netw_per + 0.1
    maxit = 5#Maximum iteration
    itern = 0#track current iteration

    print("loop begin:")
    while itern < maxit:
        itern += 1
        old_netw_per = netw_per
        old_persum = persum
        persum = 0
        for node in G.nodes():
            neiglist = sorted(G.neighbors(node))#Getting the list of neighbour of node in sorted order
            cur_p = per(G, node, node_community)#this stores the internal degree of node in the community
            nodeneig_comm = nodecount_comm.keys()#Getting the neighbour communitites
            cur_p_neig = 0

            for neig in neiglist:#neig is the neigbhour of node
                cur_p_neig += per(G, neig, node_community)#Calculating the total degree of neighbours inside the community
            #                #Calculate the number of triangles between the current node and the neighbor node
            #                for neig in neiglist:
            #                    tri=set(neiglist)&set(self.G.neighbors(neig))
            #                    cur_tri += tri
            #                #Degree plus 2 times triangle
            #                cur_p_neig+=2*cur_tri

            #For all communities where neig_comm is the community number
            for neig_comm in nodeneig_comm:#Getting error
                node_pre_comm = node_community[node]#Community of current node
                new_p_neig = 0
                if node_pre_comm != neig_comm:#for all communities except the community of current node
                    node_community[node] = neig_comm
                    new_p = per(G, node, node_community)#Counting no inside degree by changing the community of node to its neighno nodes community

                    if cur_p <= new_p:#Checking if the new inside degree greather or equal to or not
                        if cur_p == new_p:
                            for newneig in neiglist:
                                #Counting the total internal degree of neighbour int neew community
                                new_p_neig += per(G, newneig, node_community)
                            if cur_p_neig < new_p_neig:
                                cur_p = new_p
                                cur_p_neig = new_p_neig
                            else:
                                #Giving its original community
                                node_community[node] = node_pre_comm

                        else:
                            for newneig in neiglist:
                                new_p_neig += per(G, newneig, node_community)
                            cur_p = new_p
                            cur_p_neig = new_p_neig
                    else:
                        node_community[node] = node_pre_comm
            persum += cur_p
    #            print(node_community)
    print("loop done")
    # Convert community form, {noble: community} to {community: [noble]}
    #    graph_result = defaultdict(list)
    #    for item in node_community.keys():
    #        node_comm = int(node_community[item])
    #        graph_result[node_comm].append(item)

    return node_community


# The internal degree of node v in a community
def per(G, v, node_community):
    neiglist1 = G.neighbors(v)
    in_v = 0
    # tri=0
    global nodecount_comm

    for neig in neiglist1:
        if node_community[neig] == node_community[v]:
            in_v += 1
            # tri+=len(set(neiglist1)&set(self.G.neighbors(neig)))
        else:
            nodecount_comm[node_community[neig]] += 1#Storing the neighbur of a community
    cin_v = 1.0 * (in_v * in_v)
    per = cin_v
    return per


# ****************************************************************************

def Errorrate(clusters, classes, n):
    # Calculate the error rate, the formula ||A*A'-C*C'||, A' represents the transpose of the matrix. A stands for the community found, the behavior node, listed as a community, and the node belongs to a community, then the corresponding cross cell is 1, otherwise it is 0, and C is the matrix of the real community.
    A = np.zeros((n, len(clusters)), int)
    C = np.zeros((n, len(classes)), int)
    k = 0
    for nodelist in clusters:
        for node in nodelist:
            A[node - 1][k] = 1
        k = k + 1
    k = 0
    for nodelist in classes:
        for node in nodelist:
            C[node - 1][k] = 1
        k = k + 1
    t = A.dot(A.T) - C.dot(C.T)
    errors = LA.norm(t)
    return errors


def purity_score(clusters, classes):
    """
    Calculate the purity score for the given cluster assignments and ground truth classes
    
    :param clusters: the cluster assignments array
    :type clusters: numpy.array
    
    :param classes: the ground truth classes
    :type classes: numpy.array
    
    :returns: the purity score
    :rtype: float
    """

    A = np.c_[(clusters, classes)]

    n_accurate = 0.

    for j in np.unique(A[:, 0]):
        z = A[A[:, 0] == j, 1]
        x = np.argmax(np.bincount(z))
        n_accurate += len(z[z == x])

    return n_accurate / A.shape[0]


def conver_comm_to_lab(comm1):  # Convert the community format to, the label is the main key, and the node is value.
    overl_community = {}
    for node_v, com_lab in comm1.items():
        if com_lab in overl_community.keys():
            overl_community[com_lab].append(node_v)
        else:
            overl_community.update({com_lab: [node_v]})
    return overl_community


# def getscore(comm_true, comm_dete, num):
#     actual = []
#     baseline = []
#     for j in range(len(comm_true)):  # groundtruth，j Representing each community, j is the name of the community.
#         for c in comm_true[j]:  # Each node in the community represents each node.
#             flag = False
#             for k in range(len(comm_dete)):  # Detected community, k is the name of the community
#                 if c in comm_dete[k] and flag == False:
#                     flag = True
#                     actual.append(j)
#                     baseline.append(k)
#                     break

#     NMI1 = metrics.normalized_mutual_info_score(actual, baseline)
#     ARI1 = metrics.adjusted_rand_score(actual, baseline)
#     Purity1 = purity_score(baseline, actual)
#     # errors=Errorrate(comm_dete,comm_true,num)
#     errors = 0
#     print('nmi', NMI1)
#     print('ari', ARI1)
#     print('purity', Purity1)
#     print('rate error', errors)

#     return NMI1, ARI1, Purity1, errors


def drawcommunity(g,partition,filepath):#Function to draw nodes with different color for different community    
    
    pos = nx.spring_layout(g)
    count1 = 0
    t=0
    node_color=['#66CCCC','#FFCC00','#99CC33','#CC6600','#CCCC66','#FF99CC','#66FFFF','#66CC66','#CCFFFF','#CCCC00','#CC99CC','#FFFFCC']*10
    for com in set(partition.values()) :
        # print(com);
        count1 = count1 + 1.
        list_nodes = [nodes for nodes in partition.keys()
                                    if partition[nodes] == com]
        # print(list_nodes)
        nx.draw_networkx_nodes(g, pos, list_nodes, node_size = 220,
                                    node_color = node_color[t])
        nx.draw_networkx_labels(g, pos)
        t=t+1
    
    nx.draw_networkx_edges(g,pos, label=True,alpha=0.5 )
    plt.savefig(filepath)
    plt.show()


#Matrix calculation starts here
# Modularity

def getQ(Gsub,C,maxl):#Grah,communities, (maximum number of nodes)
    maxl=max(Gsub.nodes())
   
    amat = []  # adjacency matrix
    #rows, cols = (len(Gsub.nodes()), len(Gsub.nodes()))
    rows, cols=(maxl,maxl)
    for i in range(rows):
        row = []
        for j in range(cols):
            # if (i + 1, j + 1) in Gsub.edges():
            if (i+1, j+1) in Gsub.edges():
                row.append(1)
            else:
                row.append(0)
        amat.append(row)
    cmat = []
    #rows, cols = (len(Gsub.nodes()), len(Gsub.nodes()))
    rows, cols = (maxl, maxl)
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(0)
        cmat.append(row)

    # print("cmat",cmat[1][1])

    for ci in C:
        for a in ci:
            for b in ci:
                if int(a) != int(b):
                    cmat[a-1][b-1] = 1
                    cmat[b-1][a-1] = 1
    Q=0#Modularity
    sum=0#sum of contributions
    sum_m=0#sum of expected contributions
    du = Gsub.degree()#(node degrees)
    E = len(Gsub.edges())#(number of edges)
    for i in Gsub.nodes():
        for j in Gsub.nodes():
            sum = sum + (amat[i-1][j-1] -(du[i]*du[j])/(2*E))*cmat[i-1][j-1]
            sum_m = sum_m + ((du[i]*du[j])/(2*E))*cmat[i-1][j-1]

    #sum = sum/(2*m)
    Q=sum/(2*E-sum_m)
    print("modularity",Q)
    return Q


# Conductance

def conductance(graph: nx.Graph, community: object, summary: bool = True) -> object:
    """Fraction of total edge volume that points outside the community.

    .. math:: f(S) = \\frac{c_S}{2 m_S+c_S}

    where :math:`c_S` is the number of community nodes and, :math:`m_S` is the number of community edges

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.conductance(g,communities)

    :References:

    1.Shi, J., Malik, J.: Normalized cuts and image segmentation. Departmental Papers (CIS), 107 (2000)
    """

    # graph = convert_graph_formats(graph, nx.Graph)
    values = []
    for com in community:
        coms = nx.subgraph(graph, com)#Converting the community into grah

        ms = len(coms.edges())#Total no of edges within the grah
        edges_outside = 0#Total no of edges between nodes outside of graph
        for n in coms.nodes():
            neighbors = graph.neighbors(n)
            for n1 in neighbors:
                if n1 not in coms:
                    edges_outside += 1
        try:
            ratio = float(edges_outside) / ((2 * ms) + edges_outside)
        except:
            ratio = 0
        values.append(ratio)

    if summary:#if summary is true then min,max,mean,standard deviation are returned
        return {
        "min": min(values),
        "max": max(values),
        "mean": np.mean(values),
        "std": np.std(values)
        }
    
    return values

#to calculate external edges of a community helps is external density
def external(graph, community):
    external_edges = 0
    for node in community:
        neighbors = graph.neighbors(node)
        for neighbor in neighbors:
            if neighbor not in community:
                external_edges += 1
    return external_edges
#External density
def external_density(Gsub,C):
    #Calculate proportion of edges that are connected to nodes outside the community. 
    ck = 0
    esum = 0#total external edges of all communitites
    for ci in C:
        esum = esum + external(Gsub, ci)
        ck = ck + (len(ci) * (len(ci) - 1))
    #print(esum / 2)
    #print(len(Gsub.nodes()) * len(Gsub.nodes()) - 1)
    #print(ck)
    eden = (esum / 2) / ((len(Gsub.nodes()) * len(Gsub.nodes()) - 1) - (ck))
    print("External density is")
    print(eden)
    return eden
# Coverage

def coverage(Gsub,C):
    maxl=max(Gsub.nodes())
    cmate = [[0] * maxl for _ in range(maxl)]
    for ci in C:
        for a in ci:
            for b in ci:
                if int(a) != int(b):
                    if (a, b) in Gsub.edges():
                        cmate[a - 1][b - 1] = 1
                        cmate[b - 1][a - 1] = 1
    cov=0
    m = len(Gsub.edges())
    for i in Gsub.nodes():
        for j in Gsub.nodes():
            cov=cov+cmate[i-1][j-1]
    cov=cov/(2*m)
    print("coverage",cov)
    return cov

# Cut-Ratio

def cut_ratio(graph: nx.Graph, community: object, summary: bool = True) -> object:
    """Fraction of existing edges (out of all possible edges) leaving the community.

    ..math:: f(S) = \\frac{c_S}{n_S (n − n_S)}

    where :math:`c_S` is the number of community nodes and, :math:`n_S` is the number of edges on the community boundary

    :param graph: a networkx/igraph object
    :param community: NodeClustering object
    :param summary: boolean. If **True** it is returned an aggregated score for the partition is returned, otherwise individual-community ones. Default **True**.
    :return: If **summary==True** a FitnessResult object, otherwise a list of floats.

    Example:

    >>> from cdlib.algorithms import louvain
    >>> from cdlib import evaluation
    >>> g = nx.karate_club_graph()
    >>> communities = louvain(g)
    >>> mod = evaluation.cut_ratio(g,communities)

    :References:

    1. Fortunato, S.: Community detection in graphs. Physics reports 486(3-5), 75–174 (2010)
    """

    values = []
    for com in community:
        coms = nx.subgraph(graph, com)

        ns = len(coms.nodes())
        edges_outside = 0
        for n in coms.nodes():
            neighbors = graph.neighbors(n)
            for n1 in neighbors:
                if n1 not in coms:
                    edges_outside += 1
        try:
            ratio = float(edges_outside) / (ns * (len(graph.nodes()) - ns))
        except:
            ratio = 0
        values.append(ratio)

    if summary:
        return {
        "min": min(values),
        "max": max(values),
        "mean": np.mean(values),
        "std": np.std(values)
        }
    return values

# AVI Average isolability

def Isolability(G, Ci):
    sum1 = 0
    sum2 = 0
    if len(Ci)!=0:

        for i in Ci:
            for j in Ci:
                if (i, j) in G.edges():
                    sum1 += 1

        for i in Ci:
            for j in G:
                if j not in Ci:
                    if (i, j) in G.edges():
                        sum2 += 1

        #print("\n\n",Ci,"------Isolability----",sum1/2,sum2,(sum1 /2)/ (sum1/2 + sum2))
        return (sum1 / 2) / (sum1 / 2 + sum2)
def AVI (G,C):
    sum=0
    for ci in C:
        sum=sum+Isolability(G,ci)
    avg=sum/len(C)
    print("Average Isolability",avg)
    return avg


def getscore(comm_va,comm_list):#comm_list is grpund truth community and comm_va is predicted community
    actual=[]
    baseline=[]
    for j in range(len(comm_va)):#ground truth, where 'j' represents each community, and 'j' is the community name
    	for c in comm_va[j]: #Each node in the community represents a specific element
    		flag=False
    		for k in range(len(comm_list)): #The detected communities, where 'k' represents each community name.
    			if c in comm_list[k] and flag==False:
    				flag=True 
    				actual.append(j)
    				baseline.append(k)
    				break
    print ('nmi', metrics.normalized_mutual_info_score(actual, baseline))
    print ('ari', metrics.adjusted_rand_score(actual, baseline))


# Purity

def purity(communities: object) :
    """Purity is the product of the frequencies of the most frequent labels carried by the nodes within the communities

    :param communities: AttrNodeClustering object
    :return: FitnessResult object

    Example:

    >>> from cdlib.algorithms import eva
    >>> from cdlib import evaluation
    >>> import random
    >>> l1 = ['A', 'B', 'C', 'D']
    >>> l2 = ["E", "F", "G"]
    >>> g = nx.barabasi_albert_graph(100, 5)
    >>> labels=dict()
    >>> for node in g.nodes():
    >>>    labels[node]={"l1":random.choice(l1), "l2":random.choice(l2)}
    >>> communities = eva(g_attr, labels, alpha=0.5)
    >>> pur = evaluation.purity(communities)

    :References:

    1. Citraro, Salvatore, and Giulio Rossetti. "Eva: Attribute-Aware Network Segmentation." International Conference on Complex Networks and Their Applications. Springer, Cham, 2019.
    """

    pur = evaluation.purity(communities.coms_labels)
    return pur



###All MAtrix calculation
def calculateMatrix(G,comm_va,comm_list,i):
    #Matrix calculation
    print("For Timestamp ",i)
    print("No of communities are :")
    print(len(comm_va))
    getQ(G,comm_va,G.number_of_nodes())
    Conductance_values=conductance(G,comm_va,True)
    print("Conductance value is ")
    print(Conductance_values['std'])
    external_density(G,comm_va)
    getscore(comm_va,comm_list)
    coverage(G,comm_va)
    Cut_ratio_value=cut_ratio(G,comm_va,True)
    print("Cut Ratio value is ")
    print(Cut_ratio_value['std'])
    AVI(G,comm_va)




############################################################
# ----------main-----------------
edges_added = set()
edges_removed = set()
nodes_added = set()
nodes_removed = set()
G = nx.Graph()

edge_file='15node_t01.txt'
path='./data/realworlddata1/non_cumulative/'
comm_file='15node_comm_t01.txt'
filename='15node_.txt'
no_of_input=5

edgeFile=[
    "1960_1963_edges_author.txt",
    "1963_1966_edges_author.txt",
    "1966_1969_edges_author.txt",
    "1969_1972_edges_author.txt",
    "1972_1975_edges_author.txt"
]
commFile=[
    "1960_1963_comm.txt",
"1963_1966_comm.txt",
"1966_1969_comm.txt",
"1969_1972_comm.txt",
"1972_1975_comm.txt"
]



if no_of_input>0:

    with open(path + edgeFile[0], 'r') as f:

        edge_list = f.readlines()
        for edge in edge_list:
            edge = edge.split()
            G.add_node(int(edge[0]))
            G.add_node(int(edge[1]))
            G.add_edge(int(edge[0]), int(edge[1]))
        f.close()
    G = G.to_undirected()
    print('Network G0 for Time T0*********************************************')
    nx.draw_networkx(G)
    nodenumber = G.number_of_nodes()
    with open(path + commFile[0], 'r') as f:
        comm_list = f.readlines()
        comm_list = str_to_int(comm_list)  # 真实社区
        f.close()

    # The third stage, the simulation of the Matthew effect stage

    comm = {}  # Used to store the detected community structure, format {node: community tag}
    comm = CDME(G)  # Initial community
    # Painting community
    print('T0 Time Film Community C0*********************************************')
    print(comm)
    # drawcommunity(G,comm,'./data/pic/community_1.png')
    initcomm = conver_comm_to_lab(comm)
    comm_va = list(initcomm.values())
    commu_num = len(comm_va)
    tru_num = len(comm_list)
    print("For the graph ",edgeFile[0])
    calculateMatrix(G,comm_va,comm_list,0)
    
    start = time.time()
    G1 = nx.Graph()
    G2 = nx.Graph()
    G1 = G

    
    for i in range(1, no_of_input):
        print('begin loop:', i)
        comm_new_file = open(path +edgeFile[i], 'r')
        comm_new = comm_new_file.readlines()
        comm_new_file.close()
        comm_new = str_to_int(comm_new)

        edge_list_new_file = open(path +edgeFile[i],  'r')
        edge_list_new = edge_list_new_file.readlines()
        edge_list_new_file.close()

        #    for line in edge_list_old:
        #         temp = line.strip().split()
        #
        #         G1.add_edge(int(temp[0]),int(temp[1]))
        for line in edge_list_new:
            temp = line.strip().split()
            G2.add_edge(int(temp[0]), int(temp[1]))

        # The current time slice and the total number of nodes of the previous time slice are related to the two sets.
        total_nodes = set(G1.nodes()) | set(G2.nodes())

        nodes_added = set(G2.nodes()) - set(G1.nodes())
        # print ('Add the node set to：',nodes_added)
        nodes_removed = set(G1.nodes()) - set(G2.nodes())
        # print ('Delete the node set as：',nodes_removed)

        edges_added = set(G2.edges()) - set(G1.edges())
        # print ('Add the edge set to：',edges_added)
        edges_removed = set(G1.edges()) - set(G2.edges())
        # print ('Delete the side set as：',edges_removed)

        all_change_comm = set()
        # Add node processing #############################################################
        addn_ch_comm, addn_pro_edges, addn_commu = node_addition(G2, nodes_added, comm)

        edges_added = edges_added - addn_pro_edges  # Remove the processed edges.

        all_change_comm = all_change_comm | addn_ch_comm

        # Delete node processing #############################################################

        deln_ch_comm, deln_pro_edges, deln_commu = node_deletion(G1, nodes_removed, addn_commu)
        all_change_comm = all_change_comm | deln_ch_comm
        edges_removed = edges_removed - deln_pro_edges

        # Add edge processing #############################################################
        #    print('edges_added',edges_added)
        adde_ch_comm = edge_addition(edges_added, deln_commu)
        all_change_comm = all_change_comm | adde_ch_comm
        #    print('all_change_comm',all_change_comm)
        # Delete and process #############################################################
        dele_ch_comm = edge_deletion(edges_removed, deln_commu)
        all_change_comm = all_change_comm | dele_ch_comm
        #    print('all_change_comm',all_change_comm)
        unchangecomm = ()  # Unchanged community tags
        newcomm = {}  # The format is {node: community}
        newcomm = deln_commu  # Add and delete edges, only processed on existing nodes, no new nodes, delete nodes (pre-processed)
        unchangecomm = set(newcomm.values()) - all_change_comm
        unchcommunity = {key: value for key, value in newcomm.items() if
                         value in unchangecomm}  # Unchanged community: labels and nodes
        # Find the subgraph corresponding to the changing community, and then use Matthew effect dynamics to find the new community structure for the subgraph, and add the unchanged community structure to get the new community structure.
        #    print('change community:',all_change_comm)
        Gtemp = nx.Graph()
        Gtemp = getchangegraph(all_change_comm, newcomm, G2)

        unchagecom_maxlabe = 0
        if len(unchangecomm) > 0:
            unchagecom_maxlabe = max(unchangecomm)
        #    print('subG',Gtemp.edges())
        if Gtemp.number_of_edges() < 1:  # The community has not changed.
            comm = newcomm
        else:
            getnewcomm = CDME(Gtemp)
            #        print('newcomm',getnewcomm)
            # Merge the community structure, the unchanged plus the newly acquired
            mergecomm = {}  # Merge the yellow format as {node: community}
            mergecomm.update(unchcommunity)
            mergecomm.update(getnewcomm)
            # mergecomm=dict(**unchcommunity, **getnewcomm )
            comm = mergecomm  # Take the currently acquired community structure as the next community input.
            detectcom = list(conver_comm_to_lab(comm).values())
            commu_num = len(detectcom)
            tru_num = len(comm_new)
        #    print detectcom
        comm_va=list(conver_comm_to_lab(comm).values())
        nodenumber = G2.number_of_nodes()
        #        print('T'+str(i-1)+'Time Film's Network Community Structure C'+str(i-1)+'*********************************************')
        #        print(unchcommunity)
        #        print(newcomm)
        #        print(comm)
        #        drawcommunity(G2,comm,'./data/pic/community_'+str(i-1)+'.png')
        #    print ('getcommunity:',conver_comm_to_lab(comm))
        # NMI, ARI, Purity, Errors = getscore(comm_new, detectcom,
        #                                     nodenumber)  # The first reference data is the real community, and the second is the detected community.
        #    print('community number:', len(set(comm.values())))
        #    print(comm)
        print("For the graph ",edgeFile[0])
        calculateMatrix(G2,comm_va,comm_new,i)

        G1.clear()
        G1.add_edges_from(G2.edges())
        G2.clear()
        
print('all done')

