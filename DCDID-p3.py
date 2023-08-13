# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 10:02:42 2018

@author: Dell
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 16:41:26 2018

@author: Administrator
"""
import networkx as nx
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from collections import defaultdict
import time
import datetime
from itertools import count
from sklearn import metrics
import math
import matplotlib.pyplot as plt  # 画图用
import numpy as np
from cdlib import evaluation
# from some_module import convert_graph_formats




def str_to_int(x):#converting the list of string into list os list of integer
    return [[int(v) for v in line.split()] for line in x]

def node_addition(G,addnodes,communitys):	#The input format for the "communitys" parameter is as follows: {node: community_name}.
    change_comm=set()#Stores community labels that may have undergone changes.
    processed_edges=set()#Processed edges that need to be removed from the added edges."
    
    for u in addnodes:
        neighbors_u=G.neighbors(u)
        neig_comm=set()#The community labels of the neighbors.
        pc=set()  #Stores the edges of newly added     
        for v in neighbors_u:
            neig_comm.add(communitys[v])
            pc.add((u,v))
            pc.add((v,u))#In an undirected graph, each edge is represented only once, even though it 
            #connects two nodes bidirectionally. Adding the same edge twice is not necessary for graph operations.
        if len(neig_comm)>1:   #This addition of a node is not within the community.
           change_comm=change_comm | neig_comm
           
           lab=max(communitys.values())+1
           communitys.setdefault(u,lab)#To assign a community label to vertex u.
           change_comm.add(lab)
        else:
            if len(neig_comm)==1:#Indicates that the node is inside a community or is connected only to one community.
               communitys.setdefault(u,neig_comm[0])#Add the node to the current community
               processed_edges=processed_edges|pc
            else:
                #If the newly added node has no connections with other nodes, assign a new community label.
               communitys.setdefault(u,max(communitys.values())+1)
    
    #Return the communities that may have changed,
    #  the processed edges, and the latest community structure.
    return change_comm,processed_edges,communitys

def node_deletion(G,delnodes,communitys):					#tested, correct
    change_comm=set()#Storage structure for potentially changing community labels.
    processed_edges=set()#Processed edges need to be removed from the added edges.
    for u in delnodes:
        neighbors_u=G.neighbors(u)
        neig_comm=set()#Community labels of neighboring nodes.        
        for v in neighbors_u:
            neig_comm.add(communitys[v])
            processed_edges.add((u,v))
            processed_edges.add((v,u))
        del communitys[u] #Deletion of nodes and communities.
        change_comm=change_comm | neig_comm     
    #Return potentially changing communities, processed edges, and the most recent community structure.  
    return change_comm,processed_edges,communitys

def edge_addition(addedges,communitys):#If adding edges within the community does not result in any 
    #changes, no action will be taken; otherwise, mark it.
    change_comm=set()#Store labels for communities where changes might be detected in the structure.   
    for item in addedges:        
        neig_comm=set()#Label of the community where the neighbor belongs.
        neig_comm.add(communitys[item[0]])#Determine the communities to which the nodes at both ends of an edge belong.
        neig_comm.add(communitys[item[1]])
        if len(neig_comm)>1:   #Indicate that this added edge is not within the community.
           change_comm=change_comm | neig_comm
    return change_comm # Return the communities where changes are possible.

def edge_deletion(deledges,communitys):#If removing an edge within a community 
    #could lead to community changes, it will not change if the edge is outside the community.
    change_comm=set()#Store labels for communities where structural changes may be detected.   
    for item in deledges:        
        neig_comm=set()#Neighbor's community label.
        neig_comm.add(communitys[item[0]])#Determine the communities to which the nodes at both ends of an edge belong.
        neig_comm.add(communitys[item[1]])
        if len(neig_comm)==1:   #Indicate that this added edge is not within the community.
           change_comm=change_comm | neig_comm	
    return change_comm # Return the communities where changes may occur.

def getchangegraph(all_change_comm,newcomm,Gt):
    Gte=nx.Graph()
    com_key=newcomm.keys()
    for v in Gt.nodes():
        if v not in com_key or newcomm[v] in all_change_comm:            
            Gte.add_node(v)
            neig_v= Gt.neighbors(v)        
            for u in neig_v:                          
               if u not in com_key or newcomm[u] in all_change_comm:                   
                   Gte.add_edge(v,u)
                   Gte.add_node(u)          
    
    return Gte


def CDID(Gsub,maxlabel):#G_sub is the subgraph for the potentially changing community structure. Information dynamics is app
    #lied to the subgraph to discover new community structures. maxlabel represents the maximum label among the unchanged 
    # communities' labels.
    
    #initial information
    starttime=datetime.datetime.now()
    Neigb = {}
    info = 0
    # Average Degree、Maximum Degree   
    avg_d = 0
    max_deg = 0
    N = Gsub.number_of_nodes()    
    deg = dict(Gsub.degree())
    max_deg = max(deg.values())
    avg_d = sum(deg.values()) * 1.0 / N

    ti = 1
    list_I = {}  # Store information for each node, initialized as the degree of each node, and dynamically update it at each iteration
    maxinfo = 0
    starttime = datetime.datetime.now()
    for v in Gsub.nodes():
        #Initial information is given
        if deg[v] == max_deg:            
            info_t = 1 + ti * 0
            ti = ti + 1
            maxinfo = info_t            
        else:
            info_t = deg[v] * 1.0 / max_deg
            # info_t=round(random.uniform(0,1),3)
        #    info_t=deg[v]*1.0/max_deg
        #Initial information is stored is list_I
        list_I.setdefault(v, info_t)
        Neigb.setdefault(v, list(Gsub.neighbors(v)))#The neighbors of node v.
        info += info_t#Total info is added
    
    node_order = sorted(list_I.items(), key=lambda t: t[1], reverse=True)
    node_order_list = list(zip(*node_order))[0]#Only the node from the sorted initial information list is stored

    # Calculate the similarity between nodes using the Jaccard coefficient.
    def sim_jkd(u, v):
        list_v = list(Gsub.neighbors(v))#Storing the neighbours
        list_v.append(v)
        list_u = list(Gsub.neighbors(u))
        list_u.append(u)
        t = set(list_v)
        s = set(list_u)
     
        return len(s & t) * 1.0 / len(s | t)

    # Calculate the hop-2 distance between nodes
    def hop2(u, v):
        list_v = (Neigb[v])
        list_u = (Neigb[u])
        t = set(list_v)
        s = set(list_u)
        return len(s & t)
    
    st = {}  # store similarity
    hops = {}  # store hop-2 distance
    hop2v = {}  # store hop-2 distance ratio
    sum_s = {}  # store the sum of neighbor similarity for each node
    avg_sn = {}  # store the local average similarity for each node, where local refers to neighboring nodes
    avg_dn = {}  # store the local average degree for each node   

    #print(list(Neigb[1]))
    for v, Iv in list_I.items():
        #Iv store information on nodes v
        sum_v = 0
        sum_deg = 0        
        tri = nx.triangles(Gsub, v) * 1.0
        listv = Neigb[v]
        # print(v)
        # print(list(listv))
        num_v = len(list(listv))
        sum_deg += deg[v]
      
        for u in listv:
            keys = str(v) + '_' + str(u)
            p = st.setdefault(keys, sim_jkd(v, u))
            h2 = hop2(v, u)
            hops.setdefault(keys, h2)
            if tri == 0:
                if deg[v] == 1:
                    hop2v.setdefault(keys, 1)
                else:
                    hop2v.setdefault(keys, 0)
            else:
                hop2v.setdefault(keys, h2 / tri)
           
            sum_v += p#Store the total similarity of eighbour of v
            sum_deg += deg[u]#adding the degree of all neighbour        
            
            
        sum_s.setdefault(v, sum_v)#Store the total similarity of neighbour of v
        avg_sn.setdefault(v, sum_v * 1.0 / num_v)#Store the average similarity of neighbour of v
        avg_dn.setdefault(v, sum_deg * 1.0 / (num_v + 1))#storing average degree of all neighbour along with node itself 
#    print('begin loop') 
    
#    oldinfo = 0
    info = 0
    t = 0
    # print(list_I)
    while 1:
        info = 0
        t = t + 1
        Imax = 0
        
        for i in range(len(node_order_list)):
            v = node_order_list[i]
            Iv = list_I[v]
            
            for u in Neigb[v]:
                
                # p=sim_jkd(v,u)
                keys = str(v) + '_' + str(u)

                Iu = list_I[u]
                if Iu - Iv < 0:
                    #                           It=It*1.0/E
                    It = 0
                else:
                    It = (math.exp(Iu - Iv) - 1)
                # It=It*1.0*deg[u]/(deg[v]+deg[u])
                if It < 0.0001:
                    It = 0  #
                fuv = It
                #                       print(fuv)
                p = st[keys]
                p1 = p * hop2v[keys]
                Iin = p1 * fuv  #Information prpagation from v-u
                Icost = avg_sn[v] * fuv * (1 - p) / avg_dn[v]#Information loss from v-u
                #                Icost=avg_s*fuv*avg_c/avg_d
                #                Icost=(avg_sn[v])*fuv/avg_dn[v]

                Iin = Iin - Icost
                # print(Iin)
                if Iin < 0:
                    Iin = 0
                Iv = Iv + Iin
                #                       print(v,u,Iin,Icost,Iv,Iu,It)
                if Iin > Imax:
                    Imax = Iin
            # print(v)
            # print(Iv)
            if Iv > maxinfo:
                Iv = maxinfo
            list_I[v] = Iv
            # print(v,u,Iin,Iv,Iu,tempu[0],pu,tempu[1],fuv)
            info += list_I[v]
        # if v==3:
        

        
        if Imax < 0.0001:
            break


    
#    print ('time:', (endtime - starttime).seconds)    
# community partition**************************************************************

    queue = []
    order = []
    community = {}
    lab = maxlabel
    number = 0
    for v, Info in list_I.items():
        if v not in community.keys():
            lab = lab + 1
            queue.append(v)
            order.append(v)
            community.setdefault(v, lab)
            number = number + 1
            while len(queue) > 0:
                node = queue.pop(0)
                for n1 in Neigb[node]:
                    if (not n1 in community.keys()) and (not n1 in queue):
                        if abs(list_I[n1] - list_I[node]) < 0.001:
                            queue.append(n1)
                            order.append(n1)
                            community.setdefault(n1, lab)
                            number = number + 1
        if number == N:
            break

            #    print (order)
            #    print(community)
    order_value = [community[k] for k in sorted(community.keys())]
    commu_num = len(set(order_value))  # 社团数量
    endtime1 = datetime.datetime.now()
    endtime = datetime.datetime.now()
    print('Running Time:', (endtime1 - starttime).seconds)
    return community  #It was returning only nodes whith community nodes like 1:1,2:2....
    #return list_I

def conver_comm_to_lab(comm1):#Convert the community format to have labels as keys and nodes as values.
    overl_community={}
    for node_v,com_lab in comm1.items():
        if com_lab in overl_community.keys():
            overl_community[com_lab].append(node_v)
        else:
            overl_community.update({com_lab:[node_v]})
    return overl_community

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


def drawcommunity(g,partition,filepath):#Function to draw nodes with different color for different community    
    
    pos = nx.spring_layout(g)
    count1 = 0
    t=0
    node_color=['#66CCCC','#FFCC00','#99CC33','#CC6600','#CCCC66','#FF99CC','#66FFFF','#66CC66','#CCFFFF','#CCCC00','#CC99CC','#FFFFCC']*10
#    print(node_color[1])  
    #print(partition)
    # print(set(partition.values()))
    # print(set(partition.keys()))
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


#Matrix Calculation

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
#----------main-----------------
edges_added = set()
edges_removed = set()
nodes_added = set()
nodes_removed = set()
G=nx.Graph()
#edge_file='switch.t01.edges'
edge_file='15node_t01.txt'
#path='./LFR/muw=0.1/'
path='./data/test1/'
with open(path+edge_file,'r') as f:
    
    edge_list=f.readlines()#storing data from file to list
    for edge in edge_list:
        edge=edge.split()
        G.add_node(int(edge[0]))
        G.add_node(int(edge[1]))
        G.add_edge(int(edge[0]),int(edge[1]))
G=G.to_undirected()
# Initial graph
print('Network G0 for Time T0*********************************************')
nx.draw_networkx(G)
fpath='./data/pic/G_0.png'
plt.savefig(fpath)           # Output method 1: Save the image as a png file
plt.show()
#print G.edges()
#comm_file='switch.t01.comm'
comm_file='15node_comm_t01.txt'
with open(path+comm_file,'r') as f:
    comm_list=f.readlines()
    comm_list=str_to_int(comm_list)
comm={}#Used to store the detected community structure, format: {node: community_label}
comm=CDID(G,0)   #Initial community
# Draw communities
print('Communities C0 for Time T0*********************************************')
drawcommunity(G,comm,'./data/pic/community_0.png')
initcomm=conver_comm_to_lab(comm)
print("initcomm is ")
print(initcomm)
comm_va=list(initcomm.values())
# print(comm_va)

#Matrix calculation
calculateMatrix(G,comm_va,comm_list,0)

start=time.time()
G1=nx.Graph()
G2=nx.Graph()
G1=G
#filename='switch.t0'
filename='15node_'
for i in range(2,5):
    print('begin loop:', i-1)
    comm_new_file=open(path+filename+'comm_t0'+str(i)+'.txt','r')
    if i<10:#Reading the community file and checking if less than 10 or not
        edge_list_new_file=open(path+filename+'t0'+str(i)+'.txt','r')
        edge_list_new=edge_list_new_file.readlines()
        comm_new=comm_new_file.readlines()
    elif i==10:
        edge_list_new_file=open(path+'switch.t10.edges','r')
        edge_list_new=edge_list_new_file.readlines()
        comm_new=comm_new_file.readlines()
    else:
        edge_list_new_file=open(path+'switch.t'+str(i)+'.edges','r')
        edge_list_new=edge_list_new_file.readlines()
        comm_new=comm_new_file.readlines()
    comm_new=str_to_int(comm_new)
    for line in edge_list_new:
         temp = line.strip().split()     
         G2.add_edge(int(temp[0]),int(temp[1]))
    print('Network G' + str(i - 1) + ' for Time T' + str(i - 1) +'*********************************************')
    nx.draw_networkx(G2)
    fpath='./data/pic/G_'+str(i-1)+'.png'
    plt.savefig(fpath)           #Output method 1: Save the image as a png file
    plt.show()
    #The total number of nodes in the current time slice and the previous time slice are related to each other.
    total_nodes =set(G1.nodes())| set(G2.nodes())    
    
    #Checking which node and edges are added and removded in new greaph
    nodes_added=set(G2.nodes())-set(G1.nodes())
    # print ('Nodes added:',nodes_added)
    nodes_removed=set(G1.nodes())-set(G2.nodes())
    # print ('Nodes removed:',nodes_removed)
    edges_added = set(G2.edges())-set(G1.edges())
    # print ('Edges added:',edges_added)
    edges_removed = set(G1.edges())-set(G2.edges())
    # print ('Edges removed:',edges_removed)
    all_change_comm=set()
    #Node Addition Handling#############################################################
    #addn_ch_comm=Communities which can change,addn_pro_edges=edges connected to new node,addn_commu=updated community
    addn_ch_comm,addn_pro_edges,addn_commu = node_addition(G2,nodes_added,comm)
    edges_added=edges_added-addn_pro_edges#Remove processed edges
    #    print edges_added
    all_change_comm=all_change_comm | addn_ch_comm
    #    print('addn_ch_comm',addn_ch_comm)
   
    #Node Deletion Handling#############################################################
    deln_ch_comm,deln_pro_edges,deln_commu  = node_deletion(G1,nodes_removed,addn_commu)
    all_change_comm=all_change_comm | deln_ch_comm
    edges_removed=edges_removed-deln_pro_edges
    adde_ch_comm= edge_addition(edges_added,deln_commu)
    all_change_comm=all_change_comm | adde_ch_comm
    #Edge Deletion Handling#############################################################
    dele_ch_comm= edge_deletion(edges_removed,deln_commu)
    all_change_comm=all_change_comm | dele_ch_comm
    unchangecomm=()#Unchanged Community Labels
    newcomm={}#The format "{node: community}"
    newcomm=deln_commu# Edge addition and deletion are only processed on existing nodes,
    # and no new nodes or deleted nodes (already processed) are added
    unchangecomm=set(newcomm.values())-all_change_comm
    unchcommunity={ key:value for key,value in newcomm.items() if value in unchangecomm}#Invariant communities: labels and nodes
    #Identify the subgraph corresponding to the changing communities, then apply information dynamics to the subgraph to discover the new
    #  community structure. Combine the unchanged community structure with the newly discovered structure to obtain the updated community structure.
    Gtemp=nx.Graph()
    Gtemp=getchangegraph(all_change_comm,newcomm,G2)
    unchagecom_maxlabe=0    
    if len(unchangecomm)>0:
        unchagecom_maxlabe=max(unchangecomm)
    if Gtemp.number_of_edges()<1:#Communities remain unchanged
        comm=newcomm
    else:           
        getnewcomm=CDID(Gtemp,unchagecom_maxlabe)
        print('T'+str(i-1)+'Temporal networkdelta_g'+str(i-1)+'*********************************************')
        nx.draw_networkx(Gtemp)
        fpath='./data/pic/delta_g'+str(i-1)+'.png'
        plt.savefig(fpath)  
        plt.show()
        
        #Merge community structures, adding unchanged communities with newly obtained communities
        d=dict(unchcommunity)        
        d.update(getnewcomm)        
        comm=dict(d) #Using the currently obtained community structure as the input for the next iteration
        print('T'+str(i-1)+'Community Structure of the Time Slice.'+str(i-1)+'*********************************************')
        drawcommunity(G2,comm,'./data/pic/community_'+str(i-1)+'.png')
    comm_va=list(conver_comm_to_lab(comm).values())
    calculateMatrix(G2,comm_va,comm_new,i-1)
    
    G1.clear()
    G1.add_edges_from(G2.edges())
    G2.clear()
print ('all done')

