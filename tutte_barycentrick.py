import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


#input: a graph in the form of a dictionary and an outter_face in the form of a list of vertices.
def tutte_embedding(graph, outter_face):
	pos = {} #a dictionary of node positions
	tmp = nx.Graph()
	for edge in outter_face:
		a,b = edge
		tmp.add_edge(a,b)
	tmp_pos = nx.spectral_layout(tmp) #ensures that outter_face is a convex shape
	pos.update(tmp_pos)
	outter_vertices = tmp.nodes()
	remaining_vertices = [x for x in graph.nodes() if x not in outter_vertices]

	size = len(remaining_vertices)
	A = [[0 for i in range(size)] for i in range(size)] #create the the system of equations that will determine the x and y positions of remaining vertices
	b = [0 for i in range(size)] #the elements of theses matrices are indexed by the remaining_vertices list
	C = [[0 for i in range(size)] for i in range(size)]
	d = [0 for i in range(size)]
	for u in remaining_vertices:
		i = remaining_vertices.index(u)
		neighbors = graph.neighbors(u)
		n = len(list(neighbors))
		A[i][i] = 1
		C[i][i] = 1
		for v in neighbors:
			if v in outter_vertices:
				b[i] += float(pos[v][0])/n
				d[i] += float(pos[v][1])/n
			else:
				j = remaining_vertices.index(v)
				A[i][j] = -(1/float(n))
				C[i][j] = -(1/float(n))

	x = np.linalg.solve(A, b)
	y = np.linalg.solve(C, d)
	for u in remaining_vertices:
		i = remaining_vertices.index(u)
		pos[u] = [x[i],y[i]]

	return pos


def kill(graph_matrix, fixed_vertics):
    G = nx.from_numpy_matrix(graph_matrix)
    print(G)
    print(list(G.edges()))
    nx.draw_networkx(G, node_color = "red")
    pos = tutte_embedding(G, fixed_vertics)
    nx.draw_networkx(G, pos, node_color="blue")
    plt.show()


diamond1 = np.matrix([[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]])
clique = np.matrix([[0,1,1,1,1], [1,0,1,1,1], [1,1,0,1,1], [1,1,1,0,1], [1,1,1,1,0]])

# kill(diamond1, [(0,1), (1,4), (4,3), (3,0)])
kill(clique, [(0,1), (1,2), (2,0)])
