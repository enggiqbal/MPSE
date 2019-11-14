import networkx as nx
import math


def nodesArray(Gnx):

    nodes = []

    for sourceStr in nx.nodes(Gnx):

        nodes.append(sourceStr)

    return nodes

def euclidean_distance(source, target):

    x_source1 = float(source['pos'].split(",")[0])
    x_target1 = float(target['pos'].split(",")[0])

    y_source1 = float(source['pos'].split(",")[1])
    y_target1 = float(target['pos'].split(",")[1])

    geomDistance = math.sqrt((x_source1 - x_target1)**2 + (y_source1 - y_target1)**2)

    return geomDistance




def scale_graph(GD, alpha):

    H = GD.copy()

    for currVStr in nx.nodes(H):

        currV = H.node[currVStr]

        x = float(currV['pos'].split(",")[0])
        y = float(currV['pos'].split(",")[1])

        x = x * alpha
        y = y * alpha

        currV['pos'] = str(x)+","+str(y)

    return H


def computeScalingFactor(Gnx):

    num = 0
    den = 0

    nodes = nodesArray(Gnx)

    for i in range(0, len(nodes)):

        sourceStr = nodes[i]
        source = Gnx.node[sourceStr]

        for j in range(i+1, len(nodes)):

            targetStr = nodes[j]

            if(sourceStr == targetStr):
                continue

            target = Gnx.node[targetStr]

            graph_theoretic_distance = nx.shortest_path_length(Gnx, sourceStr, targetStr)
            geomDistance = euclidean_distance(source, target)

            if (graph_theoretic_distance <= 0):
                continue

            weight = 1/(graph_theoretic_distance**2)

            num = num + (graph_theoretic_distance * geomDistance * weight)
            den = den + (weight * (geomDistance**2))

    scale = num/den

    return scale


def stress(Gnx):

    GnxOriginal = Gnx.copy()

    alpha = computeScalingFactor(GnxOriginal)
    Gnx = scale_graph(GnxOriginal, alpha)

    vertices = nodesArray(Gnx)

    stress = 0

    for i in range(0, len(vertices)):

        sourceStr = vertices[i]
        source = Gnx.node[sourceStr]

        for j in range(i+1, len(vertices)):

            targetStr =  vertices[j]
            target = Gnx.node[targetStr]

            graph_theoretic_distance = nx.shortest_path_length(Gnx, sourceStr, targetStr)

            eu_dist = euclidean_distance(source, target)

            if (graph_theoretic_distance <= 0):
                continue

            delta_squared = (eu_dist - graph_theoretic_distance)**2
            weight = 1/(graph_theoretic_distance**2)
            stress = stress +  (weight * delta_squared)

    scale_graph(Gnx, 1/alpha)


    stress = round(stress, 3)

    return stress




def stress_multilevel_ratio(GD_prev, GD_curr):


	GD_curr_with_old_positions = GD_curr.copy()

	vertices_old_pos = nx.get_node_attributes(GD_prev, 'pos')

	nx.set_node_attributes(GD_curr_with_old_positions, vertices_old_pos, 'pos')

	stress_curr = stress(GD_curr)
	stress_curr_old_pos = stress(GD_curr_with_old_positions)

	return stress_curr/stress_curr_old_pos

