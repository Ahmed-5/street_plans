class AdjacencyGraph():
    def __init__(self, edges, n_vertices, directed=False):
        self.edges = [True]*len(edges)
        self.vertices = []
        self.directed = directed
        self.visited = [False]*n_vertices

        for i in range(n_vertices):
            self.vertices.append([])
            for j in range(n_vertices):
                self.vertices[i].append(0)
        for e in edges:
            self.vertices[e[0]][e[1]] = e[2]

    def _topo_sort(self, node=0, reverse=False):
        nodes=[node]
        index = 1

        self.visited[node] = True
        if reverse:
            for i in range(len(self.vertices)):
                if self.vertices[i][node] > 0:
                    next_node = i
                    if self.visited[next_node]:
                        continue
                    temp = self._topo_sort(next_node, reverse)
                    for i in range(len(temp)):
                        nodes.insert(index+i, temp[i])
        else:
            for i, e in enumerate(self.vertices[node]):
                if e > 0:
                    next_node = i
                    if self.visited[next_node]:
                        continue
                    temp = self._topo_sort(next_node)
                    for i in range(len(temp)):
                        nodes.insert(index+i, temp[i])

        return nodes

    def topo_sort(self, reverse=False):
        topo = []
        for i in range(len(self.vertices)):
            if self.visited[i]==False:
                temp = self._topo_sort(i, reverse)
                temp.extend(topo)
                topo = temp
        self.visited = [False for i in self.visited]
        return topo

    def strongly_connected(self):
        sort_topo = self.topo_sort(True)
        scc = []
        for i in sort_topo:
            if self.visited[i]:
                continue
            scc.append(self._topo_sort(i))
        self.visited = [False for i in self.visited]
        return scc