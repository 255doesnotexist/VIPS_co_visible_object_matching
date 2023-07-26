# node.py

class Node:
    def __init__(self, id = None, category, bounding_box, tracking_id = None, position, world_position, heading):
        self.id = id
        self.category = category
        self.bounding_box = bounding_box
        self.tracking_id = tracking_id
        self.position = position
        self.world_position = world_position
        self.heading = heading
        self.affinities = {}

    def set_affinity(self, node, affinity):
        self.affinities[node] = affinity

    def get_affinity(self, node):
        if node in self.affinities:
            return self.affinities[node]
        return 0.0

    def get_adjacent_nodes(self, graph):
        return graph.get_adjacent_nodes(self)

    def get_outgoing_edges(self, graph):
        return graph.get_outgoing_edges(self)

    def get_incoming_edges(self, graph):
        return graph.get_incoming_edges(self)

    def get_edge_affinity(self, node, graph):
        return graph.get_edge_affinity(self, node)

    def set_edge_affinity(self, node, affinity, graph):
        graph.set_edge_affinity(self, node, affinity)

    def get_node_affinity(self, node, graph):
        return graph.get_node_affinity(self, node)

    def set_node_affinity(self, node, affinity, graph):
        graph.set_node_affinity(self, node, affinity)

    def __str__(self):
        return f"Node {self.id}: Category={self.category}, Bounding Box={self.bounding_box}, Tracking ID={self.tracking_id}, Position={self.position}"
