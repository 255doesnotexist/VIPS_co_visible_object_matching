# graph.py

class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, edge):
        self.edges.append(edge)

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges

    def get_node_by_id(self, node_id):
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_edge_by_nodes(self, node1, node2):
        for edge in self.edges:
            if edge.node1 == node1 and edge.node2 == node2:
                return edge
        return None

    def get_adjacent_nodes(self, node):
        adjacent_nodes = []
        for edge in self.edges:
            if edge.node1 == node:
                adjacent_nodes.append(edge.node2)
            elif edge.node2 == node:
                adjacent_nodes.append(edge.node1)
        return adjacent_nodes

    def get_outgoing_edges(self, node):
        outgoing_edges = []
        for edge in self.edges:
            if edge.node1 == node:
                outgoing_edges.append(edge)
        return outgoing_edges

    def get_incoming_edges(self, node):
        incoming_edges = []
        for edge in self.edges:
            if edge.node2 == node:
                incoming_edges.append(edge)
        return incoming_edges

    def get_edge_affinity(self, node1, node2):
        edge = self.get_edge_by_nodes(node1, node2)
        if edge:
            return edge.affinity
        return 0.0

    def set_edge_affinity(self, node1, node2, affinity):
        edge = self.get_edge_by_nodes(node1, node2)
        if edge:
            edge.affinity = affinity

    def get_node_affinity(self, node1, node2):
        node1_adjacent_nodes = self.get_adjacent_nodes(node1)
        node2_adjacent_nodes = self.get_adjacent_nodes(node2)
        common_adjacent_nodes = set(node1_adjacent_nodes).intersection(node2_adjacent_nodes)
        return len(common_adjacent_nodes)

    def set_node_affinity(self, node1, node2, affinity):
        node1.set_affinity(node2, affinity)
        node2.set_affinity(node1, affinity)

    def get_node_by_tracking_id(self, tracking_id):
        for node in self.nodes:
            if node.tracking_id == tracking_id:
                return node
        return None

    def get_node_by_position(self, position):
        for node in self.nodes:
            if node.position == position:
                return node
        return None

    def get_node_by_category(self, category):
        for node in self.nodes:
            if node.category == category:
                return node
        return None

    def get_node_by_bounding_box(self, bounding_box):
        for node in self.nodes:
            if node.bounding_box == bounding_box:
                return node
        return None

    def get_node_by_id(self, node_id):
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_edge_by_id(self, edge_id):
        for edge in self.edges:
            if edge.id == edge_id:
                return edge
        return None

    def get_edge_by_label(self, label):
        for edge in self.edges:
            if edge.label == label:
                return edge
        return None

    def get_edge_by_distance(self, distance):
        for edge in self.edges:
            if edge.distance == distance:
                return edge
        return None

    def get_edge_by_relative_direction(self, relative_direction):
        for edge in self.edges:
            if edge.relative_direction == relative_direction:
                return edge
        return None

    def get_edge_by_semantic_labels(self, semantic_labels):
        for edge in self.edges:
            if edge.semantic_labels == semantic_labels:
                return edge
        return None

    def get_edge_by_nodes(self, node1, node2):
        for edge in self.edges:
            if edge.node1 == node1 and edge.node2 == node2:
                return edge
        return None

    def get_edge_by_nodes_and_semantic_labels(self, node1, node2, semantic_labels):
        for edge in self.edges:
            if edge.node1 == node1 and edge.node2 == node2 and edge.semantic_labels == semantic_labels:
                return edge
        return None

    def get_edge_by_nodes_and_distance(self, node1, node2, distance):
        for edge in self.edges:
            if edge.node1 == node1 and edge.node2 == node2 and edge.distance == distance:
                return edge
        return None

    def get_edge_by_nodes_and_relative_direction(self, node1, node2, relative_direction):
        for edge in self.edges:
            if edge.node1 == node1 and edge.node2 == node2 and edge.relative_direction == relative_direction:
                return edge
        return None

    def get_edge_by_nodes_and_label(self, node1, node2, label):
        for edge in self.edges:
            if edge.node1 == node1 and edge.node2 == node2 and edge.label == label:
                return edge
        return None

    def get_edge_by_nodes_and_semantic_labels_and_distance(self, node1, node2, semantic_labels, distance):
        for edge in self.edges:
            if edge.node1 == node1 and edge.node2 == node2 and edge.semantic_labels == semantic_labels and edge.distance == distance:
                return edge
        return None

    def get_edge_by_nodes_and_semantic_labels_and_relative_direction(self, node1, node2, semantic_labels, relative_direction):
        for edge in self.edges:
            if edge.node1 == node1 and edge.node2 == node2 and edge.semantic_labels == semantic_labels and edge.relative_direction == relative_direction:
                return edge
        return None

    def get_edge_by_nodes_and_semantic_labels_and_label(self, node1, node2, semantic_labels, label):
        for edge in self.edges:
            if edge.node1 == node1 and edge.node2 == node2 and edge.semantic_labels == semantic_labels and edge.label == label:
                return edge
        return None

    def get_edge_by_nodes_and_distance_and_relative_direction(self, node1, node2, distance, relative_direction):
        for edge in self.edges:
            if edge.node1 == node1 and edge.node2 == node2 and edge.distance == distance and edge.relative_direction == relative_direction:
                return edge
        return None

    def get_edge_by_nodes_and_distance_and_label(self, node1, node2, distance, label):
        for edge in self.edges:
            if edge.node1 == node1 and edge.node2 == node2 and edge.distance == distance and edge.label == label:
                return edge
        return None

    def get_edge_by_nodes_and_relative_direction_and_label(self, node1, node2, relative_direction, label):
        for edge in self.edges:
            if edge.node1 == node1 and edge.node2 == node2 and edge.relative_direction == relative_direction and edge.label == label:
                return edge
        return None

    def get_edge_by_nodes_and_semantic_labels_and_distance_and_relative_direction(self, node1, node2, semantic_labels, distance, relative_direction):
        for edge in self.edges:
            if edge.node1 == node1 and edge.node2 == node2 and edge.semantic_labels == semantic_labels and edge.distance == distance and edge.relative_direction == relative_direction:
                return edge
        return None

    def get_edge_by_nodes_and_semantic_labels_and_distance_and_label(self, node1, node2, semantic_labels, distance, label):
        for edge in self.edges:
            if edge.node1 == node1 and edge.node2 == node2 and edge.semantic_labels == semantic_labels and edge.distance == distance and edge.label == label:
                return edge
        return None

    def get_edge_by_nodes_and_semantic_labels_and_relative_direction_and_label(self, node1, node2, semantic_labels, relative_direction, label):
        for edge in self.edges:
            if edge.node1 == node1 and edge.node2 == node2 and edge.semantic_labels == semantic_labels and edge.relative_direction == relative_direction and edge.label == label:
                return edge
        return None

    def get_edge_by_nodes_and_distance_and_relative_direction_and_label(self, node1, node2, distance, relative_direction, label):
        for edge in self.edges:
            if edge.node1 == node1 and edge.node2 == node2 and edge.distance == distance and edge.relative_direction == relative_direction and edge.label == label:
                return edge
        return None

