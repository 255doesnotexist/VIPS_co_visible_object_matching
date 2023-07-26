# edge.py

class Edge:
    def __init__(self, node1, node2, semantic_labels, distance, relative_direction):
        self.node1 = node1
        self.node2 = node2
        self.semantic_labels = semantic_labels
        self.distance = distance
        self.relative_direction = relative_direction
        self.affinity = 0.0

    def get_node1(self):
        return self.node1

    def get_node2(self):
        return self.node2

    def get_semantic_labels(self):
        return self.semantic_labels

    def get_distance(self):
        return self.distance

    def get_relative_direction(self):
        return self.relative_direction

    def get_affinity(self):
        return self.affinity

    def set_affinity(self, affinity):
        self.affinity = affinity

    def __str__(self):
        return f"Edge: Node1={self.node1.id}, Node2={self.node2.id}, Semantic Labels={self.semantic_labels}, Distance={self.distance}, Relative Direction={self.relative_direction}"
