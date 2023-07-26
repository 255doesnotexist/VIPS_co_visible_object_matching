# edge.py

class Edge:
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2

    def get_node1(self):
        return self.node1

    def get_node2(self):
        return self.node2

    def __str__(self):
        return f"Edge: Node1={self.node1.id}, Node2={self.node2.id}, Semantic Labels={self.semantic_labels}, Distance={self.distance}, Relative Direction={self.relative_direction}"
