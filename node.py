class Node:
    def __init__(self, id, category, bounding_box, position, world_position, heading):
        self.id = id
        self.category = category
        self.bounding_box = bounding_box
        self.position = position
        self.world_position = world_position
        self.heading = heading

    def __str__(self):
        return f"Node {self.id}: Category={self.category}, Bounding Box={self.bounding_box}, Position={self.tracking_id}, World Position={self.world_position}, Heading={self.heading}"
