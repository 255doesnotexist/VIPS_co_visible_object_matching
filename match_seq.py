import numpy as np


def get_max_connected_agent(selected_ids, remaining_ids, similarity_matrix):
    max_similarity = -1
    max_sim_agent = None

    for agent_id in remaining_ids:
        for selected_id in selected_ids:
            similarity = similarity_matrix[agent_id][selected_id]
            if similarity > max_similarity:
                start_id = selected_id
                max_similarity = similarity
                max_sim_agent = agent_id

    return start_id, max_sim_agent, max_similarity


def construct_similarity_tree(self_id, similarity_matrix):
    num_agents = len(similarity_matrix)
    if self_id >= num_agents:
        return []
    selected_ids = set()
    selected_ids.add(self_id)
    remaining_ids = set(range(num_agents)) - selected_ids
    result_sequences = []

    while remaining_ids:
        start_agent, max_sim_agent, max_similarity = get_max_connected_agent(selected_ids, remaining_ids,
                                                                             similarity_matrix)

        if max_sim_agent is not None and max_similarity > 1e-6:
            selected_ids.add(max_sim_agent)
            remaining_ids.remove(max_sim_agent)
            if start_agent > max_sim_agent:
                start_agent, max_sim_agent = max_sim_agent, start_agent
            result_sequences.append([start_agent, max_sim_agent, max_similarity])
        else:
            break

    return result_sequences