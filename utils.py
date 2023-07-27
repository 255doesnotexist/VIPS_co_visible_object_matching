def threshold_matching_results(matching_results, threshold):
    thresholded_results = []
    for node_pair in matching_results:
        if node_pair[2] >= threshold:
            thresholded_results.append(node_pair)
    return thresholded_results
