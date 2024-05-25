def optimize_seed_set_based_on_neighbors(
    network, current_seed_set, diversity_factor=0.1
):
    optimized_seed_set = current_seed_set.copy()

    for node in current_seed_set:
        neighbors = network.get_neighbors(node)

        # Evaluate the influence of each neighbor if added to the seed set
        for neighbor in neighbors:
            if neighbor not in optimized_seed_set:
                temp_seed_set = optimized_seed_set.copy()
                temp_seed_set.remove(node)
                temp_seed_set.add(neighbor)

                temp_influence = network.evaluate_fitness(temp_seed_set)
                current_influence = network.evaluate_fitness(optimized_seed_set)

                # Replace node in the seed set if the temporary set has higher influence
                if temp_influence > current_influence:
                    optimized_seed_set.remove(node)
                    optimized_seed_set.add(neighbor)
                    break
                else:
                    # temp_influence == current_influence
                    # and np.random.random() < diversity_factor
                    # ):
                    pass

    return optimized_seed_set
