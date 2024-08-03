from multiprocessing import Pool, cpu_count

from src.utils import get_result


def main():
    datasets = [
        "enron",
        "soc-twitter-follows",
        "soc-linkedin",
        "congress",
        "hamsterster",
        "food",
        "pgp",
    ]
    population_size = 30
    seed_set_size = 5
    max_iter = 20

    args = [(dataset, population_size, seed_set_size, max_iter) for dataset in datasets]

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(get_result, args)

    for result in results:
        print(result)


if __name__ == "__main__":
    main()
