import logging

from src.utils import get_network, profile_code

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Script started.")

# get_result("fsfs", population_size=10, seed_set_size=3, max_iter=10)

# get_result("hamsterster", population_size=60, seed_set_size=5, max_iter=40)

# get_result("pgp", population_size=80, seed_set_size=3, max_iter=30)

profile_code(get_network("enron"), population_size=10, seed_set_size=5, max_iter=5)
# get_result("soc-twitter-follows", population_sisze=10, seed_set_size=3, max_iter=10)
