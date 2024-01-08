

my_list = [1, 2, 3, 4, 5]

with open("output.txt", "w") as file:
    for item in my_list:
        file.write(str(item) + "\n")




with open("output.txt", "r") as file:
    processed_files = file.read().splitlines()

print(processed_files)
# processed_files = [int(item) for item in processed_files]

import logging


logger = logging.getLogger(__name__)

logger.debug("This is  DEBUG")
logger.info("This is  INFO")