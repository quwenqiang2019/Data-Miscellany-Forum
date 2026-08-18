import logging


logger = logging.getLogger(__name__)


def  logging_test():
    logger.debug("This is  DEBUG")
    logger.info("This is  INFO")
    logger.warning("This is  WARNING")
    logger.error("This is  ERROR")
    logger.critical("This is  CRITICAL")

