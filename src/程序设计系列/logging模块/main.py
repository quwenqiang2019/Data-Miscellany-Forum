import logging
import module
from rich.logging import RichHandler

# 配置日志记录器
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    handlers = [RichHandler(show_time=False, show_path=False, keywords=["total", "packages", "Fetching"], rich_tracebacks=True),
                              logging.FileHandler("demo.log")])

# 创建日志记录器
logger = logging.getLogger(__name__)


def main():
    logger = logging.getLogger(__name__)
    logger.debug("This is  DEBUG !!")
    logger.info("This is  INFO !!")
    logger.warning("This is  WARNING !!")
    logger.error("This is  ERROR !!")
    logger.critical("This is  CRITICAL !!")


    try:
        3 / 0
    except Exception as e:
        # logging.error(e)
        logger.exception(e)

if __name__ == '__main__':
    main()
    module.logging_test()