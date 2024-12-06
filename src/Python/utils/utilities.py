
import yaml
import logging
import logging.config
from time import gmtime, strftime
from src.Python.utils.collect_data import create_file_name

def get_config(config_file):
    """Function to load configuration from a YAML file.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration as a dictionary.
    """

    with open(config_file, 'rt') as f:
        config = yaml.safe_load(f.read())

    return config

def configure_logging(config_file, name_list):

    """Function to configure logging.

    Args:
        config (dict): Logging configuration in YAML format.
        log_file_name (str): Name of the log file.
    """

    # grey = '\x1b[38;20m'
    yellow = '\x1b[33;20m'
    # red = '\x1b[31;20m'
    # bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'

    time_suffix = strftime("%Y%m%d_%H%M%S", gmtime())
    log_file = create_file_name(
        name_list = name_list + [time_suffix], 
        ext = '.log'
    )
    cfg = get_config(config_file)
    cfg['handlers']['file']['filename'] = 'logs/' + log_file   
    f = cfg['formatters']['colored']['format']
    cfg['formatters']['colored']['format'] = yellow + f + reset
    
    logging.config.dictConfig(cfg)

    return

def create_logger(name = None):

    """Function to create a logger.

    Returns:
        logging.Logger: Logger object.
    """

    return logging.getLogger(name)

def stop_logger(logger):

    """Function to stop logging.

    Args:
        logger (logging.Logger): Logger object.
    """

    logger.handlers.clear()
    logging.shutdown()

    return

