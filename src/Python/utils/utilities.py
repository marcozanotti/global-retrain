
import logging
import logging.config


def configure_logging(log_file):

    """Function to configure logging.

    Args:
        log_file (str): Name of the log file.
    """

    grey = '\x1b[38;20m'
    yellow = '\x1b[33;20m'
    red = '\x1b[31;20m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'

    logging.config.dictConfig({ # Centralized logging configuration using dictConfig
        'version': 1, # Configuration schema version
        'disable_existing_loggers': False, # Ensure existing loggers are not disabled
        'handlers': { # Handlers define where and how logs are output
            'file': {
                'class': 'logging.FileHandler',  
                'filename': 'logs/' + log_file,          
                'formatter': 'simple'           
            },
            'console': {
                'class': 'logging.StreamHandler',  
                'formatter':'colored'  
            }
        },
        'formatters': { # Formatters define the structure of the log messages
            'simple': {
                'format': '%(asctime)s : %(name)s : %(levelname)s : %(message)s'
            },
            'colored': {
                'format': yellow + '%(asctime)s : %(name)s : %(levelname)s : %(message)s' + reset
            }
        },
        'root': { # The root logger configuration
            'level': 'INFO',
            'handlers': ['file', 'console']
        }
    })

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

