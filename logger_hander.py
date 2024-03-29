import logging
import os

def set_logger():
    """
        This function will add logger
        :param null
        :return: file handler & stream handler instance
    """
    logger = logging.getLogger(__name__)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(filename)s:%(levelname)s:%(lineno)d:%(message)s')

    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    file_handler = logging.FileHandler('logs/web_cam.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

