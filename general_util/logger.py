import logging
import os
import sys
import oss2

_root_name = 'fangxi'


def get_child_logger(child_name):
    return logging.getLogger(_root_name + '.' + child_name)


def setting_logger(log_file: str):
    model_name = "-".join(log_file.replace('/', ' ').split()[1:])

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(_root_name)
    logger.setLevel(logging.INFO)

    rf_handler = logging.StreamHandler(sys.stderr)
    rf_handler.setLevel(logging.INFO)
    rf_handler.setFormatter(logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                              datefmt='%m/%d/%Y %H:%M:%S'))

    output_dir = './log_dir'
    # output_dir = '/data/output2/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    f_handler = logging.FileHandler(os.path.join(
        output_dir, model_name + '-output.log'))
    f_handler.setLevel(logging.INFO)
    f_handler.setFormatter(logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                                             datefmt='%m/%d/%Y %H:%M:%S'))

    # logger.addHandler(rf_handler)
    logger.addHandler(f_handler)
    return logger


def setting_logger_pai(log_dir, rank):

    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir, exist_ok=True)
    # f_handler = logging.FileHandler(os.path.join(log_dir,  f'{rank}-output.log'))
    # rf_handler = logging.StreamHandler(sys.stderr)  # Error in flush to volume

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(f"{_root_name}.rank{rank}")
    logger.setLevel(logging.INFO)

    return logger


class OSSLoggingHandler(logging.StreamHandler):
    def __init__(self, endpoint, bucket, auth, log_file):
        logging.StreamHandler.__init__(self)
        self._bucket = oss2.Bucket(auth, endpoint, bucket)
        self._log_file = log_file
        self._pos = self._bucket.append_object(self._log_file, 0, '')

    def emit(self, record):
        msg = self.format(record)
        self._pos = self._bucket.append_object(
            self._log_file, self._pos.next_position, msg)


def setting_logger_oss(end_point, bucket, auth, bucket_dir, log_file: str, local_rank=-1):
    model_name = "-".join(log_file.replace('/', ' ').split()[1:])

    oss_handler = OSSLoggingHandler(
        end_point, bucket, auth, bucket_dir + 'log_dir/' + f'{model_name}-{local_rank}' + '-log.txt')

    rf_handler = logging.StreamHandler(sys.stderr)

    logging.basicConfig(handlers=[oss_handler, rf_handler],
                        format='[%(asctime)s] [%(levelname)s] [%(process)d#%(threadName)s] ' +
                        '[%(filename)s:%(lineno)d] %(message)s',
                        level=logging.INFO)

    logger = logging.getLogger(_root_name)

    return logger
