#

import os


class Path(object):
    """
    User-specific path configuration.
    """
    @staticmethod
    def db_root_dir(database=''):
        db_root = '../../data' # VOC will be automatically downloaded
        db_names = ['VOCSegmentation']

        if database == '':
            return db_root

        if database in db_names:
            return os.path.join(db_root, database)

        else:
            raise ValueError('Invalid database {}'.format(database))
