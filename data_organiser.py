import os
import shutil
import logging
import utils.id_utils as IdUtils

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S', level=logging.INFO)

class DataOrganiser:

    def __init__(self, dataset, src_dir, organised_dir):
        self.src_dir = src_dir
        self.organised_dir = organised_dir
        self.dataset = dataset
        
    def _convertToIds(self, split_dict):
        ids = []
        for participant in split_dict:
            for day in split_dict[participant]:
                for hour in split_dict[participant][day]:
                    ids.append(IdUtils.get_id(participant, day, hour))
        return ids

    def organise(self):

        train_ids = self._convertToIds(self.dataset.get_train_split())
        test_ids = self._convertToIds(self.dataset.get_test_split())

        train_dir = os.path.join(self.organised_dir, "train")
        test_dir = os.path.join(self.organised_dir, "test")

        all_files = os.listdir(self.src_dir)
        train_files = [f for f in all_files if any(id_str in f for id_str in train_ids)]
        test_files = [f for f in all_files if any(id_str in f for id_str in test_ids)]

        assert len(list(set(train_files) & set(test_files))) == 0, \
          "Overlap between train and test"
        
        logging.info("%d train .tfrecords" % len(train_files))
        logging.info("%d test .tfrecords" % len(test_files))

        def copy_to_dir(file, origin, dest):
            if not os.path.exists(dest):
                os.makedirs(dest)
            origin_file = os.path.join(origin, file)
            if os.path.isfile(origin_file):
                dest_file = os.path.join(dest, file)
                if not os.path.exists(dest_file):
                    shutil.copy(origin_file, dest)
                    logging.info("Copied %s!" % origin_file)
                else:
                    logging.info("%s already exists. No need to copy!" % dest_file)
            else:
                raise RuntimeError('File {} does not exist'.format(origin_file))

        for file in train_files:
            copy_to_dir(file, self.src_dir, train_dir)

        for file in test_files:
            copy_to_dir(file, self.src_dir, test_dir)

        logging.info("Done organising")