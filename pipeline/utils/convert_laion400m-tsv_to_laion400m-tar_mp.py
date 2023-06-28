import argparse
import json
import os
import tarfile
import uuid
import sys
import braceexpand
import webdataset as wds
from typing import List
import logging
import gc
import os.path as op
import base64
from PIL import Image
from io import BytesIO
import multiprocessing
from multiprocessing import Pool

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--output_dir", type=str)
arg_parser.add_argument("--mp_num", type=int)
arg_parser.add_argument(
    "--tsv_root",
    type=str,
    help="Pass in a root of tsv",
)
args = arg_parser.parse_args()

from tqdm import tqdm


class TSVFile(object):
    def __init__(self,
                 tsv_root: str,
                 tsv_file: str,
                 if_generate_lineidx: bool = False,
                 lineidx: str = None,
                 class_selector: List[str] = None):
        self.tsv_file = op.join(tsv_root,tsv_file)
        self.lineidx = op.splitext(tsv_file)[0] + '.lineidx' \
            if not lineidx else lineidx
        self.lineidx = op.join(tsv_root,self.lineidx)
        self.linelist = op.splitext(tsv_file)[0] + '.linelist'
        self.linelist = op.join(tsv_root,self.linelist)
        self.chunks = op.splitext(tsv_file)[0] + '.chunks'
        self.chunks = op.join(tsv_root,self.chunks)
        self._fp = None
        self._lineidx = None
        self._sample_indices = None
        self._class_boundaries = None
        self._class_selector = class_selector
        self._len = None
        # the process always keeps the process which opens the file.
        # If the pid is not equal to the currrent pid, we will re-open the file.
        self.pid = None
        # generate lineidx if not exist
        if not op.isfile(self.lineidx) and if_generate_lineidx:
            generate_lineidx(self.tsv_file, self.lineidx)

    def __del__(self):
        self.gcidx()
        if self._fp:
            self._fp.close()

    def __str__(self):
        return "TSVFile(tsv_file='{}')".format(self.tsv_file)

    def __repr__(self):
        return str(self)

    def gcidx(self):
        logging.debug('Run gc collect')
        self._lineidx = None
        self._sample_indices = None
        #self._class_boundaries = None
        return gc.collect()

    def get_class_boundaries(self):
        return self._class_boundaries

    def num_rows(self, gcf=False):
        if (self._len is None):
            self._ensure_lineidx_loaded()
            retval = len(self._sample_indices)

            if (gcf):
                self.gcidx()

            self._len = retval

        return self._len

    def seek(self, idx: int):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        try:
            pos = self._lineidx[self._sample_indices[idx]]
        except:
            logging.info('=> {}-{}'.format(self.tsv_file, idx))
            raise
        self._fp.seek(pos)
        return [s.strip() for s in self._fp.readline().split('\t')]

    def seek_first_column(self, idx: int):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        pos = self._lineidx[idx]
        self._fp.seek(pos)
        return read_to_character(self._fp, '\t')

    def get_key(self, idx: int):
        return self.seek_first_column(idx)

    def __getitem__(self, index: int):
        return self.seek(index)

    def __len__(self):
        return self.num_rows()

    def _ensure_lineidx_loaded(self):
        if self._lineidx is None:
            logging.debug('=> loading lineidx: {}'.format(self.lineidx))
            with open(self.lineidx, 'r') as fp:
                lines = fp.readlines()
                lines = [line.strip() for line in lines]
                self._lineidx = [int(line) for line in lines]

            # read the line list if exists
            linelist = None
            if op.isfile(self.linelist):
                with open(self.linelist, 'r') as fp:
                    linelist = sorted(
                        [
                            int(line.strip())
                            for line in fp.readlines()
                        ]
                    )

            if op.isfile(self.chunks):
                self._sample_indices = []
                self._class_boundaries = []
                class_boundaries = json.load(open(self.chunks, 'r'))
                for class_name, boundary in class_boundaries.items():
                    start = len(self._sample_indices)
                    if class_name in self._class_selector:
                        for idx in range(boundary[0], boundary[1] + 1):
                            # NOTE: potentially slow when linelist is long, try to speed it up
                            if linelist and idx not in linelist:
                                continue
                            self._sample_indices.append(idx)
                    end = len(self._sample_indices)
                    self._class_boundaries.append((start, end))
            else:
                if linelist:
                    self._sample_indices = linelist
                else:
                    self._sample_indices = list(range(len(self._lineidx)))

    def _ensure_tsv_opened(self):
        if self._fp is None:
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()

        if self.pid != os.getpid():
            logging.debug('=> re-open {} because the process id changed'.format(self.tsv_file))
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()


def convert_tsv(tsv_id, tsv_root, output_dir):
    try:
        with wds.ShardWriter(output_dir + f"/{tsv_id.replace('.tsv','.').split('-')[-1]}%09d.tar", maxcount=100, maxsize=1e9) as sink:
            cur_tsv_image = TSVFile(tsv_root=tsv_root, tsv_file=tsv_id)
            cur_tsv_caption = TSVFile(tsv_root=tsv_root, tsv_file=tsv_id.replace("image","text"))
            for _ in tqdm(range(cur_tsv_image.__len__()),desc="Converting image"):
                cur_image = cur_tsv_image[_]
                cur_caption = cur_tsv_caption[_]
                assert cur_image[0] == cur_caption[0], f"the file name of {cur_image[0]} does not equals to {cur_caption[0]}"
                key_str = uuid.uuid4().hex
                sink.write({"__key__": key_str, "png": cur_image[1], "txt": eval(cur_caption[1])["captions"][0].encode('utf-8', 'replace').decode()})
 
    except Exception as e:
        print(e)
        return

def main(args, start_number=0):
    os.makedirs(args.output_dir, exist_ok=True)
    tsv_root = args.tsv_root
    tsv_id_list = list(set(cur_file for cur_file in os.listdir(tsv_root) if "tsv" in cur_file and "image" in cur_file))
    tsv_id_list = tsv_id_list + tsv_id_list
    # Set up multiprocessing pool
    pool = Pool(processes=args.mp_num)
    for idx in tqdm(range(0, len(tsv_id_list)), desc="Converting tsv"):
        tsv_id = tsv_id_list[idx]
        pool.apply_async(convert_tsv, args=(tsv_id, tsv_root,args.output_dir))
    pool.close()
    pool.join()

if __name__ == "__main__":
    main(args=args)
