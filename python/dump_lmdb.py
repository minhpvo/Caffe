import os
import sys
import lmdb
import numpy as np
from argparse import ArgumentParser, RawTextHelpFormatter
from scipy.misc import imsave

from caffe.io import datum_to_array
from caffe.proto.caffe_pb2 import Datum


def main(args):
    datum = Datum()
    env = lmdb.open(args.input_db)
    with env.begin() as txn:
        cursor = txn.cursor()
        cursor.first()
        if args.image_ext is not None:
            # images
            if not os.path.isdir(args.output):
                os.makedirs(args.output)
            for i, (key, value) in enumerate(cursor):
                if args.max_num is not None and i >= args.max_num: break
                name = os.path.splitext(os.path.basename(key))[0]
                file_path = os.path.join(args.output,
                            '{:010d}_{}.{}'.format(i, name, args.image_ext))
                datum.ParseFromString(value)
                if args.encoded:
                    with open(file_path, 'wb') as f:
                        f.write(datum.data)
                else:
                    img = datum_to_array(datum)
                    img = img.transpose(1, 2, 0)  # CxHxW -> HxWxC
                    img = img[:, :, ::-1]  # BGR -> RGB
                    imsave(file_path, img)
        else:
            # features
            datum.ParseFromString(cursor.value())
            feat = datum_to_array(datum)
            num = env.stat()['entries']
            num = num if args.max_num is None else min(num, args.max_num)
            dim = feat.shape if args.keepdims else (feat.size,)
            data = np.empty((num,) + dim, dtype=feat.dtype)
            for i, (key, value) in enumerate(cursor):
                if args.max_num is not None and i >= args.max_num: break
                datum.ParseFromString(value)
                feat = datum_to_array(datum)
                data[i] = feat if args.keepdims else feat.ravel()
            np.save(args.output, data)


if __name__ == '__main__':
    parser = ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description="Dump LMDB entries to image files or numpy array",
        epilog="# To dump encoded JPEG images\n"
                "%(prog)s input_db output_dir --image_ext jpg --encoded\n\n"
                "# To dump 100 decoded JPEG images\n"
                "%(prog)s input_db output_dir --image_ext jpg --max_num 100\n\n"
                "# To dump 1D features of 100 items\n"
                "%(prog)s input_db output_npy --max_num 100\n\n"
                "# To dump 3D features\n"
                "%(prog)s input_db output_npy --keepdims")
    parser.add_argument(
        'input_db',
        help="Directory of the LMDB to be dumped")
    parser.add_argument(
        'output',
        help="Output directory (for images) or numpy file path (for features)")
    parser.add_argument(
        '--max_num',
        type=int,
        help="Max number of entries to be dumped")
    parser.add_argument(
        '--image_ext',
        choices=['jpg', 'png'],
        help="The extension of the images to be saved as.")
    parser.add_argument(
        '--encoded',
        action='store_true',
        help="When set, will directly save the binary as image file.")
    parser.add_argument(
        '--keepdims',
        action='store_true',
        help="When set, the dumped numpy array will keep the shape "
             "of the original datum. Otherwise it will be flattened.")
    args = parser.parse_args()
    main(args)
