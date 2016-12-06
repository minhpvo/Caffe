import numpy as np
import os
import sys
import argparse
import glob
import time

import caffe


def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "input_dir",
        help="Input image directory."
    )
    parser.add_argument(
        "output_file",
        help="Output npy filename."
    )
    parser.add_argument(
        'num_jobs',
        type=int,
        help="The number of jobs run in parallel")
    parser.add_argument(
        'job_id',
        type=int,
        help="The job id of this process. Ranges from 1 to num_jobs.")
    # Optional arguments.
    parser.add_argument(
        "--model_def",
        default=os.path.join(pycaffe_dir,
                "../models/bvlc_reference_caffenet/deploy.prototxt"),
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        default=os.path.join(pycaffe_dir,
                "../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"),
        help="Trained model weights file."
    )
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--center_only",
        action='store_true',
        help="Switch for prediction from center crop alone instead of " +
             "averaging predictions across crops (default)."
    )
    parser.add_argument(
        "--images_dim",
        help="Canonical 'height,width' dimensions of input images. " +
             "None for not resize."
    )
    parser.add_argument(
        "--mean_file",
        default=os.path.join(pycaffe_dir,
                             'caffe/imagenet/ilsvrc_2012_mean.npy'),
        help="Data set image mean of [Channels x Height x Width] dimensions " +
             "(numpy array). Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=255.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."
    )
    parser.add_argument(
        "--ext",
        default='jpg',
        help="Image file extension to take as input when a directory " +
             "is given as the input file."
    )
    args = parser.parse_args()

    if args.images_dim is not None:
        image_dims = [int(s) for s in args.images_dim.split(',')]
    else:
        image_dims = None

    mean, channel_swap = None, None
    if args.mean_file:
        mean = np.load(args.mean_file)
    if args.channel_swap:
        channel_swap = [int(s) for s in args.channel_swap.split(',')]

    if args.gpu:
        caffe.set_device(args.job_id - 1)
        caffe.set_mode_gpu()
        print("GPU mode")
    else:
        caffe.set_mode_cpu()
        print("CPU mode")

    # Make classifier.
    classifier = caffe.Classifier(args.model_def, args.pretrained_model,
            image_dims=image_dims, mean=mean,
            input_scale=args.input_scale, raw_scale=args.raw_scale,
            channel_swap=channel_swap)

    # Get image files to be processed.
    net_batch_size = classifier.blobs[classifier.inputs[0]].data.shape[0]
    if args.center_only:
        im_batch_size = net_batch_size
    else:
        if net_batch_size % 10 != 0:
            raise ValueError("Batch size defined in deploy file should be "
                             "divisible by 10")
        im_batch_size = net_batch_size // 10

    files = glob.glob(os.path.join(args.input_dir, '*.' + args.ext))
    files.sort()
    num_files_per_job = (len(files) - 1) // args.num_jobs + 1
    files = files[(args.job_id - 1) * num_files_per_job:
                  min(len(files), args.job_id * num_files_per_job)]
    num_batches = (len(files) - 1) // im_batch_size + 1

    all_outputs = []
    start_time = time.time()
    for batch_id in xrange(num_batches):
        begin = batch_id * im_batch_size
        end = min(begin + im_batch_size, len(files))
        inputs = [caffe.io.load_image(f) for f in files[begin:end]]
        outputs = classifier.predict(inputs, not args.center_only)
        all_outputs.extend(outputs)
        time_elapsed = time.time() - start_time
        eta = time_elapsed / (batch_id + 1) * (num_batches - batch_id - 1)
        print "Batch {} / {}, Time {:.2f} s, ETA {:.2f} s".format(
                batch_id, num_batches, time_elapsed, eta)
    all_outputs = np.asarray(all_outputs)
    print "Done in {:.2f} s".format(time.time() - start_time)

    # Save
    dirname = os.path.dirname(args.output_file)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    np.save(args.output_file, all_outputs)
    print "Saved to", args.output_file


if __name__ == '__main__':
    main(sys.argv)
