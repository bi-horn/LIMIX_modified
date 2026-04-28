import click

from limix_modified.stats._kinship import linear_kinship
from limix_modified.io.plink import _read_dosage as read_dosage
from limix_modified.io._detect import infer_filetype
import numpy as np


@click.command()
@click.pass_context
@click.argument("input-file", type=click.Path(exists=True))
@click.option(
    "--output-file",
    help="Specify the output file path.",
    default=None,
    type=click.Path(exists=False),
)
@click.option(
    "--filetype", help="Specify the file type instead of guessing.", default="guess"
)
@click.option(
    "--verbose/--quiet", "-v/-q", help="Enable or disable verbose mode.", default=True
)
def estimate_kinship(ctx, input_file, output_file, filetype, verbose):
    """Estimate a kinship matrix."""

    if filetype == "guess":
        #hier änderung da Name der Fkt nicht detect_file_type sondern infer_filetype ist
        filetype = infer_filetype(input_file)

    if verbose:
        print("Detected file type: {}".format(filetype))

    if filetype == "bgen":
        raise NotImplementedError()
        # G = limix.io.bgen._read_dosage(input_file, verbose=verbose)
    elif filetype == "bed":
        G = read_dosage(input_file, verbose=verbose)
    else:
        print("Unknown file type: %s" % input_file)

    K = linear_kinship(G, verbose=verbose)

    if output_file is None:
        output_file = input_file + ".npy"

    output_file, oft = infer_filetype(output_file)

    if oft == "npy":
        #save_kinship function not existent in npy.py so workaround
        # Save the kinship matrix as a numpy file
        np.save(output_file, K)
        if verbose:
            print("Kinship matrix saved as {}".format(output_file))
    else:
        print("Unknown output file type: %s" % output_file)
    