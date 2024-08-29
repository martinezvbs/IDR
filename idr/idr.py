import os
import sys
import gzip
import io
import math
import argparse
import numpy as np
import matplotlib
from scipy.stats import rankdata
from collections import defaultdict, namedtuple
from statistics import mean

# Use Agg backend for matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import idr and related functions
import idr
import idr.optimization
from idr.optimization import estimate_model_params, old_estimator
from idr.utility import calc_post_membership_prbs, compute_pseudo_values

# Define namedtuple for Peak and MergedPeak
Peak = namedtuple(
    'Peak', ['chr', 'strand', 'start', 'stop', 'signal', 'summit', 'signalValue', 'pValue', 'qValue']
)

MergedPeak = namedtuple(
    'MergedPeak', ['chr', 'strand', 'start', 'stop', 'summit',
                   'merged_signal', 'signals', 'pks']
)


# Define mean function
def mean(items):
    items = list(items)
    return sum(items) / len(items)


def load_gff(fp):
    """
    chr20   GRIT    TSS     36322438        36322468        44      +       .       gene_id 'chr20_plus_36322407_36500530'; gene_name 'chr20_plus_36322407_36500530'; tss_id 'TSS_chr20_plus_36322407_36500530_pk1'; peak_cov '7,0,11,0,0,0,0,0,3,0,1,0,0,0,6,0,0,0,0,0,3,0,4,0,0,0,8,0,0,1';
    """
    grpd_peaks = defaultdict(list)
    for line in fp:
        if line.startswith("#"): continue
        if line.startswith("track"): continue
        data = line.split()
        signal = float(data[5])
        peak = Peak(data[0], data[6],
                    int(float(data[3])), int(float(data[4])),
                    signal, None,
                    None, None, None )
        grpd_peaks[(peak.chr, peak.strand)].append(peak)
    return grpd_peaks

def load_bed(fp, signal_index, peak_summit_index=None):
    grpd_peaks = defaultdict(list)
    for line in fp:
        if line.startswith("#"): continue
        if line.startswith("track"): continue
        data = line.split()
        signal = float(data[signal_index])
        if idr.ONLY_ALLOW_NON_NEGATIVE_VALUES and signal < 0:
            raise ValueError("Invalid Signal Value: {:e}".format(signal))
        if peak_summit_index is None or int(data[peak_summit_index]) == -1:
            summit = None
        else:
            summit = int(data[peak_summit_index]) + int(float(data[1]))
        assert summit is None or summit >= 0
        peak = Peak(data[0], data[5],
                    int(float(data[1])), int(float(data[2])),
                    signal, summit,
                    float(data[6]), float(data[7]), float(data[8])
                    )
        grpd_peaks[(peak.chr, peak.strand)].append(peak)
    return grpd_peaks

def correct_multi_summit_peak_IDR_values(idr_values, merged_peaks):
    assert len(idr_values) == len(merged_peaks)
    new_values = idr_values.copy()
    # find the maximum IDR value for each peak
    pk_idr_values = defaultdict(lambda: float('inf'))
    for i, pk in enumerate(merged_peaks):
        pk_idr_values[(pk.chr, pk.strand, pk.start, pk.stop)] = min(
            pk_idr_values[(pk.chr, pk.strand, pk.start, pk.stop)],
            idr_values[i]
        )
    # store the indices best peak indices, and update the values
    best_indices = []
    for i, pk in enumerate(merged_peaks):
        region = (pk.chr, pk.strand, pk.start, pk.stop)
        if new_values[i] == pk_idr_values[region]:
            best_indices.append(i)
        else:
            new_values[i] = pk_idr_values[region]
    return np.array(best_indices), new_values

def iter_merge_grpd_intervals(
        intervals, n_samples, pk_agg_fn,
        use_oracle_pks, use_nonoverlapping_peaks):
    """
    Group intervals by their source and calculate merged peak boundaries.

    Parameters:
    - intervals: A list of tuples where each tuple contains an interval and its sample_id.
    - n_samples: Number of samples.
    - pk_agg_fn: Function to aggregate peak signals (e.g., sum, mean).
    - use_oracle_pks: Boolean indicating if oracle peaks should be used.
    - use_nonoverlapping_peaks: Boolean indicating if only non-overlapping peaks should be considered.

    Yields:
    - A tuple containing the merged peak information.
    """
    # Initialize grouped peaks
    grpd_peaks = OrderedDict((i + 1, []) for i in range(n_samples))
    pk_start, pk_stop = float('inf'), float('-inf')

    for interval, sample_id in intervals:
        if (not use_oracle_pks) or sample_id == 0:
            pk_start = min(interval.start, pk_start)
            pk_stop = max(interval.stop, pk_stop)

        if sample_id > 0:
            grpd_peaks[sample_id].append(interval)

    # If no peaks are identified, return None
    if pk_stop == float('-inf'):
        return

    # Skip regions that don't have a peak in all replicates if needed
    if not use_nonoverlapping_peaks:
        if any(len(peaks) == 0 for peaks in grpd_peaks.values()):
            return

    # Find the merged peak summit
    replicate_summits = []
    for sample_id, pks in grpd_peaks.items():
        if use_oracle_pks and sample_id != 0:
            continue

        # Initialize summit to the first peak
        try:
            replicate_summit = pks[0].summit
            summit_signal = pks[0].signal
        except IndexError:
            replicate_summit = None
            summit_signal = -1e9

        # Find the summit with the highest signal value
        for pk in pks[1:]:
            if pk.summit is not None and pk.signal > summit_signal:
                summit_signal = pk.signal
                replicate_summit = pk.summit

        if replicate_summit is not None:
            replicate_summits.append(replicate_summit)

    summit = int(mean(replicate_summits)) if replicate_summits else None

    # Aggregate signals
    signals = [pk_agg_fn(pk.signal for pk in pks) if pks else 0 for pks in grpd_peaks.values()]
    merged_pk = (pk_start, pk_stop, summit, pk_agg_fn(signals), signals, grpd_peaks)

    yield merged_pk

def iter_matched_oracle_pks(
        pks, n_samples, pk_agg_fn, use_nonoverlapping_peaks=False):
    """Match each oracle peak to its nearest replicate peaks."""
    oracle_pks = [pk for pk, sample_id in pks if sample_id == 0]
    # if there are zero oracle peaks in this
    if len(oracle_pks) == 0:
        return None
    # for each oracle peak, find and score the replicate peaks
    for oracle_pk in oracle_pks:
        pass

    return

def merge_peaks_in_contig(all_s_peaks, pk_agg_fn, oracle_pks=None,
                      use_nonoverlapping_peaks=False, use_oracle_pks=False):
    # merge and sort all peaks, keeping track of which sample they originated in
    oracle_pks_iter = []
    if oracle_pks is not None:
        pass
    # merge and sort all the intervals, keeping track of their source
    all_intervals = []
    for sample_id, peaks in enumerate([oracle_pks_iter,] + all_s_peaks):
        for pk in peaks:
            all_intervals.append((pk, sample_id))
    all_intervals.sort()
    # grp overlapping intervals. Since they're already sorted, all we need
    # to do is check if the current interval overlaps the previous interval
    grpd_intervals = [[],]
    curr_start, curr_stop = all_intervals[0][:2]
    for pk, sample_id in all_intervals:
        if pk.start > curr_stop:
            grpd_intervals.append([])
            curr_start, curr_stop = pk.start, pk.stop
        else:
            curr_stop = max(curr_stop, pk.stop)
        grpd_intervals[-1].append((pk, sample_id))
    # build the unified peak list, setting the score to
    # zero if it doesn't exist in both replicates
    merged_pks = []
    if oracle_pks is None:
        for intervals in grpd_intervals:
            merged_pk = next(iter_merge_grpd_intervals(
                intervals, len(all_s_peaks), pk_agg_fn,
                use_oracle_pks, use_nonoverlapping_peaks))
            if merged_pk is not None:
                merged_pks.append(merged_pk)
    else:
        for intervals in grpd_intervals:
            merged_pk = next(iter_merge_grpd_intervals(
                intervals, len(all_s_peaks), pk_agg_fn,
                use_oracle_pks, use_nonoverlapping_peaks))
            if merged_pk is not None:
                merged_pks.append(merged_pk)
    return merged_pks

def write_results_to_file(merged_peaks, output_file,
                          output_file_type, signal_type,
                          max_allowed_idr=1.0,
                          soft_max_allowed_idr=1.0,
                          localIDRs=None, IDRs=None,
                          useBackwardsCompatibleOutput=False):
    if useBackwardsCompatibleOutput:
        [...]
    else:
        [...]

    # write out the result
    idr.log("Writing results to file", "VERBOSE")

    if localIDRs is None or IDRs is None:
        [...]

    num_peaks_passing_hard_thresh = 0
    num_peaks_passing_soft_thresh = 0
    for localIDR, IDR, merged_peak in zip(localIDRs, IDRs, merged_peaks):
        if max_allowed_idr is not None and IDR > max_allowed_idr:
            continue
        num_peaks_passing_hard_thresh += 1
        if soft_max_allowed_idr is not None and IDR > soft_max_allowed_idr:
            continue
        num_peaks_passing_soft_thresh += 1

    if len(merged_peaks) == 0:
        return

    idr.log("Number of peaks passing hard threshold: {}".format(num_peaks_passing_hard_thresh), "VERBOSE")
    idr.log("Number of peaks passing soft threshold: {}".format(num_peaks_passing_soft_thresh), "VERBOSE")

    return

def merge_peaks(all_s_peaks, pk_agg_fn, oracle_pks=None,
                use_nonoverlapping_peaks=False):
    """Merge peaks over all contig/strands

    """
    # if we have reference peaks, use its contigs: otherwise use
    # the union of the replicates contigs
    if oracle_pks is not None:
        contigs = set(pk.chr for pk in oracle_pks)
    else:
        contigs = set(pk.chr for peaks in all_s_peaks for pk in peaks)

    merged_peaks = []
    for key in contigs:
        for strand in ('+', '-'):
            merged_peaks.extend(
                merge_peaks_in_contig(
                    [peaks for peaks in all_s_peaks
                     if peaks[0].chr == key and peaks[0].strand == strand],
                    pk_agg_fn, oracle_pks, use_nonoverlapping_peaks))

    merged_peaks.sort(key=lambda x:x.merged_signal, reverse=True)
    return merged_peaks

def build_rank_vectors(merged_peaks):
    # allocate memory for the ranks vector
    s1 = np.zeros(len(merged_peaks))
    s2 = np.zeros(len(merged_peaks))
    # add the signal
    for i, x in enumerate(merged_peaks):
        s1[i], s2[i] = x.signals

    rank1 = np.lexsort((np.random.random(len(s1)), s1)).argsort()
    rank2 = np.lexsort((np.random.random(len(s2)), s2)).argsort()

    return ( np.array(rank1, dtype=np.int),
             np.array(rank2, dtype=np.int) )

def build_idr_output_line_with_bed6(
        m_pk, IDR, localIDR, output_file_type, signal_type,
        use_oracle_peak_values=True, outputFormat=None):
    # initialize the line with the bed6 entries - these are
    # present in all the output types
    rv = [m_pk.chr, str(m_pk.start), str(m_pk.stop),
          ".", "%i" % (min(1000, int(-125*math.log2(IDR+1e-12)))), m_pk.strand]
    if output_file_type == 'bed':
        # if we just want a bed, there's nothing else to be done
        pass
    # for narrow/broad peak files, we need to add the 3 score fields
    elif output_file_type in ('narrowPeak', 'broadPeak'):
        # if we want to use the oracle peak values for the scores, and an oracle
        # peak is specified
        if use_oracle_peak_values and 0 in m_pk.pks:
            signal_values = [m_pk.pks[0].signalValue,
                             m_pk.pks[0].pValue,
                             m_pk.pks[0].qValue]
        else:
            signal_values = [m_pk.merged_signal,
                             m_pk.pks[0].pValue,
                             m_pk.pks[0].qValue]
        rv.extend(signal_values)
        # if this is a narrow peak, we also need to add the summit
        if output_file_type == 'narrowPeak':
            rv.append(str(m_pk.summit))
    else:
        raise ValueError("Unrecognized output format '{}'".format(outputFormat))

    rv.append("%f" % -math.log10(max(1e-5, localIDR)))
    rv.append("%f" % -math.log10(max(1e-5, IDR)))

    for key, signal in enumerate(m_pk.signals):
        # we add one to the key because key=0 corresponds to the oracle peaks
        key += 1
        # if there is no matching peak for this replicate
        if m_pk.pks[key] is None:
            rv.append("0")
        else:
            rv.append("%.5f" % signal)


    return "\t".join(rv)

def build_backwards_compatible_idr_output_line(
        m_pk, IDR, localIDR, output_file_type, signal_type):
    rv = [m_pk.chr,]
    for key, signal in enumerate(m_pk.signals):
        rv.append( "%i" % min(x.start for x in m_pk.pks[key+1]))
        rv.append( "%i" % max(x.stop for x in m_pk.pks[key+1]))
        rv.append( "." )
        rv.append( "%.5f" % signal )

    rv.append("%.5f" % localIDR)
    rv.append("%.5f" % IDR)
    rv.append(m_pk.strand)

    return "\t".join(rv)

def calc_local_IDR(theta, r1, r2):
    """
    idr <- 1 - e.z
    o <- order(idr)
    idr.o <- idr[o]
    idr.rank <- rank(idr.o, ties.method = "max")
    top.mean <- function(index, x) {
        mean(x[1:index])
    }
    IDR.o <- sapply(idr.rank, top.mean, idr.o)
    IDR <- idr
    IDR[o] <- IDR.o
    """
    mu, sigma, rho, p = theta
    z1 = compute_pseudo_values(r1, mu, sigma, p, EPS=1e-12)
    z2 = compute_pseudo_values(r2, mu, sigma, p, EPS=1e-12)
    localIDR = 1 - calc_post_membership_prbs(np.array(theta), z1, z2)
    if idr.FILTER_PEAKS_BELOW_NOISE_MEAN:
        localIDR[z1 + z2 < 0] = 1

    # it doesn't make sense for the IDR values to be smaller than the
    # optimization tolerance
    localIDR = np.clip(localIDR, idr.CONVERGENCE_EPS_DEFAULT, 1)
    return localIDR

def calc_global_IDR(localIDR):
    local_idr_order = localIDR.argsort()
    ordered_local_idr = localIDR[local_idr_order]
    ordered_local_idr_ranks = rankdata( ordered_local_idr, method='max' )
    IDR = []
    for i, rank in enumerate(ordered_local_idr_ranks):
        IDR.append(ordered_local_idr[:rank].mean())
    IDR = np.array(IDR)[local_idr_order.argsort()]
    return IDR

def fit_model_and_calc_local_idr(r1, r2,
                                 starting_point=None,
                                 max_iter=idr.MAX_ITER_DEFAULT,
                                 convergence_eps=idr.CONVERGENCE_EPS_DEFAULT,
                                 fix_mu=False, fix_sigma=False):
    """
    Fit the model to the provided data and calculate local IDR values.

    Parameters:
    - r1: First set of ranks.
    - r2: Second set of ranks.
    - starting_point: Tuple of initial parameter values (mu, sigma, rho, mix_param).
    - max_iter: Maximum number of iterations for the fitting process.
    - convergence_eps: Convergence criterion for the fitting process.
    - fix_mu: Boolean to fix the mu parameter during fitting.
    - fix_sigma: Boolean to fix the sigma parameter during fitting.

    Returns:
    - localIDRs: Array of local IDR values.
    """
    # Set default starting point if none is provided
    if starting_point is None:
        starting_point = (idr.DEFAULT_MU, idr.DEFAULT_SIGMA,
                          idr.DEFAULT_RHO, idr.DEFAULT_MIX_PARAM)

    idr.log("Initial parameter values: [%s]" % " ".join(
        "%.2f" % x for x in starting_point))

    # Fit the model parameters
    idr.log("Fitting the model parameters", 'VERBOSE')

    if idr.PROFILE:
        import cProfile
        cProfile.runctx(
            """
            theta, loss = estimate_model_params(
                r1, r2, starting_point,
                max_iter=max_iter,
                convergence_eps=convergence_eps,
                fix_mu=fix_mu, fix_sigma=fix_sigma)
            """,
            globals(), locals()
        )
        # Exiting after profiling; remove the assertion if you want to continue execution
        assert False

    # Estimate model parameters
    theta, loss = estimate_model_params(
        r1, r2,
        starting_point,
        max_iter=max_iter,
        convergence_eps=convergence_eps,
        fix_mu=fix_mu, fix_sigma=fix_sigma
    )

    idr.log("Finished running IDR on the datasets", 'VERBOSE')
    idr.log("Final parameter values: [%s]" % " ".join("%.2f" % x for x in theta))

    # Calculate the global IDR
    localIDRs = calc_local_IDR(np.array(theta), r1, r2)

    return localIDRs


def write_results_to_file(merged_peaks, output_file,
                          output_file_type, signal_type,
                          max_allowed_idr=1.0,
                          soft_max_allowed_idr=1.0,
                          localIDRs=None, IDRs=None,
                          useBackwardsCompatibleOutput=False):
    if useBackwardsCompatibleOutput:
        build_idr_output_line = build_backwards_compatible_idr_output_line
    else:
        build_idr_output_line = build_idr_output_line_with_bed6

    # write out the result
    idr.log("Writing results to file", "VERBOSE")

    if localIDRs is None or IDRs is None:
        assert IDRs is None
        assert localIDRs is None
        localIDRs = np.ones(len(merged_peaks))
        IDRs = np.ones(len(merged_peaks))

    num_peaks_passing_hard_thresh = 0
    num_peaks_passing_soft_thresh = 0
    for localIDR, IDR, merged_peak in zip(localIDRs, IDRs, merged_peaks):
        # skip peaks with global idr values below the threshold
        if max_allowed_idr is not None and IDR > max_allowed_idr:
            continue
        num_peaks_passing_hard_thresh += 1
        if IDR <= soft_max_allowed_idr:
            num_peaks_passing_soft_thresh += 1
        opline = build_idr_output_line(
            merged_peak, IDR, localIDR, output_file_type, signal_type)
        print(opline, file=output_file)

    if len(merged_peaks) == 0:
        return

    idr.log(
        "Number of reported peaks - {}/{} ({:.1f}%)\n".format(
            num_peaks_passing_hard_thresh, len(merged_peaks),
            100 * float(num_peaks_passing_hard_thresh) / len(merged_peaks))
    )

    idr.log(
        "Number of peaks passing IDR cutoff of {} - {}/{} ({:.1f}%)\n".format(
            soft_max_allowed_idr,
            num_peaks_passing_soft_thresh, len(merged_peaks),
            100 * float(num_peaks_passing_soft_thresh) / len(merged_peaks))
    )

    return

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=f"""
Program: IDR (Irreproducible Discovery Rate)
Version: {idr.__version__}
Contact: Nathan Boley <npboley@gmail.com>
"""
    )

    def possibly_gzipped_file(fname):
        if fname.endswith(".gz"):
            # Add handling for gzipped files
            return fname
        else:
            # Handle non-gzipped files
            return fname

    parser.add_argument('--samples', '-s', type=possibly_gzipped_file, nargs=2,
                        required=True, help='Files containing peaks and scores.')
    parser.add_argument('--peak-list', '-p', type=possibly_gzipped_file,
                        help='If provided, all peaks will be taken from this file.')
    parser.add_argument('--input-file-type', default='narrowPeak',
                        choices=['narrowPeak', 'broadPeak', 'bed', 'gff'],
                        help='File type of --samples and --peak-list.')
    parser.add_argument('--rank',
                        help="Which column to use to rank peaks.")
    parser.add_argument('--output-file', "-o", default="idrValues.txt",
                        help='File to write output to. Default: %(default)s')
    parser.add_argument('--output-file-type',
                        choices=['narrowPeak', 'broadPeak', 'bed'],
                        help='Output file type. Defaults to input file type when available, otherwise bed.')
    parser.add_argument('--log-output-file', "-l", type=argparse.FileType("w"),
                        default=sys.stderr,
                        help='File to write output to. Default: stderr')
    parser.add_argument('--idr-threshold', "-i", type=float,
                        help="Only return peaks with a global IDR threshold below this value.")
    parser.add_argument('--soft-idr-threshold', type=float,
                        help="Report statistics for peaks with a global IDR below this value.")
    parser.add_argument('--use-old-output-format', action='store_true',
                        help="Use old output format.")
    parser.add_argument('--plot', action='store_true',
                        help='Plot the results to [OFNAME].png')
    parser.add_argument('--use-nonoverlapping-peaks', action="store_true",
                        help='Use peaks without an overlapping match and set the value to 0.')
    parser.add_argument('--peak-merge-method',
                        choices=["sum", "avg", "min", "max"],
                        help="Which method to use for merging peaks.")
    parser.add_argument('--initial-mu', type=float, default=idr.DEFAULT_MU,
                        help=f"Initial value of mu. Default: {idr.DEFAULT_MU:.2f}")
    parser.add_argument('--initial-sigma', type=float, default=idr.DEFAULT_SIGMA,
                        help=f"Initial value of sigma. Default: {idr.DEFAULT_SIGMA:.2f}")
    parser.add_argument('--initial-rho', type=float, default=idr.DEFAULT_RHO,
                        help=f"Initial value of rho. Default: {idr.DEFAULT_RHO:.2f}")
    parser.add_argument('--initial-mix-param', type=float, default=idr.DEFAULT_MIX_PARAM,
                        help=f"Initial value of the mixture params. Default: {idr.DEFAULT_MIX_PARAM:.2f}")
    parser.add_argument('--fix-mu', action='store_true',
                        help="Fix mu to the starting point and do not let it vary.")
    parser.add_argument('--fix-sigma', action='store_true',
                        help="Fix sigma to the starting point and do not let it vary.")
    parser.add_argument('--dont-filter-peaks-below-noise-mean', action='store_true',
                        help="Allow signal points that are below the noise mean (should only be used if you know what you are doing).")
    parser.add_argument('--use-best-multisummit-IDR', action='store_true',
                        help="Set the IDR value for a group of multi summit peaks to the best value across all of these peaks.")
    parser.add_argument('--allow-negative-scores', action='store_true',
                        help="Allow negative values for scores. (should only be used if you know what you are doing)")
    parser.add_argument('--random-seed', type=int, default=0,
                        help="The random seed value (for breaking ties). Default: %(default)s")
    parser.add_argument('--max-iter', type=int, default=idr.MAX_ITER_DEFAULT,
                        help=f"The maximum number of optimization iterations. Default: {idr.MAX_ITER_DEFAULT}")
    parser.add_argument('--convergence-eps', type=float, default=idr.CONVERGENCE_EPS_DEFAULT,
                        help=f"The maximum change in parameter value changes for convergence. Default: {idr.CONVERGENCE_EPS_DEFAULT:.2e}")
    parser.add_argument('--only-merge-peaks', action='store_true',
                        help="Only return the merged peak list.")
    parser.add_argument('--verbose', action="store_true",
                        help="Print out additional debug information")
    parser.add_argument('--quiet', action="store_true",
                        help="Don't print any status messages")
    parser.add_argument('--version', action='version', version=f'IDR {idr.__version__}')

    args = parser.parse_args()

    # Handle output file and logging
    args.output_file = open(args.output_file, "w")
    idr.log_ofp = args.log_output_file

    # Determine output file type
    if args.output_file_type is None:
        args.output_file_type = args.input_file_type if args.input_file_type in ('narrowPeak', 'broadPeak', 'bed') else 'bed'

    # Set verbosity and other flags
    idr.VERBOSE = args.verbose
    idr.QUIET = args.quiet
    if args.quiet:
        idr.VERBOSE = False

    idr.FILTER_PEAKS_BELOW_NOISE_MEAN = not args.dont_filter_peaks_below_noise_mean
    idr.ONLY_ALLOW_NON_NEGATIVE_VALUES = not args.allow_negative_scores

    # Set IDR thresholds
    if args.idr_threshold is None:
        args.idr_threshold = idr.DEFAULT_IDR_THRESH
    if args.soft_idr_threshold is None:
        args.soft_idr_threshold = args.idr_threshold

    np.random.seed(args.random_seed)

    # Handle plotting
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            # Add plotting logic here
        except ImportError:
            print("matplotlib is not installed. Please install it to enable plotting.")

    return args

if __name__ == "__main__":
    args = parse_args()

def load_samples(args):
    from statistics import mean

    idr.log("Loading the peak files", 'VERBOSE')

    # Decide what aggregation function to use for peaks that need to be merged
    if args.input_file_type in ['narrowPeak', 'broadPeak']:
        if args.rank is None:
            signal_type = 'signal.value'
        else:
            signal_type = args.rank

        try:
            signal_index = {'signal.value': 6, 'p.value': 7, 'q.value': 8}[signal_type]
        except KeyError:
            signal_index = int(signal_type)

        if args.peak_merge_method:
            peak_merge_fn = {'sum': sum, 'avg': mean, 'min': min, 'max': max}[args.peak_merge_method]
        elif signal_index in (4, 6):
            peak_merge_fn = sum
        else:
            peak_merge_fn = min

        summit_index = 9 if args.input_file_type == 'narrowPeak' else None
        f1, f2 = [load_bed(fp, signal_index, summit_index) for fp in args.samples]
        oracle_pks = None if args.peak_list is None else load_bed(args.peak_list, signal_index, summit_index)

    elif args.input_file_type == 'bed':
        if args.rank is None:
            signal_type = 'score'
        else:
            signal_type = args.rank

        signal_index = 4 if signal_type == 'score' else int(signal_type)

        if args.peak_merge_method:
            peak_merge_fn = {'sum': sum, 'avg': mean, 'min': min, 'max': max}[args.peak_merge_method]
        else:
            peak_merge_fn = sum

        f1, f2 = [load_bed(fp, signal_index) for fp in args.samples]
        oracle_pks = None if args.peak_list is None else load_bed(args.peak_list, signal_index)

    elif args.input_file_type == 'gff':
        if args.rank is None:
            signal_type = 'score'
        else:
            signal_type = args.rank

        if args.peak_merge_method:
            peak_merge_fn = {'sum': sum, 'avg': mean, 'min': min, 'max': max}[args.peak_merge_method]
        else:
            peak_merge_fn = sum

        f1, f2 = [load_gff(fp) for fp in args.samples]
        oracle_pks = None if args.peak_list is None else load_gff(args.peak_list)

    else:
        raise ValueError(f"Unrecognized file type: '{args.input_file_type}'")

    # Build a unified peak set
    idr.log("Merging peaks", 'VERBOSE')
    merged_peaks = merge_peaks([f1, f2], peak_merge_fn, oracle_pks, args.use_nonoverlapping_peaks)

    return merged_peaks, signal_type


def plot(args, scores, ranks, IDRs, ofprefix=None):
    assert len(args.samples) == 2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if ofprefix is None:
        ofprefix = args.output_file.name

    colors = np.zeros(len(ranks[0]), dtype=str)
    colors[:] = 'k'
    colors[IDRs > args.soft_idr_threshold] = 'r'

    #matplotlib.rc('font', family='normal', weight='bold', size=10)

    fig = plt.figure(num=None, figsize=(12, 12))

    plt.subplot(221)
    plt.axis([0, 1, 0, 1])
    plt.xlabel("Sample 1 Rank")
    plt.ylabel("Sample 2 Rank")
    plt.title("Ranks - (red >= %.2f IDR)" % args.soft_idr_threshold)
    plt.scatter((ranks[0]+1)/float(max(ranks[0])+1),
                (ranks[1]+1)/float(max(ranks[1])+1),
                edgecolor=colors,
                c=colors,
                alpha=0.05)

    plt.subplot(222)
    plt.xlabel("Sample 1 log10 Score")
    plt.ylabel("Sample 2 log10 Score")
    plt.title("Log10 Scores - (red >= %.2f IDR)" % args.soft_idr_threshold)
    plt.scatter(np.log10(scores[0]+1),
                np.log10(scores[1]+1),
                edgecolor=colors,
                c=colors,
                alpha=0.05)

    def make_boxplots(sample_i):
        groups = defaultdict(list)
        norm_ranks = (ranks[sample_i]+1)/float(max(ranks[sample_i])+1)
        for rank, idr_val in zip(norm_ranks, -np.log10(IDRs)):
            groups[int(20*rank)].append(float(idr_val))
        group_labels = sorted((x + 2.5)/20 for x in groups.keys())
        groups = [x[1] for x in sorted(groups.items())]

        plt.title("Sample %i Ranks vs IDR Values" % (sample_i+1))
        plt.axis([0, 21, 0, 0.5-math.log10(idr.CONVERGENCE_EPS_DEFAULT)])
        plt.xlabel("Sample %i Peak Rank" % (sample_i+1))
        plt.ylabel("-log10 IDR")
        plt.xticks([], [])

        plt.boxplot(groups, sym="")

        plt.axis([0, 21, 0, 0.5-math.log10(idr.CONVERGENCE_EPS_DEFAULT)])
        plt.scatter(20*norm_ranks, -np.log10(IDRs), alpha=0.01, c='black')

    plt.subplot(223)
    make_boxplots(0)

    plt.subplot(224)
    make_boxplots(1)

    plt.savefig(ofprefix + ".png")
    return


def correct_multi_summit_peak_idr_values(idr_values, merged_peaks):
    assert len(idr_values) == len(merged_peaks)

    # Initialize a dictionary to store the minimum IDR value for each peak
    pk_idr_values = defaultdict(lambda: float('inf'))

    # Update the dictionary with the minimum IDR value for each peak
    for i, pk in enumerate(merged_peaks):
        pk_idr = idr_values[i]
        key = (pk.chr, pk.start, pk.stop)
        pk_idr_values[key] = min(pk_idr_values[key], pk_idr)

    # Find the indices of the best peaks and update the IDR values
    best_indices = []
    new_values = np.zeros(len(idr_values))
    for i, pk in enumerate(merged_peaks):
        key = (pk.chr, pk.start, pk.stop)
        if idr_values[i] == pk_idr_values[key]:
            best_indices.append(i)
        new_values[i] = pk_idr_values[key]

    return np.array(best_indices), new_values

def calc_global_idr(local_idr):
    local_idr_order = local_idr.argsort()
    ordered_local_idr = local_idr[local_idr_order]
    ordered_local_idr_ranks = rankdata(ordered_local_idr, method='max')

    idr = []
    for rank in ordered_local_idr_ranks:
        idr.append(ordered_local_idr[:int(rank)].mean())

    idr = np.array(idr)[local_idr_order.argsort()]
    return idr

def main():
    args = parse_args()

    # Load and merge peaks
    merged_peaks, signal_type = load_samples(args)
    s1 = np.array([pk.signals[0] for pk in merged_peaks])
    s2 = np.array([pk.signals[1] for pk in merged_peaks])

    # Build the ranks vector
    idr.log("Ranking peaks", "VERBOSE")
    r1, r2 = build_rank_vectors(merged_peaks)

    if args.only_merge_peaks:
        local_idrs, idrs = None, None
    else:
        if len(merged_peaks) < 20:
            idr.log("Not enough peaks to perform IDR analysis", "VERBOSE")
            local_idrs, idrs = np.ones(len(merged_peaks)), np.ones(len(merged_peaks))
        else:
            local_idrs = fit_model_and_calc_local_idr(
                r1, r2,
                starting_point=(args.initial_mu, args.initial_sigma, args.initial_rho, args.initial_mix_param),
                max_iter=args.max_iter,
                convergence_eps=args.convergence_eps,
                fix_mu=args.fix_mu,
                fix_sigma=args.fix_sigma
            )

            # Apply multi-summit IDR correction if requested
            if args.use_best_multisummit_idr:
                best_indices, local_idrs = correct_multi_summit_peak_idr_values(local_idrs, merged_peaks)
                merged_peaks = [merged_peaks[i] for i in best_indices]
                r1, r2 = build_rank_vectors(merged_peaks)
                idrs = calc_global_idr(local_idrs)
            else:
                idrs = calc_global_idr(local_idrs)

            if args.plot:
                plot(args, [s1, s2], [r1, r2], idrs)

    num_peaks_passing_thresh = write_results_to_file(
        merged_peaks,
        args.output_file,
        args.output_file_type,
        signal_type,
        local_idrs=local_idrs,
        idrs=idrs,
        max_allowed_idr=args.idr_threshold,
        soft_max_allowed_idr=args.soft_idr_threshold,
        use_backwards_compatible_output=args.use_old_output_format
    )

    args.output_file.close()

if __name__ == '__main__':
    try:
        main()
    finally:
        if idr.log_ofp != sys.stderr:
            idr.log_ofp.close()