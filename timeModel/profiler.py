# This script analyze neural network architectures.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

import click
import numpy as np

from paleo.graph import OperationGraph
from paleo import profilers
from paleo import simulation
from paleo.utils import save_layer
from paleo import comm

FORMAT = "%(levelname)s %(pathname)s:%(lineno)d] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger("paleo")
logger.setLevel(logging.INFO)


class Profiler():
    def __init__(self, filename, separator='\t'):
        """Initialize a profiler for the given network architecture."""
        self._filename = filename

        # Parse the net spec and flatten into a list in topology order.
        self.graph = OperationGraph(filename)
        logger.debug('Net spec loaded from %s.' % filename)
        logger.debug('Dependencies: %s' % str(self.graph.nested_list))
        self._separator = separator

    def print_static_summary(self):
        """Print a static summary about the network."""
        print('A summary of static characteristics of network.')
        print('  LAYER\tOUTPUTS')
        num_params = 0
        weights_in_bytes = 0
        num_activations = 0
        for layer_spec in self.graph.topology_order:
            layer = layer_spec.layer_op
            print('  %s' % layer)
            num_params += layer.num_params
            weights_in_bytes += layer.weights_in_bytes
            num_activations += np.prod(layer.outputs)
        print('Number of params: {:,} ({:,} Bytes)'.format(num_params,
                                                           weights_in_bytes))
        print('Activation: {:,} Bytes'.format(num_activations * 4))

    def save_conv_layers(self, save_dir):
        """Save convolution layers into separate files."""
        for layer_spec in self.graph.topology_order:
            if layer_spec['type'] != 'Convolution':
                continue
            layer = layer_spec.layer_op
            outfilename = os.path.join(save_dir, "%s.json" % layer_spec.name)
            save_layer.save_conv_layer(outfilename, layer)

    def profile(self, options, executor=None):
        """
        Returns:
            A dictionary contains the following keys:
              (layers, flops, executor, executor_std, flops_message,
              executor_msg)
        """

        results = []
        conv_signs = ['conv', 'incept', 'res', 'cccp']
        fc_signs = ['ip', 'fc', 'innerproduct']
        for layer_spec in self.graph.topology_order:
            if any(conv in layer_spec.name for conv in conv_signs) or any(fc in layer_spec.name for fc in fc_signs):
                layer = layer_spec.layer_op

                # Always run flop-based profiler.
                if executor == 'tensorflow':
                    # Here we disable the cudnn heuristics.
                    # Tensorflow requires creating a cuda stream and does not allow
                    # multiple context under one process.
                    # We cannot use cuda stream because of the python wrapper.
                    options.use_cudnn_heuristics = False

                flops_profiler = profilers.FlopsProfiler(options)
                flop_based_time = flops_profiler.profile(layer)

                logger.info('Layer: %s' % layer_spec.name)
                logger.info('- %s: %s  %s' % (flops_profiler.name, flop_based_time,
                                              flops_profiler.message))


                if flops_profiler:
                    executor_time = flops_profiler.profile(layer)
                    logger.info('- %s: %s  %s' % (flops_profiler.name, executor_time, flops_profiler.message))
                    results.append(
                        (layer_spec.name, flop_based_time.total_time,
                            executor_time.total_time, 0, flops_profiler.message,
                            flops_profiler.message))
                print("Runtime: ", flop_based_time, end=' ')
        return results


class BaseProfiler(object):
    """API for creating customized profilers."""

    def __init__(self, filename, separator='\t'):
        """Initialize a profiler for the given network architecture."""
        self.graph = OperationGraph(filename)


    def estimate_forward(self, batch_sizes):
        forward_times, params_in_bytes = simulation._profile_for_batch_size(
            self.graph.topology_order, 'forward', self.device_spec,
            batch_sizes, self._options['use_only_gemm'],
            self._options['ppp_comp'], self._options['ppp_comm'])

        if self._options['use_pipeline']:
            return sum([t.lowerbound for t in forward_times]), params_in_bytes

        return sum(forward_times).total_time, params_in_bytes

    def estimate_backward(self, batch_sizes):
        backward_times, _ = simulation._profile_for_batch_size(
            self.graph.topology_order, 'backward', self.device_spec,
            batch_sizes, self._options['use_only_gemm'],
            self._options['ppp_comp'], self._options['ppp_comm'])

        if self._options['use_pipeline']:
            return sum([t.lowerbound for t in backward_times])
        return sum(backward_times).total_time

    def estimate_update(self, params_in_bytes):
        time_apply_updates = simulation._profile_for_apply_updates(
            params_in_bytes, self.device_spec)
        if self._options['use_pipeline']:
            return time_apply_updates.lowerbound
        return time_apply_updates.total_time

    def estimate_comm(self, workers, params_in_bytes, scheme='TreeAllReduce'):
        comm_scheme = comm.get_comm_scheme(scheme, workers, self.network_spec,
                                           self._options['ppp_comm'])
        return comm_scheme.all_reduce(params_in_bytes)


HELP_VERBOSE = 'Whether to display debug level log messages.'
HELP_DEVICE_NAME = 'Device to estimate.'


@click.group()
@click.option('--verbose', is_flag=True, help=HELP_VERBOSE)
def cli(verbose):
    if verbose:
        logger.setLevel(logging.DEBUG)


HELP_EXECUTIOR = 'Which executor to use.'
HELP_WARMUP = 'Iterations to burn in.'
HELP_ITER = 'Iterations to run for profiling.'
HELP_EXTRACT_CONV_DIR = 'Path to extract conv layers.'


@cli.command()
@click.argument('netspec_files', nargs=-1)
@click.option('--num_warmup', default=10, help=HELP_WARMUP)
@click.option('--num_iter', default=50, help=HELP_ITER)
@click.option('--extract_conv_dir', help=HELP_EXTRACT_CONV_DIR)
@click.option('--direction', default='forward')
@click.option('--gradient_wrt', default='data')
@click.option('--use_only_gemm', is_flag=True)
@click.option('--ppp_comp', default=1.0)
@click.option('--executor')
@click.option('--separator', default='\t')
def profile(netspec_files, num_warmup, num_iter, extract_conv_dir,
            direction, gradient_wrt, use_only_gemm, executor, ppp_comp,
            separator):
    """Profiling a neural network."""

    def _print_tabular(cudnn_result, tensorflow_result):
        assert len(cudnn_result) == len(tensorflow_result)

        print(separator.join(
            ['layer', 'ours', 'cudnn', 'tensorflow', 'ours_alg', 'cu_alg']))
        sum_ours, sum_cu, sum_tf = 0, 0, 0
        for cudnn_prof, tf_prof in zip(cudnn_result, tensorflow_result):
            (layer_name, ours_time, cudnn_time, tf_time, our_msg,
             cu_msg) = ['', 0, 0, 0, '', '']
            if cudnn_prof:
                layer_name, ours_time, cudnn_time, _, our_msg, cu_msg = (
                    cudnn_prof)
            if tf_prof:
                layer_name, ours_time, tf_time, _, our_msg, _ = tf_prof

            our_msg = our_msg.replace('CUDNN_CONVOLUTION_', '')
            cu_msg = cu_msg.replace('CUDNN_CONVOLUTION_', '')

            if layer_name == 'data':
                continue

            sum_ours += ours_time
            sum_cu += cudnn_time
            sum_tf += tf_time

            print(separator.join([
                str(x)
                for x in (layer_name, ours_time, cudnn_time, tf_time, our_msg,
                          cu_msg)
            ]))
        print(separator.join(['Sum', str(sum_ours), str(sum_cu), str(sum_tf)]))

    all_results = dict()
    for netspec_file in netspec_files:
        profiler = Profiler(netspec_file, separator=separator)

        if extract_conv_dir:
            profiler.save_conv_layers(extract_conv_dir)

        if profile:
            options = profilers.ProfilerOptions()
            options.direction = direction
            options.gradient_wrt = gradient_wrt
            options.num_iter = num_iter
            options.num_warmup = num_warmup
            options.ppp_comp = ppp_comp
            #Add by Ermao
            print('Network: %s' % netspec_file)
            print('Direction: %s' % direction)
            if direction == 'backward':
                print('Gradient wrt: %s' % gradient_wrt)

            tensorflow_result, cudnn_result = None, None
            if executor == 'tensorflow':
                options.use_cudnn_heuristics = False
                tensorflow_result = profiler.profile(options, executor='tensorflow')

            if not use_only_gemm:
                options.use_cudnn_heuristics = True

            if executor == 'cudnn':
                cudnn_result = profiler.profile(options, executor='cudnn')

            if cudnn_result:
                tensorflow_result = [None] * len(cudnn_result)
            elif tensorflow_result:
                cudnn_result = [None] * len(tensorflow_result)
            all_results[netspec_file] = (cudnn_result, tensorflow_result)
    ''' #by Ermao
    for net in all_results:
        print('Network: %s' % net)
        print('Direction: %s' % direction)
        if direction == 'backward':
            print('Gradient wrt: %s' % gradient_wrt)
        (cu, tf) = all_results[net]
        _print_tabular(cu, tf)
    '''


if __name__ == '__main__':
    cli()
