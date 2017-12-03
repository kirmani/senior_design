#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Sean Kirmani <sean@kirmani.io>
#
# Distributed under terms of the MIT license.
"""
TODO(kirmani): DESCRIPTION GOES HERE
"""


class MultiplicativeIntegrationLSTMCell(rnn_cell_impl.RNNCell):
    """Multiplicative Integration LSTM(Long short-term memory cell)
  recurrent network cell.

  The implementation is based on:
    https://arxiv.org/abs/1606.06630

    Yuhuai Wu, Saizheng Zhang, Ying Zhang, Yoshua Bengio, Ruslan Salakhutdinov,
    On Multiplicative Integration with Recurrent Neural Networks. NIPS, 2016.
  """

    def __init__(self,
                 num_units,
                 forget_bias=0.0,
                 bias_start=0.0,
                 alpha_start=1.0,
                 beta_start=1.0,
                 activation=math_ops.tanh,
                 reuse=None):
        """Initialize the Multiplicative Integration LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates,
        0.0 by default.
      bias_start: float. Starting value to initialize the bias, b.
        1.0 by default.
      alpha_start: float. Starting value to initialize the bias, alpha.
        1.0 by default.
      beta_start: float. Starting value to initialize the two bias,
        beta_1 and beta_2. 1.0 by default.
      activation: Activation function of the inner states.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
    """
        super(MultiplicativeIntegrationLSTMCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._bias_start = bias_start
        self._alpha_start = alpha_start
        self._beta_start = beta_start
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return rnn_cell_impl.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        """Run one step of Multiplicative Integration Long short-term memory(LSTM)."""
        c, h = state
        concat = _multiplicative_integration(
            [inputs, h], [
                self._num_units, self._num_units, self._num_units,
                self._num_units
            ],
            True,
            bias_start=self._bias_start,
            alpha_start=self._alpha_start,
            beta_start=self._beta_start)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)

        new_c = (c * math_ops.sigmoid(f + self._forget_bias) +
                 math_ops.sigmoid(i) * self._activation(j))
        new_h = self._activation(new_c) * math_ops.sigmoid(o)

        new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_h)
        return new_h, new_state


def _multiplicative_integration(args,
                                output_sizes,
                                bias,
                                bias_start=0.0,
                                alpha_start=1.0,
                                beta_start=1.0):
    """Multiplicative Integration: alpha * args[0] * W[0] + beta1 * args[1] * W[1]
            + beta1 * args[2] * W[2],
        where alpha, beta1, beta2 and W[i] are variables.

  Args:
    args: a list of 2D, batch x n, Tensors.
    output_sizes: a list of second dimension of W[i], where list[i] is int.
    bias: boolean, whether to add a bias term or not.
    bias_start: float, starting value to initialize the bias; 0 by default.
    alpha_start: float, starting value to initialize the alpha;
      1 by default.
    beta_start: float, starting value to initialize the beta1 and beta2;
      1 by default.

  Returns:
    A 2D Tensor with shape [batch x output_size] equal
      to the Multiplicative Integration above.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        raise ValueError("`args` must be list")
    if len(args) != 2:
        raise ValueError("`args` must contain 2 tensors")
    if output_sizes is None or (nest.is_sequence(output_sizes) and
                                not output_sizes):
        raise ValueError("`output_sizes` must be specified")
    if not nest.is_sequence(output_sizes):
        raise ValueError("`output_sizes` must be list")
    if not all(isinstance(x, int) for x in output_sizes):
        raise ValueError("`output_sizes` must contain integers only")

    total_output_size = sum(output_sizes)
    arg_sizes = []
    total_arg_size = 0
    shapes = [x.get_shape() for x in args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError(
                "multiplicative_ntegration is expecting 2D arguments: %s" %
                shapes)
        if shape[1].value is None:
            raise ValueError("multiplicative_ntegration expects shape[1] to be "
                             "provided for shape %s, but saw %s" % (shape,
                                                                    shape[1]))
        else:
            total_arg_size += shape[1].value
            arg_sizes.append(shape[1].value)

    dtype = [a.dtype for a in args][0]

    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
        kernel = vs.get_variable(
            "kernel", [total_arg_size, total_output_size], dtype=dtype)
        alphas = vs.get_variable(
            "alphas", [total_output_size],
            dtype=dtype,
            initializer=init_ops.constant_initializer(alpha_start, dtype=dtype))
        betas = vs.get_variable(
            "betas", [2 * total_output_size],
            dtype=dtype,
            initializer=init_ops.constant_initializer(beta_start, dtype=dtype))
        w1, w2 = array_ops.split(kernel, num_or_size_splits=arg_sizes, axis=0)
        b1, b2 = array_ops.split(betas, num_or_size_splits=2, axis=0)
        wx1, wx2 = math_ops.matmul(args[0], w1), math_ops.matmul(args[1], w2)
        res = alphas * wx1 * wx2 + b1 * wx1 + b2 * wx2

        if not bias:
            return res
        with vs.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            bias = vs.get_variable(
                "bias", [total_output_size],
                dtype=dtype,
                initializer=init_ops.constant_initializer(
                    bias_start, dtype=dtype))
    return nn_ops.bias_add(res, bias)
