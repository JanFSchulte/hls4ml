==================
SymbolicExpression
==================

This backend can be used to implement expressions obtained through symbolic regression tools such as `PySR <https://github.com/MilesCranmer/PySR>`_ or `SymbolNet <https://github.com/hftsoi/SymbolNet>`_. The backend targets Vivado/Vitis HLS and relies on HLS math libraries provided with a licensed installation of these tools.

Catapult HLS
============

Passing ``hls_compiler='catapult'`` to ``convert_from_symbolic_expression`` generates a Catapult HLS project instead,
which can target FPGAs or ASICs (use ``tech='asic'`` and ``asiclibs`` to select the technology libraries). In this case
the math functions are implemented with the open-source `AC Math <https://github.com/hlslibs/ac_math>`_ library, so
Python integration (``compile()`` and ``predict()``) does not require a Catapult installation.

The following functions are supported: ``sin``, ``cos``, ``tan``, ``asin``, ``acos``, ``atan``, ``atan2``, ``sinh``,
``cosh``, ``tanh``, ``exp``, ``log``, ``sqrt``, ``abs``, ``floor``, ``ceiling``, as well as reciprocals and powers.
Trigonometric, exponential and logarithmic functions use CORDIC implementations, while the reciprocal and ``tanh`` use
piecewise-linear approximations. Expressions containing other functions raise an error. Built-in ``sin``/``cos`` lookup
tables (``use_built_in_lut_functions=True``) and user-defined lookup tables are supported as with Vivado/Vitis HLS.

.. code-block:: python

    hls_model = hls4ml.converters.convert_from_symbolic_expression(
        'x0**2 + 2.5382*cos(x3) - 0.5',
        n_symbols=5,
        precision='ap_fixed<18,6>',
        hls_compiler='catapult',
        tech='asic',
        asiclibs='nangate-45nm',
        output_dir='my-sr-catapult',
    )

*TODO expand this section*
