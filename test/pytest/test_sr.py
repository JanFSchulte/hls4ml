from pathlib import Path

import numpy as np
import pytest

import hls4ml

test_root_path = Path(__file__).parent


@pytest.fixture(scope='module')
def data():
    X = 2 * np.random.rand(100, 5)
    y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5

    return X, y


@pytest.mark.parametrize('compiler', ['vivado_hls', 'catapult'])
def test_hlssr(test_case_id, data, compiler):
    expr = 'x0**2 + 2.5382*cos_lut(x3) - 0.5'

    lut_functions = {'cos_lut': {'math_func': 'cos', 'range_start': -4, 'range_end': 4, 'table_size': 2048}}

    output_dir = str(test_root_path / test_case_id)

    hls_model = hls4ml.converters.convert_from_symbolic_expression(
        expr,
        n_symbols=5,
        precision='ap_fixed<18,6>',
        output_dir=output_dir,
        lut_functions=lut_functions,
        hls_compiler=compiler,
        hls_include_path='',
        hls_libs_path='',
    )
    hls_model.write()
    hls_model.compile()

    X, y = data
    y_hls = hls_model.predict(X)
    y_hls = y_hls.reshape(y.shape)

    np.testing.assert_allclose(y, y_hls, rtol=1e-2, atol=1e-2, verbose=True)


@pytest.mark.parametrize('use_built_in_luts', [False, True])
def test_hlssr_catapult_math(test_case_id, data, use_built_in_luts):
    expr = [
        'sin(x0) + cos(x1)',
        'exp(x0) - log(x1 + 1)',
        'sqrt(x2) + 1/(x3 + 1)',
        'atan2(x4, x0 + 0.1) + tanh(x1)',
        'log(x2 + 1, 2) + x0**1.5',
    ]

    output_dir = str(test_root_path / test_case_id)

    hls_model = hls4ml.converters.convert_from_symbolic_expression(
        expr,
        n_symbols=5,
        precision='ap_fixed<18,6>',
        output_dir=output_dir,
        use_built_in_lut_functions=use_built_in_luts,
        hls_compiler='catapult',
    )
    hls_model.compile()

    X, _ = data
    y = np.stack(
        [
            np.sin(X[:, 0]) + np.cos(X[:, 1]),
            np.exp(X[:, 0]) - np.log(X[:, 1] + 1),
            np.sqrt(X[:, 2]) + 1 / (X[:, 3] + 1),
            np.arctan2(X[:, 4], X[:, 0] + 0.1) + np.tanh(X[:, 1]),
            np.log2(X[:, 2] + 1) + X[:, 0] ** 1.5,
        ],
        axis=1,
    )
    y_hls = hls_model.predict(X)

    np.testing.assert_allclose(y, y_hls, rtol=1e-2, atol=1e-2, verbose=True)


def test_hlssr_catapult_unsupported(test_case_id):
    with pytest.raises(NotImplementedError, match='erf'):
        hls4ml.converters.convert_from_symbolic_expression(
            'erf(x0)',
            n_symbols=1,
            precision='ap_fixed<18,6>',
            output_dir=str(test_root_path / test_case_id),
            hls_compiler='catapult',
        )


def test_pysr_luts(data):
    try:
        from pysr import PySRRegressor
    except ImportError:
        pytest.skip('Failed to import PySR, test will be skipped.')

    function_definitions = ['cos_lut(x) = math_lut(cos, x, N=1024, range_start=-4, range_end=4)']
    hls4ml.utils.symbolic_utils.init_pysr_lut_functions(init_defaults=True, function_definitions=function_definitions)

    model = PySRRegressor(
        model_selection='best',  # Result is mix of simplicity+accuracy
        niterations=10,
        binary_operators=['+', '*'],
        unary_operators=[
            'cos_lut',
        ],
        loss='loss(x, y) = (x - y)^2',
        temp_equation_file=True,
    )

    X, y = data

    model.fit(X, y)

    eq = str(model.sympy())

    assert 'cos_lut' in eq


@pytest.mark.parametrize('part', ['some_part', None])
@pytest.mark.parametrize('clock_period', [8, None])
@pytest.mark.parametrize('clock_unc', ['15%', None])
@pytest.mark.parametrize('compiler', ['vivado_hls', 'vitis_hls'])
def test_sr_backend_config(test_case_id, part, clock_period, clock_unc, compiler):
    expr = 'x0**2 + 2.5382*cos_lut(x3) - 0.5'

    output_dir = test_root_path / test_case_id

    hls_model = hls4ml.converters.convert_from_symbolic_expression(
        expr,
        n_symbols=5,
        precision='ap_fixed<18,6>',
        output_dir=str(output_dir),
        part=part,
        clock_period=clock_period,
        clock_uncertainty=clock_unc,
        compiler=compiler,
        hls_include_path='',
        hls_libs_path='',
    )
    hls_model.write()

    # Check if config was properly parsed into the ModelGraph

    read_part = hls_model.config.get_config_value('Part')
    expected_part = part if part is not None else 'xcvu13p-flga2577-2-e'
    assert read_part == expected_part

    read_clock_period = hls_model.config.get_config_value('ClockPeriod')
    expected_period = clock_period if clock_period is not None else 5
    assert read_clock_period == expected_period

    read_clock_unc = hls_model.config.get_config_value('ClockUncertainty')
    expected_unc = clock_unc
    if expected_unc is None:
        if compiler == 'vivado_hls' or compiler == 'vitis_hls':
            expected_unc = '12.5%'
        else:
            expected_unc = '27%'
    assert read_clock_unc == expected_unc

    # Check if Writer properly wrote tcl scripts
    part_ok = period_ok = unc_ok = False

    prj_tcl_path = output_dir / 'project.tcl'
    with open(prj_tcl_path) as f:
        for line in f.readlines():
            if 'set part' in line and expected_part in line:
                part_ok = True
            if f'set clock_period {expected_period}' in line:
                period_ok = True
            if f'set clock_uncertainty {expected_unc}' in line:
                unc_ok = True

    assert part_ok and period_ok and unc_ok


@pytest.mark.parametrize('tech', ['fpga', 'asic'])
def test_sr_catapult_config(test_case_id, tech):
    output_dir = test_root_path / test_case_id

    hls_model = hls4ml.converters.convert_from_symbolic_expression(
        'x0**2 + 2.5382*cos(x3) - 0.5',
        n_symbols=5,
        precision='ap_fixed<18,6>',
        output_dir=str(output_dir),
        hls_compiler='catapult',
        tech=tech,
        part='xcvu9p-flga2577-2-e',
        asiclibs='nangate-45nm',
        clock_period=8,
    )
    hls_model.write()

    assert hls_model.config.get_config_value('Technology') == tech

    with open(output_dir / 'build_prj.tcl') as f:
        build_prj = f.read()
    if tech == 'fpga':
        assert 'setup_xilinx_part {xcvu9p-flga2577-2-e}' in build_prj
    else:
        assert 'setup_asic_libs {nangate-45nm}' in build_prj
    assert 'set hls_clock_period 8' in build_prj

    with open(output_dir / 'firmware' / 'myproject.cpp') as f:
        assert 'nnet::cos<result_t>' in f.read()
