#ifndef NNET_MATH_H_
#define NNET_MATH_H_

#include "ac_complex.h"
#include "ac_fixed.h"
#include <ac_math/ac_arccos_cordic.h>
#include <ac_math/ac_arcsin_cordic.h>
#include <ac_math/ac_atan2_cordic.h>
#include <ac_math/ac_hcordic.h>
#include <ac_math/ac_reciprocal_pwl.h>
#include <ac_math/ac_sincos_cordic.h>
#include <ac_math/ac_sincos_lut.h>
#include <ac_math/ac_sqrt.h>
#include <ac_math/ac_tanh_pwl.h>

namespace nnet {

// Math functions used by the SymbolicExpression layer, implemented with AC Math.
// All functions take and return the same type T (an ac_fixed), so they can be used in expressions without casting
// and as the function of a nnet::lookup_table. Internal precision is derived from the fractional bits of T.
// Functions with a restricted domain (log, sqrt, pow) expect the input to lie in that domain.

template <class T> T sin(T x) {
    ac_fixed<T::width + 4, T::i_width, true> angle_over_pi = x * ac_fixed<20, 0, false>(0.318309886183790671);
    ac_fixed<T::width - T::i_width + 2, 2, true> res;
    ac_math::ac_sin_cordic(angle_over_pi, res);
    return res;
}

template <class T> T cos(T x) {
    ac_fixed<T::width + 4, T::i_width, true> angle_over_pi = x * ac_fixed<20, 0, false>(0.318309886183790671);
    ac_fixed<T::width - T::i_width + 2, 2, true> res;
    ac_math::ac_cos_cordic(angle_over_pi, res);
    return res;
}

template <class T> T tan(T x) {
    ac_fixed<T::width - T::i_width + 2, 2, true> s = sin(x), c = cos(x);
    return s / c;
}

// ac_arcsin_cordic and ac_arccos_cordic return the angle scaled by 1/pi

template <class T> T asin(T x) {
    ac_fixed<T::width, T::i_width, true> t = x;
    ac_fixed<T::width - T::i_width + 3, 1, true> res_over_pi;
    ac_math::ac_arcsin_cordic(t, res_over_pi);
    return res_over_pi * ac_fixed<20, 2, false>(3.14159265358979323846);
}

template <class T> T acos(T x) {
    ac_fixed<T::width, T::i_width, true> t = x;
    ac_fixed<T::width - T::i_width + 2, 1, false> res_over_pi;
    ac_math::ac_arccos_cordic(t, res_over_pi);
    return res_over_pi * ac_fixed<20, 2, false>(3.14159265358979323846);
}

template <class T> T atan2(T y, T x) {
    ac_fixed<T::width, T::i_width, true> y_s = y, x_s = x;
    ac_fixed<T::width - T::i_width + 3, 3, true> res;
    ac_math::ac_atan2_cordic(y_s, x_s, res);
    return res;
}

template <class T> T atan(T x) { return atan2(x, T(1)); }

template <class T> T exp(T x) {
    ac_fixed<T::width, T::i_width, false> res;
    ac_math::ac_exp_cordic(x, res);
    return res;
}

template <class T> T log(T x) {
    ac_fixed<T::width, T::i_width, false> x_u = x;
    T res;
    ac_math::ac_log_cordic(x_u, res);
    return res;
}

template <class T> T log2(T x) {
    ac_fixed<T::width, T::i_width, false> x_u = x;
    T res;
    ac_math::ac_log2_cordic(x_u, res);
    return res;
}

template <class T> T log10(T x) { return log2(x) * ac_fixed<20, -1, false>(0.301029995663981195); }

template <class T> T sqrt(T x) {
    ac_fixed<T::width, T::i_width, false> x_u = x;
    ac_fixed<T::width, T::i_width, false> res;
    ac_math::ac_sqrt(x_u, res);
    return res;
}

template <class T> T pow(T x, T y) {
    ac_fixed<T::width, T::i_width, false> x_u = x;
    ac_fixed<T::width, T::i_width, false> res;
    ac_math::ac_pow_cordic(x_u, y, res);
    return res;
}

template <class T> T recip(T x) {
    T res;
    ac_math::ac_reciprocal_pwl(x, res);
    return res;
}

template <class T> T sinh(T x) {
    ac_fixed<T::width, T::i_width, false> ep, em;
    ac_math::ac_exp_cordic(x, ep);
    ac_math::ac_exp_cordic(-x, em);
    return (ep - em) * ac_fixed<1, 0, false>(0.5);
}

template <class T> T cosh(T x) {
    ac_fixed<T::width, T::i_width, false> ep, em;
    ac_math::ac_exp_cordic(x, ep);
    ac_math::ac_exp_cordic(-x, em);
    return (ep + em) * ac_fixed<1, 0, false>(0.5);
}

template <class T> T tanh(T x) {
    T res;
    ac_math::ac_tanh_pwl(x, res);
    return res;
}

template <class T> T abs(T x) { return x < 0 ? T(-x) : x; }

template <class T> T floor(T x) {
    ac_fixed<T::i_width, T::i_width, T::sign> res = x;
    return res;
}

template <class T> T ceil(T x) { return -floor<T>(-x); }

// LUT-based sin/cos (see ac_sincos_lut.h), accurate up to 12 fractional bits of the input.
// The input is scaled to revolutions and wrapped to [0, 1).

template <class T> T sin_lut(const T input) {
    ac_fixed<12, 0, false, AC_RND, AC_WRAP> scaled_input = input * ac_fixed<16, 0, false>(0.15915494309); // 1/(2*pi)
    ac_complex<ac_fixed<T::width - T::i_width + 2, 2, true>> res;
    ac_math::ac_sincos_lut(scaled_input, res);
    return res.i();
}

template <class T> T cos_lut(const T input) {
    ac_fixed<12, 0, false, AC_RND, AC_WRAP> scaled_input = input * ac_fixed<16, 0, false>(0.15915494309); // 1/(2*pi)
    ac_complex<ac_fixed<T::width - T::i_width + 2, 2, true>> res;
    ac_math::ac_sincos_lut(scaled_input, res);
    return res.r();
}

} // namespace nnet

#endif
