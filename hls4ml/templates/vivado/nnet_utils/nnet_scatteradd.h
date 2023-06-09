#ifndef NNET_SCATTERADD_H_
#define NNET_SCATTERADD_H_

namespace nnet {

struct scatter_add_config_3d {
    static const unsigned in_x = 5;
    static const unsigned in_y = 3;
    static const unsigned in_z = 3;
    static const unsigned index_x = 5;
    static const unsigned index_y = 1;
    static const unsigned index_z = 3;
    static const unsigned src_x = 5;
    static const unsigned src_y = 2;
    static const unsigned src_z = 3;
    static const unsigned dim = 0;
};

template <class input1_T, class res_T, class input2_T, class input3_T, typename CONFIG_T>
void scatter_add_3d(input1_T target[CONFIG_T::in_z * CONFIG_T::in_y * CONFIG_T::in_x],
                    res_T result[CONFIG_T::in_z * CONFIG_T::in_y * CONFIG_T::in_x],
                    input2_T index[CONFIG_T::index_z * CONFIG_T::index_y * CONFIG_T::index_x],
                    input3_T src[CONFIG_T::src_z * CONFIG_T::src_y * CONFIG_T::src_x]) {

    // prepare tensor array with values from input tensor
    for (int i = 0; i < CONFIG_T::in_z; ++i) {
        for (int j = 0; j < CONFIG_T::in_y; ++j) {
            for (int k = 0; k < CONFIG_T::in_x; ++k) {
                result[i * CONFIG_T::in_x * CONFIG_T::in_y + j * CONFIG_T::in_x + k] =
                    target[i * CONFIG_T::in_x * CONFIG_T::in_y + j * CONFIG_T::in_x + k];
            }
        }
    }

    // perform scatter_add operation (not yet optimized)
    for (int i = 0; i < CONFIG_T::index_z; ++i) {
        for (int j = 0; j < CONFIG_T::index_y; ++j) {
            for (int k = 0; k < CONFIG_T::index_x; ++k) {
                if (CONFIG_T::dim == 0)
                    result[CONFIG_T::in_x * CONFIG_T::in_y *
                               index[i * CONFIG_T::index_x * CONFIG_T::index_y + j * CONFIG_T::index_x + k] +
                           j * CONFIG_T::in_x + k] +=
                        src[i * CONFIG_T::index_x * CONFIG_T::index_y + j * CONFIG_T::index_x + k];
                else if (CONFIG_T::dim == 1)
                    result[i * CONFIG_T::in_x * CONFIG_T::in_y +
                           j * index[i * CONFIG_T::index_x * CONFIG_T::index_y + j * CONFIG_T::index_x + k] + k] +=
                        src[i * CONFIG_T::index_x * CONFIG_T::index_y + j * CONFIG_T::index_x + k];
                else if (CONFIG_T::dim == 2)
                    result[i * CONFIG_T::in_x * CONFIG_T::in_y + j * CONFIG_T::in_x +
                           index[i * CONFIG_T::index_x * CONFIG_T::index_y + j * CONFIG_T::in_x + k]] +=
                        src[i * CONFIG_T::index_x * CONFIG_T::index_y + j * CONFIG_T::index_x + k];
            }
        }
    }
}

struct scatter_add_config_2d {
    static const unsigned in_x = 5;
    static const unsigned in_y = 3;
    static const unsigned index_x = 5;
    static const unsigned index_y = 1;
    static const unsigned src_x = 5;
    static const unsigned src_y = 2;
    static const unsigned dim = 0;
};

template <class input1_T, class res_T, class input2_T, class input3_T, typename CONFIG_T>
void scatter_add_2d(input1_T target[CONFIG_T::in_y * CONFIG_T::in_x], res_T result[CONFIG_T::in_y * CONFIG_T::in_x],
                    input2_T index[CONFIG_T::index_y * CONFIG_T::index_x], input3_T src[CONFIG_T::src_y * CONFIG_T::src_x]) {

    // prepare tensor array with values from input tensor
    for (int i = 0; i < CONFIG_T::in_y; ++i) {
        for (int j = 0; j < CONFIG_T::in_x; ++j) {
            result[i * CONFIG_T::in_x + j] = target[i * CONFIG_T::in_x + j];
        }
    }

    // perform scatter_add operation (not yet optimized)
    for (int i = 0; i < CONFIG_T::index_y; ++i) {
        for (int j = 0; j < CONFIG_T::index_x; ++j) {
            if (CONFIG_T::dim == 0)
                result[CONFIG_T::in_x * index[i * CONFIG_T::index_x + j] + j] += src[i * CONFIG_T::index_x + j];
            else if (CONFIG_T::dim == 1)
                result[i * CONFIG_T::in_x + index[i * CONFIG_T::index_x + j]] += src[i * CONFIG_T::index_x + j];
        }
    }
}

struct scatter_add_config_1d {
    static const unsigned in_x = 5;
    static const unsigned index_x = 5;
    static const unsigned src_x = 5;
    static const unsigned dim = 0;
};

template <class input1_T, class res_T, class input2_T, class input3_T, typename CONFIG_T>
void scatter_add_1d(input1_T target[CONFIG_T::in_x], res_T result[CONFIG_T::in_x], input2_T index[CONFIG_T::index_x],
                    input3_T src[CONFIG_T::src_x]) {

    // prepare tensor array with values from input tensor
    for (int i = 0; i < CONFIG_T::in_y; ++i) {
        result[i] = target[i];
    }

    // perform scatter_add operation (not yet optimized)
    for (int i = 0; i < CONFIG_T::index_y; ++i) {
        result[index[i]] += src[i];
    }
}

} // namespace nnet

#endif
