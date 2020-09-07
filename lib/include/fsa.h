#pragma once

#ifdef __cplusplus
extern "C"
{
#endif
#include "dl_lib_matrix3d.h"
#include "dl_lib_matrix3dq.h"

typedef struct {
    float pitch;
    float yaw;
    float roll;
} angles_t;

/**
 * @bried The forward propagation of head pose estimation
 *
 * @note
 *     1. The input must be a normalized image of 56*56
 *     2. The input image will be released inside the function
 *
 * @param in            Pointer to input image
 * @param mode          Implementation mode
 * @return              Pointer to three angles(Pitch, Yaw, Roll -99 degrees ~ 99 degrees)
 */
angles_t *fsa_14m(dl_matrix3dq_t *in, dl_conv_mode mode);

#ifdef __cplusplus
}
#endif
