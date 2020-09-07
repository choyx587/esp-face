#pragma once
#if __cplusplus
extern "C"
{
#endif
#include "image_util.h"
#include "dl_lib_matrix3d.h"
#include "dl_lib_matrix3dq.h"
#include "image_util.h"
#include "fsa.h"

#define CROP_SCALE          0.6f    /*<! Crop zoom factor, the range is 0. to 1., 0.6 is recommended */
#define INPUT_WIDTH           56    /*<! Hyper parameter, changes are not currently supported */
#define INPUT_HEIGHT          56    /*<! Hyper parameter, changes are not currently supported */
#define INPUT_CHANNEL         3     /*<! Hyper parameter, changes are n ot currently supported */
#define MEAN                  127   /*<! Hyper parameter, used to normalize the image and scale the image to (-1, 1) */

typedef struct {
    angles_t **angles;
    int len;            /*<! People number */
} head_angles_t;

/**
 * @brief Release head posture structure
 *
 * @param head_angles           Pointer to the angles of head
 */
void free_head_angles(head_angles_t *head_angles);

/**
 * @brief Head pose estimation forward propagation
 *
 * @note
 *     1. This function interface supports the presence of multiple human heads in an image
 *     2. The actual value of the image cropping ratio is crop_k * CROP_SCALE
 *
 * @param image_matrix          Pointer to input image
 * @param bboxes                Pointer to the output of face detection model(MTCNN or LSSH)
 * @param crop_k                Cropped factor. 0.0 ~ 1.0, recommended to use 0.8
 * @return                      Pointer to angle in three directions(Pitch Yaw Roll)
 */
head_angles_t *head_pose_estimation(dl_matrix3du_t *image_matrix, box_array_t *bboxes, float crop_k);

/**
 * @brief Draw axes in three directions and bounding box on the image
 *
 * @param dst_img               Pointer to input image
 * @param dst_width             Input image width
 * @param dst_height            Input image height
 * @param angles                Pointer to angles in three directions(Pitch Yaw Roll)
 * @param box                   Pointer to bounding box
 * @param size                  Length base, 10 ~ 100, recommended to use 50
 */
void draw_axes_and_bbox(uint8_t *dst_img, int dst_width, int dst_height, angles_t *angles, box_t *box, int size);

#if __cplusplus
}
#endif