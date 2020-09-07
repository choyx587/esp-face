#include "hpe_forward.h"
#include "esp_log.h"
#include "esp_image.hpp"

static const char *TAG = "HPE Forward";

void free_head_angles(head_angles_t *head_angles)
{
    angles_t **angles = head_angles->angles;
    for (int i = 0; i < head_angles->len; i++)
    {
        dl_lib_free(*angles);
        angles++;
    }
    dl_lib_free(head_angles->angles);
    dl_lib_free(head_angles);
}

static void random_crop_with_k(
    /*
     * Use zoom factor crop_k to crop the image
     * */
    box_t *box,
    int src_width,
    int src_height,
    int *dst_x_min,
    int *dst_y_min,
    int *dst_x_max,
    int *dst_y_max,
    float crop_k)
{
    float _x_min = box->box_p[0];
    float _y_min = box->box_p[1];
    float _x_max = box->box_p[2];
    float _y_max = box->box_p[3];
    // Clip
    _x_min = CLIP(_x_min, (float)src_width - 1, 0.f);
    _y_min = CLIP(_y_min, (float)src_height - 1, 0.f);
    _x_max = CLIP(_x_max, (float)src_width - 1, 0.f);
    _y_max = CLIP(_y_max, (float)src_height - 1, 0.f);
    float dx = fabsf(_x_max - _x_min);
    float dy = fabsf(_y_max - _y_min);
    // Compute the new cords
    *dst_x_min = (int)fmaxf(_x_min - crop_k * dx, 0.f);
    *dst_y_min = (int)fmaxf(_y_min - crop_k * dy, 0.f);
    *dst_x_max = (int)fminf(_x_max + crop_k * dx, (float)src_width - 1);
    *dst_y_max = (int)fminf(_y_max + crop_k * dy, (float)src_height - 1);
}

#ifdef CONFIG_XTENSA_IMPL
#define CONV_MODE DL_XTENSA_IMPL
#else
#define CONV_MODE DL_C_IMPL
#endif

head_angles_t *head_pose_estimation(dl_matrix3du_t *image_matrix, box_array_t *bboxes, float crop_k)
{
    float random_k = crop_k * CROP_SCALE;
    head_angles_t *head_angles = (head_angles_t *)dl_lib_calloc(1, sizeof(head_angles_t), 0);
    head_angles->angles = (angles_t **)dl_lib_calloc(1, bboxes->len * sizeof(angles_t *), 0);
    head_angles->len = bboxes->len;

    dl_matrix3dq_t *img_input;
    int x_min, y_min, x_max, y_max;
    box_t *box = bboxes->box;
    angles_t **angles = head_angles->angles;
    for (int i = 0; i < head_angles->len; i++)  // Support multiple people
    {
        // Calculate the coordinates of the upper left and lower right corners
        random_crop_with_k(box, image_matrix->w, image_matrix->h, &x_min, &y_min, &x_max, &y_max, random_k);
        img_input = dl_matrix3dq_alloc(1, INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNEL, -7);
        // Crop, zoom, and normalize input image
        Image<qtp_t>::crop_and_resize_to_rgb888(
            img_input->item,
            image_matrix->item,
            x_min, y_min, x_max, y_max,
            image_matrix->w,
            INPUT_WIDTH, INPUT_HEIGHT,
            MEAN);
        // Forward propagation
        *angles++ = fsa_14m(img_input, CONV_MODE);
        box++;
    }
    return head_angles;
}

void draw_axes_and_bbox(uint8_t *dst_img, int dst_width, int dst_height, angles_t *angles, box_t *box, int size)
{
    size_t line_size = 1;
    uint8_t color_red[3] = {0, 0, 255};
    uint8_t color_green[3] = {0, 255, 0};
    uint8_t color_blue[3] = {255, 0, 0};
    float pi = acosf(-1.f);
    float pitch = angles->pitch;
    float yaw = angles->yaw;
    float roll = angles->roll;
    float p = pitch * pi / 180.f;
    float y = (-yaw) * pi / 180.f;
    float r = roll * pi / 180.f;
    // Compute (tdx, tdy)
    float tdx = (box->box_p[0] + box->box_p[2]) / 2.f;
    float tdy = (box->box_p[1] + box->box_p[3]) / 2.f;
    // X-Axis | draw in red
    float x1 =  (float)size * cosf(y) * cosf(r) + tdx;
    float y1 =  (float)size * (cosf(p) * sinf(r) + cosf(r) * sinf(p) * sinf(y)) + tdy;
    img_rgb888_draw_line(dst_img, dst_width, dst_height, (int)tdx, (int)tdy, (int)x1, (int)y1, line_size, color_red);
    // Y-Axis | drawn in green
    float x2 = (float)size * (-cosf(y) * sinf(r)) + tdx;
    float y2 = (float)size * (cosf(p) * cosf(r) - sinf(p) * sinf(y) * sinf(r)) + tdy;
    img_rgb888_draw_line(dst_img, dst_width, dst_height, (int)tdx, (int)tdy, (int)x2, (int)y2, line_size, color_green);
    // Z-Axis (out of the screen) | drawn in blue
    float x3 = (float)size * (sinf(y)) + tdx;
    float y3 = (float)size * (-cosf(y) * sinf(p)) + tdy;
    img_rgb888_draw_line(dst_img, dst_width, dst_height, (int)tdx, (int)tdy, (int)x3, (int)y3, line_size, color_blue);
    // Draw bounding box | drawn in red
    img_rgb888_draw_line(dst_img, dst_width, dst_height, (int)box->box_p[0], (int)box->box_p[1], (int)box->box_p[2], (int)box->box_p[1], line_size, color_red);
    img_rgb888_draw_line(dst_img, dst_width, dst_height, (int)box->box_p[0], (int)box->box_p[3], (int)box->box_p[2], (int)box->box_p[3], line_size, color_red);
    img_rgb888_draw_line(dst_img, dst_width, dst_height, (int)box->box_p[0], (int)box->box_p[1], (int)box->box_p[0], (int)box->box_p[3], line_size, color_red);
    img_rgb888_draw_line(dst_img, dst_width, dst_height, (int)box->box_p[2], (int)box->box_p[1], (int)box->box_p[2], (int)box->box_p[3], line_size, color_red);
}
