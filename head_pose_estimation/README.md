## Head Pose Estimation

Without using landmark in this project, a single image is used to obtain the three angles of the head (Pitch Yaw Roll).

![angles](../img/head_angles.png)


Head pose estimation is divided into two parts: 1. Head detection 2. Pose estimation

For head detection, any detection model that provides bounding box coordinates can be used. We recommend using the [face detection model under esp-face](../face_detection) because it is fast and accurate. Here, we only introduce the pose estimation model, input the image and head detection coordinates to obtain the head pose at three angles(Pitch, Yaw, Roll), the range of angle is between -99 degrees and 99 degrees.

## API Introduction

```cpp
head_angles_t *head_pose_estimation(dl_matrix3du_t *image_matrix, box_array_t *bboxes, float crop_k);
```
`head_pose_estimation` is the forward propagation of head pose estimation.

parametets:

* **image_matrix**: An image in dl_matrix3du_t type.
* **bboxes**: Bounding boxes. It is used to find the coordinates of the face in the image to crop the image.
* **crop_k**: Clipping factor, a real number between 0 and 1. 

outouts:

* A `head_angles_t` type value contains the head's angles of `n` people. The number of people `n` is determined by the number of bounding boxes.

This structure is defined as follows:

```cpp
typedef struct {
    float pitch;
    float yaw;
    float roll;
} angles_t;

typedef struct {
    angles_t **angles;
    int len;            /*<! people number*/
} head_angles_t;
```

```cpp
void draw_axes_and_bbox(uint8_t *dst_img, int dst_width, int dst_height, angles_t *angles, box_t *box, int size);
```

`draw_axes_and_bbox`  is Used to draw the direction axis and bounding box on the input image.

parameters:

* **dst_img**:  Input image.
* **dst_width**: The width of input image.
* **dst_height**: The height of input image.
* **angles**: Three angles.
* **size**: Length base.

For example, you can show the direction axis and bounding box in the image below.

![img](../img/result.png)

```cpp
void free_head_angles(head_angles_t *head_angles);
```

`free_head_angles` is used to release the output of forward propagation.

parameters:

* **head_angles** : Pointer to the angles of head.

## Performance
||Average time when there is only one person (ms)|
|:---:|:---:|
|head pose estimation|236ms|