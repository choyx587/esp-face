PROJECT_NAME := esp_face

MODULE_PATH := $(abspath $(shell pwd))

EXTRA_COMPONENT_DIRS += $(MODULE_PATH)/lib
EXTRA_COMPONENT_DIRS += $(MODULE_PATH)/image_util
EXTRA_COMPONENT_DIRS += $(MODULE_PATH)/face_detection
EXTRA_COMPONENT_DIRS += $(MODULE_PATH)/face_recognition
EXTRA_COMPONENT_DIRS += $(MODULE_PATH)/head_pose_estimation

include $(IDF_PATH)/make/project.mk

