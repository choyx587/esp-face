#pragma once

#include <vector>
#include "detector.hpp"

class HumanFaceDetector : public DetectorAnchorBox<int16_t, int16_t>
{
public:
    /**
     * @brief Construct a new Human Face Detector object
     * 
     * @param input_shape       The shape of input image
     * @param resize_scale      The resize scale
     * @param score_threshold   The predicted boxes with higher score than the threshold will be remained
     * @param nms_threshold     The predicted boxes with lower IoU than the threshold will be remained
     * @param top_k             The k highest score boxes will be remained 
     */
    HumanFaceDetector(std::vector<int> input_shape, float resize_scale, const float score_threshold, const float nms_threshold, const int top_k);
    
    /**
     * @brief Destroy the Human Face Detector object
     * 
     */
    ~HumanFaceDetector();
    
    /**
     * @brief Forward model and parse output feature map
     * 
     */
    void call();
};
