/**
 ********************************************************************
 * @file    test_liveview.hpp
 * @brief   This is the header file for "test_liveview.cpp", defining the structure and
 * (exported) function prototypes.
 *
 * @copyright (c) 2021 DJI. All rights reserved.
 *
 * All information contained herein is, and remains, the property of DJI.
 * The intellectual and technical concepts contained herein are proprietary
 * to DJI and may be covered by U.S. and foreign patents, patents in process,
 * and protected by trade secret or copyright law.  Dissemination of this
 * information, including but not limited to data and other proprietary
 * material(s) incorporated within the information, in any form, is strictly
 * prohibited without the express written consent of DJI.
 *
 * If you receive this source code without DJI’s authorization, you may not
 * further disseminate the information, and you must immediately remove the
 * source code and notify DJI of its removal. DJI reserves the right to pursue
 * legal actions against you for any loss(es) or damage(s) caused by your
 * failure to do so.
 *
 *********************************************************************
 */

/* Define to prevent recursive inclusion -------------------------------------*/

/* Includes ------------------------------------------------------------------*/
#include "yolo_det.hpp"

/* Exported constants --------------------------------------------------------*/
/* Exported types ------------------------------------------------------------*/

YOLO_Det::YOLO_Det()
{
    std::string current_path = Get_Current_File_DirPath(__FILE__);
	Config config_v5;
	config_v5.net_type = YOLOV5;
	config_v5.detect_thresh = 0.5;
	config_v5.file_model_cfg = current_path + "cfg/yolov5-6.0/yolov5n.cfg";
	config_v5.file_model_weights = current_path + "cfg/yolov5-6.0/yolov5n.weights";
	config_v5.calibration_image_list_file_txt = current_path + "cfg/calibration_images.txt";
	config_v5.inference_precison = FP16;
	detector->init(config_v5);
}

YOLO_Det::~YOLO_Det()
{
    delete detector;
}

void YOLO_Det::det(const cv::Mat &image, T_ShareData *arg_share_data)
{
    T_ShareData *arg = arg_share_data;
    batch_img.push_back(image);
    detector->detect(batch_img, batch_res);

    uint32_t bbox_num_real = batch_res[0].size(); //只有一张图!
    uint8_t num_ = std::min((uint32_t)BBOX_NUM, bbox_num_real);

    arg->osalHandler->SemaphoreWait(arg->Bbox_Send_Sema);
    for (int i=0; i<num_; i++)
    {
        //num_保证了不会越界
        auto &rect_bbox = batch_res[0][i];
        arg->bbox_send[i][0] = rect_bbox.rect.x;
        arg->bbox_send[i][1] = rect_bbox.rect.y;
        arg->bbox_send[i][2] = rect_bbox.rect.width;
        arg->bbox_send[i][3] = rect_bbox.rect.height;
    }
    if (num_!=BBOX_NUM)
    {
        //一组框置0表示结束
        std::fill(arg->bbox_send[num_], arg->bbox_send[num_] + 4, 0);
    }
    arg->osalHandler->SemaphorePost(arg->Bbox_Send_Sema);

    batch_img.clear();//清空, 无需释放内存其实....,大小应该是统一的
    return;
}

/* Exported functions --------------------------------------------------------*/

/************************ (C) COPYRIGHT DJI Innovations *******END OF FILE******/
