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
#include "kcf_sot.hpp"

/* Exported constants --------------------------------------------------------*/

/* Exported types ------------------------------------------------------------*/
KCF_Sot::KCF_Sot()
{
}

KCF_Sot::~KCF_Sot()
{
    delete tracker;
}

void KCF_Sot::init(const cv::Mat &image, cv::Rect &roi)
{
    tracker->init(roi, image);
    return;
}

void KCF_Sot::update(const cv::Mat &image, T_ShareData *arg_share_data)
{
    T_ShareData *arg = arg_share_data;
    cv::Rect roi = tracker->update(image);

    arg->osalHandler->SemaphoreWait(arg->Bbox_Send_Sema);
    arg->bbox_send[0][0] = roi.x;
    arg->bbox_send[0][1] = roi.y;
    arg->bbox_send[0][2] = roi.width;
    arg->bbox_send[0][3] = roi.height;
    std::fill(arg->bbox_send[1], arg->bbox_send[1] + 4, 0);
    arg->osalHandler->SemaphorePost(arg->Bbox_Send_Sema);

    arg->osalHandler->SemaphoreWait(arg->Error_Sema);
    //计算偏差按红外分屏算, 尺寸应当为960*768, 对应 Liveview_RGBT 类
    arg->Error_x = roi.x + roi.width/2 - 480;
    arg->Error_y = roi.y + roi.height/2 - 384;
    if (arg->Error_x > 20 || arg->Error_x <  -20 ||
        arg->Error_y < -20 || arg->Error_y > 20)
        arg->Need_Control = true;
    arg->osalHandler->SemaphorePost(arg->Error_Sema);
    return;
}

/* Exported functions --------------------------------------------------------*/

/************************ (C) COPYRIGHT DJI Innovations *******END OF FILE******/
