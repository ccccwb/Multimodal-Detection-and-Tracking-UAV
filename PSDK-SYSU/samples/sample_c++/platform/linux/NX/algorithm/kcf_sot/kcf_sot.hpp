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
#ifndef KCF_SOT_H
#define KCF_SOT_H

/* Includes ------------------------------------------------------------------*/
#include "utils/utils.hpp"

#include "src/kcftracker.hpp"

#ifdef __cplusplus
extern "C" {
#endif

/* Exported constants --------------------------------------------------------*/

/* Exported types ------------------------------------------------------------*/

//接受share_data中的初始框, 执行跟踪并修改share_data中的bbox
class KCF_Sot
{
public:
    KCFTracker *tracker = new KCFTracker();

    KCF_Sot();
    ~KCF_Sot();

    void init(const cv::Mat &image, cv::Rect &roi); 
    void update(const cv::Mat &image, T_ShareData *arg_share_data);
};

/* Exported functions --------------------------------------------------------*/


#ifdef __cplusplus
}
#endif

#endif // TEST_LIVEVIEW_H
/************************ (C) COPYRIGHT DJI Innovations *******END OF FILE******/
