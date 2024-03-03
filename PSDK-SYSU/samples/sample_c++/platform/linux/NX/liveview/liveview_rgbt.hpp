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
#ifndef LIVEVIEW_RGBT_H
#define LIVEVIEW_RGBT_H

/* Includes ------------------------------------------------------------------*/
#include "dji_liveview.h"
#include <map>
#include "dji_camera_stream_decoder.hpp"

#ifdef __cplusplus
extern "C" {
#endif

/* Exported constants --------------------------------------------------------*/

/* Exported types ------------------------------------------------------------*/

//这个类应当用于获取H20T的流, 且该流由遥控器切换到了红外模式中的分屏模式
//这个类获取的流解码后处理应该得到的是960*768的 热红外图 + RGB图
class Liveview_RGBT 
{
public:

    Liveview_RGBT();
    ~Liveview_RGBT();

    //使用回调函数对解码得到的图做后处理, 回调函数在解码函数(这其实也是个回调函数)中以阻塞的方式运行(回调函数执行完才获取新图继续执行下一次回调函数
    static void Write_RGBT_Callback(CameraRGBImage img, void *userData_); //写图的回调函数, 作成员函数应该会报错

    T_DjiReturnCode Start_Camera_Stream(void *userData); 
    
};

/* Exported functions --------------------------------------------------------*/


#ifdef __cplusplus
}
#endif

#endif // TEST_LIVEVIEW_H
/************************ (C) COPYRIGHT DJI Innovations *******END OF FILE******/
