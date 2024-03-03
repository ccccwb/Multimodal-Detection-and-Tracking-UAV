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
#ifndef ALGORITHM_H
#define ALGORITHM_H

/* Includes ------------------------------------------------------------------*/
//导入det or sot类的头文件
#include "yolo_det/yolo_det.hpp"
#include "kcf_sot/kcf_sot.hpp"

#ifdef __cplusplus
extern "C" {
#endif

/* Exported constants --------------------------------------------------------*/

/* Exported types ------------------------------------------------------------*/

//这个类用来提供算法支持, 由share_data中的判断位来确定做些什么(吃瓜, 检测, 以及跟踪)
class Algorithm 
{
public:

    //选择相应的det or sot类来初始化对象实例
    YOLO_Det algorithm_detector;
    KCF_Sot algorithm_tracker;

    Algorithm();
    ~Algorithm();

    //使用回调函数对解码得到的图做后处理, 回调函数在解码函数(这其实也是个回调函数)中以阻塞的方式运行(回调函数执行完才获取新图继续执行下一次回调函数
    static void *Algorithm_Task(void *arg_manager); //算法任务函数, 用于创建线程, 若为成员函数会报错
};

/* Exported functions --------------------------------------------------------*/


#ifdef __cplusplus
}
#endif

#endif // TEST_LIVEVIEW_H
/************************ (C) COPYRIGHT DJI Innovations *******END OF FILE******/
