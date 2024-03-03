/**
 ********************************************************************
 * @file    test_mop_channel.h
 * @brief   This is the header file for "test_mop_channel.c", defining the structure and
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
#ifndef UTILS_H
#define UTILS_H

/* Includes ------------------------------------------------------------------*/
#include "dji_typedef.h"
#include "dji_platform.h"
#include "dji_logger.h"

#include "opencv2/opencv.hpp"
#include <sys/time.h>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

/* 定义全局的发送和接受变量 --------------------------------------------------------*/

// 三者相乘为总字节数量, 4-xywh, 2-uint16, 20-框数量
#define BBOX_NUM                                                 20
#define COORDINATE_NUM                                           4

//uint16的字节数, 如修改uint16的设定, 该值和后续的uint16 ←→ uint8 转换打印要改写
#define BYTES_NUM                                                2
#define EXTRA_FOR_RECV                                           1 //判断位
#define RECV_ALL                                                 (COORDINATE_NUM + EXTRA_FOR_RECV)

//接受图像流的宽高, 录像模式且红外才是这个宽高!!!, 务必保证遥控器切换到录像模式下的红外分屏
#define IMAGE_WIDTH    1920
#define IMAGE_HEIGHT   1080

//执行任务的频率
#define SEND_TASK_FREQ 36
#define ALGORITHM_RUN_FREQ 30

//云台pid刷新率
#define DELTA_TIME  0.1 

// 用来计算时间 gettimeofday
uint32_t Get_Duration_Time_ms(struct timeval t_start,struct timeval t_end);

//获取文件所在路径(给某些函数读文件用)
std::string Get_Current_File_DirPath(const std::string filePath);

typedef struct
{
    uint16_t bbox_send[BBOX_NUM][COORDINATE_NUM];//xywh
    T_DjiSemaHandle Bbox_Send_Sema;
    uint8_t sendArray[BBOX_NUM][COORDINATE_NUM][BYTES_NUM];

    uint16_t flag_bbox_recv[RECV_ALL]; //接受后转换回uint16 (判断位+xywh)
    T_DjiSemaHandle Flag_Bbox_Recv_Sema;
    uint8_t recvArray[RECV_ALL*BYTES_NUM]; //用来接收!
    T_DjiOsalHandler* osalHandler;

    //两个图像相关的指针应当一起被操作(即使使用双模态!), 注意: new出来的变量在算法模块获取到后应即使释放
    cv::Mat* img_ptr_rgb;
    cv::Mat* img_ptr_tir;
    T_DjiSemaHandle Image_Sema; //用于图像的信号量
    bool New_Image;//用于判断是否更新了图片

    //用于控制云台的相关变量，应当仅在跟踪模式下使用
    int16_t Error_x;
    int16_t Error_y;
    bool Need_Control;
    T_DjiSemaHandle Error_Sema;
 
}T_ShareData;

#ifdef __cplusplus
}
#endif

#endif // TEST_MOP_CHANNEL_H
/************************ (C) COPYRIGHT DJI Innovations *******END OF FILE******/
