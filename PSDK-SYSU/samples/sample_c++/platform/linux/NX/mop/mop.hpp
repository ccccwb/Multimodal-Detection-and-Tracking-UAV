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
#ifndef MOP_H
#define MOP_H

/* Includes ------------------------------------------------------------------*/
#include "dji_typedef.h"
#include "dji_platform.h"
#include "dji_mop_channel.h"

#include "utils/utils.hpp"

#ifdef __cplusplus
extern "C" {
#endif

class MOP_Channel
{
    public:
        //传输设置相关变量, 固定
        uint16_t Channel_Id = 49152;
        // uint16_t channel_init_times = 3 * 1000; //用于启动send任务之前等待recv任务中的通道建立
        uint16_t Channel_Retry_Times = 3 * 1000; //用于recv任务中通道相关重试
        bool Transfor_Using_Reliable_Trans = false;
        // uint16_t Transfor_Send_Task_Freq = 30;//发送信息30秒一次

        T_DjiSemaHandle Mop_Channel_Ready_Sema = nullptr; //用于判断通道建立的信号量
        bool Mop_Channel_Connected = false; //用于判断通道是否连接,(这两个变量的操作是照抄大疆mop sample的)

        //内部通道相关的handle
        T_DjiMopChannelHandle Mop_Channel_Handle = nullptr;
        T_DjiMopChannelHandle Mop_Channel_Out_Handle = nullptr;

        MOP_Channel();
        ~MOP_Channel();
        T_DjiReturnCode Mop_Channel_Semaphore_Init(T_ShareData *arg); //初始化通道相关

        //就用静态函数吧...
        static void *Send_Task(void *arg_manager); //实际接受Manager类的指针, 负责发送数据
        static void *Recv_Task(void *arg_manager); //实际接受Manager类的指针, 负责接受数据并且维护通道的建立

};


/* Exported functions --------------------------------------------------------*/

#ifdef __cplusplus
}
#endif

#endif // TEST_MOP_CHANNEL_H
/************************ (C) COPYRIGHT DJI Innovations *******END OF FILE******/
