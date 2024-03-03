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
#ifndef MANAGER_H
#define MANAGER_H

/* Includes ------------------------------------------------------------------*/
#include "dji_typedef.h"
#include "dji_platform.h"

#include "utils/utils.hpp"
#include "mop/mop.hpp"
#include "liveview/liveview_rgbt.hpp"
#include "algorithm/algorithm.hpp"
#include "camera_control/camera_control.hpp"
#include "position_compute/position_compute.hpp"

#ifdef __cplusplus
extern "C" {
#endif

class Manager
{
    public:
        uint32_t Mop_Channel_Task_Stack_Size = 2048;
        uint32_t Algorithm_Task_Stack_Size = 2048;
        uint32_t Camera_Control_Task_Stack_Size = 2048;
        uint32_t Position_Compute_Task_Stack_Size = 2048;

        T_DjiTaskHandle Mop_Channel_Send_Task = nullptr;
        T_DjiTaskHandle Mop_Channel_Recv_Task = nullptr;
        T_DjiTaskHandle Algorithm_Task = nullptr;
        T_DjiTaskHandle Camera_Control_Task = nullptr;
	T_DjiTaskHandle Position_Compute_Task = nullptr;

        MOP_Channel mop_channel;
        Liveview_RGBT liveview_rgbt;
        Algorithm algorithm;
        bool on_tracking = false; //using thie in Algorithm to judge whether track or init
        Camera_Control camera_control;
	Position_Compute position_compute;

        T_ShareData Share_Data = {
            bbox_send : {{10,10,100,500},{420,500,200,100}},
            Bbox_Send_Sema :nullptr,
            sendArray : {0},

            flag_bbox_recv : {1, 0, 0, 0, 0},
            Flag_Bbox_Recv_Sema : nullptr,
            recvArray : {0},
            
            osalHandler : DjiPlatform_GetOsalHandler(),

            img_ptr_rgb : nullptr,
            img_ptr_tir : nullptr,
            Image_Sema : nullptr,
            New_Image : false,

            Error_x : 0,
            Error_y : 0,
            Need_Control : false,
            Error_Sema : nullptr,
            };

        Manager();
        ~Manager();
        T_DjiReturnCode Start_MOP_Task(); //初始化通道相关
        T_DjiReturnCode Start_Liveview_Task();
        T_DjiReturnCode Start_Algorithm_Task();
        T_DjiReturnCode Start_Camera_Control_Task();
        T_DjiReturnCode Start_Position_Compute_Task();
};


/* Exported functions --------------------------------------------------------*/

#ifdef __cplusplus
}
#endif

#endif // TEST_MOP_CHANNEL_H
/************************ (C) COPYRIGHT DJI Innovations *******END OF FILE******/
