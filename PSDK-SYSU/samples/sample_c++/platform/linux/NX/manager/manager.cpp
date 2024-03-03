/**
 ********************************************************************
 * @file    test_mop_channel.c
 * @brief
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

/* Includes ------------------------------------------------------------------*/
#include <stdio.h>
#include <iostream>
#include <sys/time.h>


#include "dji_logger.h"
#include "manager.hpp"

/* Private constants ---------------------------------------------------------*/

/* Private values -------------------------------------------------------------*/

/* Private functions declaration ---------------------------------------------*/
Manager::Manager()
{
    //初始化信号量
    Share_Data.osalHandler->SemaphoreCreate(1, &Share_Data.Bbox_Send_Sema);
    Share_Data.osalHandler->SemaphoreCreate(1, &Share_Data.Flag_Bbox_Recv_Sema);
    Share_Data.osalHandler->SemaphoreCreate(1, &Share_Data.Image_Sema);
    Share_Data.osalHandler->SemaphoreCreate(1, &Share_Data.Error_Sema);
}

Manager::~Manager()
{
    Share_Data.osalHandler->SemaphoreDestroy(Share_Data.Bbox_Send_Sema);
    Share_Data.osalHandler->SemaphoreDestroy(Share_Data.Flag_Bbox_Recv_Sema);
    Share_Data.osalHandler->SemaphoreDestroy(Share_Data.Image_Sema);
    Share_Data.osalHandler->SemaphoreDestroy(Share_Data.Error_Sema);
}

T_DjiReturnCode Manager::Start_MOP_Task()
{
    T_DjiReturnCode returnCode;
    returnCode = mop_channel.Mop_Channel_Semaphore_Init(&Share_Data);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("MOP channel or sema init error, stat:0x%08llX.", returnCode);
        return DJI_ERROR_SYSTEM_MODULE_CODE_UNKNOWN;
    }

    returnCode = Share_Data.osalHandler->TaskCreate("mop_msdk_send_task", mop_channel.Send_Task,
                                         Mop_Channel_Task_Stack_Size, this, &Mop_Channel_Send_Task);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("mop channel msdk send task create error, stat:0x%08llX.", returnCode);
        return DJI_ERROR_SYSTEM_MODULE_CODE_UNKNOWN;
    }

    returnCode = Share_Data.osalHandler->TaskCreate("mop_msdk_recv_task", mop_channel.Recv_Task,
                                         Mop_Channel_Task_Stack_Size, this, &Mop_Channel_Recv_Task);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("mop channel msdk recv task create error, stat:0x%08llX.", returnCode);
        return DJI_ERROR_SYSTEM_MODULE_CODE_UNKNOWN;
    }

    return DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS;
}

T_DjiReturnCode Manager::Start_Liveview_Task()
{
    T_DjiReturnCode returnCode;
    returnCode = liveview_rgbt.Start_Camera_Stream(&Share_Data);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Start camera stream error, stat:0x%08llX.", returnCode);
        return DJI_ERROR_SYSTEM_MODULE_CODE_UNKNOWN;
    }

    return DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS;  
}

T_DjiReturnCode Manager::Start_Algorithm_Task()
{
    T_DjiReturnCode returnCode;
    returnCode = Share_Data.osalHandler->TaskCreate("algorithm_task", algorithm.Algorithm_Task,
                                         Algorithm_Task_Stack_Size, this, &Algorithm_Task);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("mop channel msdk send task create error, stat:0x%08llX.", returnCode);
        return DJI_ERROR_SYSTEM_MODULE_CODE_UNKNOWN;
    }

    return DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS;  
}

T_DjiReturnCode Manager::Start_Camera_Control_Task()
{
    T_DjiReturnCode returnCode;
    returnCode = Share_Data.osalHandler->TaskCreate("camera_control_task", camera_control.Camera_Control_Task,
                                         Camera_Control_Task_Stack_Size, this, &Camera_Control_Task);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("camera create task create error, stat:0x%08llX.", returnCode);
        return DJI_ERROR_SYSTEM_MODULE_CODE_UNKNOWN;
    }

    return DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS;  
}

T_DjiReturnCode Manager::Start_Position_Compute_Task()
{
    T_DjiReturnCode returnCode;
    returnCode = Share_Data.osalHandler->TaskCreate("position_compute_task", position_compute.Position_Compute_Task,
                                         Position_Compute_Task_Stack_Size, this, &Position_Compute_Task);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("test_computer create task create error, stat:0x%08llX.", returnCode);
        return DJI_ERROR_SYSTEM_MODULE_CODE_UNKNOWN;
    }

    return DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS;  
}

/* Exported functions definition ---------------------------------------------*/

/* Private functions definition-----------------------------------------------*/

/****************** (C) COPYRIGHT DJI Innovations *****END OF FILE****/
