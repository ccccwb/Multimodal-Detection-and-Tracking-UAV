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

#include "dji_logger.h"
#include "mop.hpp"

#include "manager/manager.hpp" //Task函数用Manager类的指针

/* Private constants ---------------------------------------------------------*/

/* Private values -------------------------------------------------------------*/

/* Private functions declaration ---------------------------------------------*/
MOP_Channel::MOP_Channel()
{
}

MOP_Channel::~MOP_Channel()
{
}

T_DjiReturnCode MOP_Channel::Mop_Channel_Semaphore_Init(T_ShareData *arg)
{
    T_DjiReturnCode returnCode;
    returnCode = arg->osalHandler->SemaphoreCreate(0, &Mop_Channel_Ready_Sema);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("mop channel create msdk sema error, stat:0x%08llX.", returnCode);
        return DJI_ERROR_SYSTEM_MODULE_CODE_UNKNOWN;
    }
    
    returnCode = DjiMopChannel_Init();
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("mop channel init error, stat:0x%08llX.", returnCode);
        return DJI_ERROR_SYSTEM_MODULE_CODE_UNKNOWN;
    }

    return DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS;
}

void *MOP_Channel::Send_Task(void *arg_manager)
{
    Manager* arg = (Manager*) arg_manager; //直接使用更上面的manager类指针(Manager类中有一个MOP_Channel类的成员
    uint32_t realLen = 0;
    T_DjiReturnCode returnCode;
    uint32_t sendLen = BBOX_NUM*COORDINATE_NUM*BYTES_NUM;

    struct timeval start_time, stop_time;//debug

REWAIT:
    returnCode = arg->Share_Data.osalHandler->SemaphoreWait(arg->mop_channel.Mop_Channel_Ready_Sema);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("mop channel wait sema error, stat:0x%08llX.", returnCode);
        return NULL;
    }

    while (1) {
        if (arg->mop_channel.Mop_Channel_Connected == false) {
            goto REWAIT;
        }

        gettimeofday(&start_time, NULL);
        
        arg->Share_Data.osalHandler->SemaphoreWait(arg->Share_Data.Bbox_Send_Sema);
        // uint16 → uint8 for SendData
        for( int i = 0; i < BBOX_NUM; i++)
        {
            for( int j=0; j < COORDINATE_NUM; j++)
            {
                arg->Share_Data.sendArray[i][j][0] = arg->Share_Data.bbox_send[i][j] & 0xff;
                arg->Share_Data.sendArray[i][j][1] = arg->Share_Data.bbox_send[i][j] >> 8;
            }
        }

        // std::cout<<"发送的第一个框"<<arg->Share_Data.bbox_send[0][0]<<" "<<arg->Share_Data.bbox_send[0][1]<<" "
        //         <<arg->Share_Data.bbox_send[0][2]<<" "<<arg->Share_Data.bbox_send[0][3]<<std::endl;

        arg->Share_Data.osalHandler->SemaphorePost(arg->Share_Data.Bbox_Send_Sema);
        
        returnCode = DjiMopChannel_SendData(arg->mop_channel.Mop_Channel_Out_Handle, arg->Share_Data.sendArray[0][0],
                                            sendLen, &realLen);

        gettimeofday(&stop_time, NULL);

        if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
            USER_LOG_ERROR("mop channel send data to channel error,stat:0x%08llX", returnCode);
        } else {
            // USER_LOG_INFO("mop channel send data to channel length:%d", realLen);
            if (realLen != sendLen)
            {
                USER_LOG_ERROR("mop channel send data from channel length:%d != expected length: %d", realLen, sendLen);
            }
        }

        // printf("SendData function time use %d ms\n", Get_Duration_Time_ms(start_time,stop_time));
        uint32_t duration = Get_Duration_Time_ms(start_time,stop_time); //ms
        int32_t sleep_time = 1000 / SEND_TASK_FREQ - duration - 5;//假定判断是否sleep的操作要5ms hhh
        // std::cout << "sleep time, duration time " << sleep_time << " "<< duration << std::endl;
        if (sleep_time > 0)
            arg->Share_Data.osalHandler->TaskSleepMs(sleep_time);

    }

    return nullptr;
}

void *MOP_Channel::Recv_Task(void *arg_manager)
{   
    Manager* arg = (Manager*) arg_manager; //直接使用更上面的manager类指针(Manager类中有一个MOP_Channel类的成员
    uint32_t realLen = 0;
    T_DjiReturnCode returnCode;
    uint32_t recvLen = RECV_ALL*BYTES_NUM;

    if (arg->mop_channel.Transfor_Using_Reliable_Trans)
    {
        returnCode = DjiMopChannel_Create(&(arg->mop_channel.Mop_Channel_Handle), DJI_MOP_CHANNEL_TRANS_RELIABLE);
        if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
            USER_LOG_ERROR("mop channel create send handle error, stat:0x%08llX.", returnCode);
            return NULL;
    }
    }
    else
    {
        returnCode = DjiMopChannel_Create(&(arg->mop_channel.Mop_Channel_Handle), DJI_MOP_CHANNEL_TRANS_UNRELIABLE);
        if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
            USER_LOG_ERROR("mop channel create send handle error, stat:0x%08llX.", returnCode);
            return NULL;
        }
    }

REBIND:
    returnCode = DjiMopChannel_Bind(arg->mop_channel.Mop_Channel_Handle, arg->mop_channel.Channel_Id);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("mop bind channel error :0x%08llX", returnCode);
        arg->Share_Data.osalHandler->TaskSleepMs(arg->mop_channel.Channel_Retry_Times);
        goto REBIND;
    }

REACCEPT:
    returnCode = DjiMopChannel_Accept(arg->mop_channel.Mop_Channel_Handle, &(arg->mop_channel.Mop_Channel_Out_Handle));
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_WARN("mop accept channel error :0x%08llX", returnCode);
        arg->Share_Data.osalHandler->TaskSleepMs(arg->mop_channel.Channel_Retry_Times);
        goto REACCEPT;
    }
    
    USER_LOG_INFO("mop channel is connected");
    arg->mop_channel.Mop_Channel_Connected = true;
    returnCode = arg->Share_Data.osalHandler->SemaphorePost(arg->mop_channel.Mop_Channel_Ready_Sema);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("mop channel post sema error, stat:0x%08llX.", returnCode);
        return nullptr;
    }

    while (1) {
        returnCode = DjiMopChannel_RecvData(arg->mop_channel.Mop_Channel_Out_Handle, arg->Share_Data.recvArray,
                                            recvLen, &realLen);
        if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
            if (returnCode == DJI_ERROR_MOP_CHANNEL_MODULE_CODE_CONNECTION_CLOSE) {
                USER_LOG_INFO("mop channel is disconnected");
                arg->mop_channel.Mop_Channel_Connected = false;
                arg->Share_Data.osalHandler->TaskSleepMs(arg->mop_channel.Channel_Retry_Times);
                DjiMopChannel_Close(arg->mop_channel.Mop_Channel_Out_Handle);
                DjiMopChannel_Destroy(arg->mop_channel.Mop_Channel_Out_Handle);
                goto REACCEPT;
            }
            else if (returnCode == DJI_ERROR_SYSTEM_MODULE_CODE_TIMEOUT)
            {
                USER_LOG_ERROR("mop channel recv timeout, continue to recv, stat:0x%08llX.", returnCode);
            }
            else
            {
                 USER_LOG_INFO("mop recv return code is stat:0x%08llX", returnCode);
            }
            
        } else {
            USER_LOG_INFO("mop channel recv data from channel length:%d", realLen);
            if (realLen != recvLen)
            {
                USER_LOG_ERROR("recv data from mop channel length:%d != expected length: %d,"\
                        " so will not convert uint16 to uint8 and flag_bbox_recv will not update !!!", realLen, recvLen);
            }
            else
            {
                arg->Share_Data.osalHandler->SemaphoreWait(arg->Share_Data.Flag_Bbox_Recv_Sema);
                // uint16 → uint8 for SendData
                char temp_string[6*RECV_ALL]; //uint16 不超过65536, 最多五位字符再加空格
                for( int i = 0; i < RECV_ALL; i++)
                {
                    arg->Share_Data.flag_bbox_recv[i]=(((uint16_t)arg->Share_Data.recvArray[2*i+1] << 8) | arg->Share_Data.recvArray[2*i]);
                    sprintf(&temp_string[6*i], "%5d ", arg->Share_Data.flag_bbox_recv[i]);
                }
                arg->Share_Data.osalHandler->SemaphorePost(arg->Share_Data.Flag_Bbox_Recv_Sema);
                USER_LOG_INFO("转换结果: %s", temp_string);
            }
        }

    }
    return nullptr;
}
/* Exported functions definition ---------------------------------------------*/

/* Private functions definition-----------------------------------------------*/

/****************** (C) COPYRIGHT DJI Innovations *****END OF FILE****/
