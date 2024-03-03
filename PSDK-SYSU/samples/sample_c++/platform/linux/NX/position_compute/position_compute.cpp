/**
 ********************************************************************
 * @file    test_gimbal_manager.c
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
#include "position_compute.hpp"
#include "manager/manager.hpp"

#include <iostream>
/* Private constants ---------------------------------------------------------*/

/* Private types -------------------------------------------------------------*/

/* Private values -------------------------------------------------------------*/

/* Private functions declaration ---------------------------------------------*/

/* Exported functions definition ---------------------------------------------*/
Position_Compute::Position_Compute()
{

    T_DjiReturnCode returnCode;
    E_DjiMountPosition position = DJI_MOUNT_POSITION_PAYLOAD_PORT_NO1;

    // chushi hua dingyue
    returnCode = DjiFcSubscription_Init();
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS)
    {
        throw std::runtime_error("DjiFcSubscription init failed");
    }

}

Position_Compute::~Position_Compute()
{
    T_DjiReturnCode returnCode;
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS)
    {
        throw std::runtime_error("DjiFcSubscription deinit failed");
    }
}

void *Position_Compute::Position_Compute_Task(void * arg_manager)
{
    Manager* arg = (Manager*) arg_manager;
    T_DjiReturnCode returnCode;
    T_DjiFcSubscriptionGimbalAngles angles;
    T_DjiFcSubscriptionRtkPosition rtk_pos;
    T_DjiFcSubscriptionRtkPositionInfo rtk_flag;
    dji_f64_t pitch, yaw, roll;
    /* topic T_DjiFcSubscriptionGimbalAngles  
    *  angles T_djiVector3f
    */

    returnCode = DjiFcSubscription_SubscribeTopic(DJI_FC_SUBSCRIPTION_TOPIC_GIMBAL_ANGLES, DJI_DATA_SUBSCRIPTION_TOPIC_50_HZ, NULL);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS)
    {
        USER_LOG_ERROR("DjiFcSubscription GIMBAL ANGLES failed");
    }
////
    returnCode = DjiFcSubscription_SubscribeTopic(DJI_FC_SUBSCRIPTION_TOPIC_RTK_POSITION, DJI_DATA_SUBSCRIPTION_TOPIC_1_HZ, NULL);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS)
    {
        USER_LOG_ERROR("DjiFcSubscription RTK failed");
    }
////
    returnCode = DjiFcSubscription_SubscribeTopic(DJI_FC_SUBSCRIPTION_TOPIC_RTK_POSITION_INFO, DJI_DATA_SUBSCRIPTION_TOPIC_1_HZ, NULL);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS)
    {
        USER_LOG_ERROR("DjiFcSubscription RTK INFO failed");
    }
    while(1){

        returnCode = DjiFcSubscription_GetLatestValueOfTopic(DJI_FC_SUBSCRIPTION_TOPIC_GIMBAL_ANGLES, (uint8_t *) &angles, sizeof(T_DjiFcSubscriptionGimbalAngles), NULL);
        if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS)
	{
            USER_LOG_ERROR("DJI_FC_SUBSCRIPTION_TOPIC_GIMBAL_ANGLES failed");
	}
        pitch = angles.x;
        roll = angles.y;
        yaw = angles.z;
        //USER_LOG_INFO("pitch = %.2f  roll = %.2f   yaw = %.2f ", pitch, roll, yaw);
	
	//rtk
        returnCode = DjiFcSubscription_GetLatestValueOfTopic(DJI_FC_SUBSCRIPTION_TOPIC_RTK_POSITION, (uint8_t *) &rtk_pos, sizeof(T_DjiFcSubscriptionRtkPosition), NULL);
        if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS)
	{
            USER_LOG_ERROR("DJI_FC_SUBSCRIPTION_TOPIC_RTK_POSITION failed");
	}

	// USER_LOG_INFO("rtk height = %.2f", rtk_pos.hfsl);


	// fang xiang jiao 
	dji_f64_t gama = 90 - pitch; // he z zhou de jiao
	dji_f64_t alpha = 90 - yaw; // he x zhou
	dji_f64_t height = rtk_pos.hfsl*1000;
	// the line throught point (0,0,height)
	// the line function is x/a = y/b = (z-height)/c
	dji_f64_t a = cos(alpha*PI/180);
	dji_f64_t c = cos(gama*PI/180)+0.0000001;
	dji_f64_t b = sqrt(1 - a*a - c*c);
	dji_f64_t z = 0;
	dji_f64_t x = (z-height)/c*a;
	dji_f64_t y = (z-height)/c*b;
    
    // USER_LOG_INFO("x = %.2f  y = %.2f   z = %.2f ", x, y, z);
	arg->Share_Data.osalHandler->TaskSleepMs(1000);

    }

}
/*static T_DjiReturnCode DjiTest_FcSubscriptionReceiveAngleCallback(const uint8_t *data, uint16_t dataSize, const T_DjiDataTimestamp *timestamp)
{
    T_DjiVector3f * angles = (T_DjiVector3f *) data;
    dji_f64_t pitch, yaw, roll;
    
    //USER_UTIL_UNUSED(dataSize);
    pitch = angles->x;
    roll = angles->y;
    yaw = angles->z;
    USER_LOG_INFO("pitch = %.2f  roll = %.2f   yaw = %.2f ", pitch, yaw, roll);
    return DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS;

}*/
/* Private functions definition-----------------------------------------------*/

/****************** (C) COPYRIGHT DJI Innovations *****END OF FILE****/
