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
#include "camera_control.hpp"
#include "manager/manager.hpp"

#include <iostream>
/* Private constants ---------------------------------------------------------*/

/* Private types -------------------------------------------------------------*/

/* Private values -------------------------------------------------------------*/

/* Private functions declaration ---------------------------------------------*/

/* Exported functions definition ---------------------------------------------*/
Camera_Control::Camera_Control()
{
    pd_x = PID_Control(0.06, 0.006, 0.003); //0.043 0.15  //左右
    pd_y = PID_Control(0.065, 0.008, 0.005);              //上下

    T_DjiReturnCode returnCode;
    E_DjiMountPosition position = DJI_MOUNT_POSITION_PAYLOAD_PORT_NO1;

    returnCode = DjiGimbalManager_Init();
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS)
    {
        throw std::runtime_error("GimbalManager init failed");
    }

    returnCode = DjiGimbalManager_SetMode(position, DJI_GIMBAL_MODE_FREE);//设置一号位云台相机 自由模式
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS)
    {
        throw std::runtime_error("GimbalManager set mode failed");
    }
}

Camera_Control::~Camera_Control()
{
    T_DjiReturnCode returnCode;
    returnCode = DjiGimbalManager_Deinit();
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS)
    {
        throw std::runtime_error("GimbalManager deinit failed");
    }
}

void *Camera_Control::Camera_Control_Task(void * arg_manager)
{
    Manager* arg = (Manager*) arg_manager;
    int16_t error_x;
    int16_t error_y;
    bool need_control;

    dji_f32_t temp_pid_x;
    dji_f32_t temp_pid_y;

    while (1)
    {
        arg->Share_Data.osalHandler->SemaphoreWait(arg->Share_Data.Error_Sema);
        //计算偏差按红外分屏算, 尺寸应当为960*768, 对应 Liveview_RGBT 类
        error_x = arg->Share_Data.Error_x;
        error_y = arg->Share_Data.Error_y;
        need_control = arg->Share_Data.Need_Control;
        arg->Share_Data.Need_Control = false;
        // std::cout << " error x y : " << error_x << " "<< error_y << std::endl;
        arg->Share_Data.osalHandler->SemaphorePost(arg->Share_Data.Error_Sema);

        if(need_control)
        {
            temp_pid_x = arg->camera_control.pd_x.pid_update(error_x);
            temp_pid_y = arg->camera_control.pd_y.pid_update(error_y);
            // std::cout << " pid result x y : " << temp_pid_x << " " << temp_pid_y << std::endl;
            
            T_DjiGimbalManagerRotation rotation = (T_DjiGimbalManagerRotation) {DJI_GIMBAL_ROTATION_MODE_SPEED, 
                            -temp_pid_y, 0, temp_pid_x, 0.05};
            T_DjiReturnCode returnCode = DjiGimbalManager_Rotate(DJI_MOUNT_POSITION_PAYLOAD_PORT_NO1, rotation);
            if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
                USER_LOG_ERROR("gimbal rotate failed, error code: 0x%08X", returnCode);
            }
        }
        else
        {
	    arg->camera_control.pd_x.pid_init();
	    arg->camera_control.pd_y.pid_init();
            //do nothing
        }

        arg->Share_Data.osalHandler->TaskSleepMs(DELTA_TIME*500);
    }
}


void Camera_Control::test_optical_zoom()
{
    T_DjiReturnCode returnCode;
    T_DjiCameraManagerOpticalZoomParam opticalZoomParam;
    E_DjiMountPosition position = DJI_MOUNT_POSITION_PAYLOAD_PORT_NO1;
    dji_f32_t factor = 5.0;

    returnCode = DjiCameraManager_GetOpticalZoomParam(position, &opticalZoomParam);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS &&
        returnCode != DJI_ERROR_CAMERA_MANAGER_MODULE_CODE_UNSUPPORTED_COMMAND) {
        USER_LOG_ERROR("Get mounted position %d camera's zoom param failed, error code :0x%08X",
                        position, returnCode);
    }

    USER_LOG_INFO("The mounted position %d camera's current optical zoom factor is:%0.1f x, "
                  "max optical zoom factor is :%0.1f x", position, opticalZoomParam.currentOpticalZoomFactor,
                  opticalZoomParam.maxOpticalZoomFactor);

    USER_LOG_INFO("Set mounted position 01 camera's zoom factor: %0.1f x.", position, factor);
    returnCode = DjiCameraManager_SetOpticalZoomParam(position, DJI_CAMERA_ZOOM_DIRECTION_IN, factor);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS &&
        returnCode != DJI_ERROR_CAMERA_MANAGER_MODULE_CODE_UNSUPPORTED_COMMAND) {
        USER_LOG_INFO("Set mounted position %d camera's zoom factor(%0.1f) failed, error code :0x%08X",
                      position, factor, returnCode);
    }
}
/* Private functions definition-----------------------------------------------*/

/****************** (C) COPYRIGHT DJI Innovations *****END OF FILE****/
