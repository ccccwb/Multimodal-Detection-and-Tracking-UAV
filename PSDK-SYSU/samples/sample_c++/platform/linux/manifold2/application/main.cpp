/**
 ********************************************************************
 * @file    main.cpp
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
#include <liveview/test_liveview_entry.hpp>
#include <perception/test_perception_entry.hpp>
#include <flight_control/test_flight_control.h>
#include <hms/test_hms.h>
#include <waypoint_v2/test_waypoint_v2.h>
#include <waypoint_v3/test_waypoint_v3.h>
#include <gimbal_manager/test_gimbal_manager.h>
#include "application.hpp"
#include "fc_subscription/test_fc_subscription.h"
#include <gimbal_emu/test_payload_gimbal_emu.h>
#include <camera_emu/test_payload_cam_emu_media.h>
#include <camera_emu/test_payload_cam_emu_base.h>
#include <dji_logger.h>
#include "widget/test_widget.h"
#include "widget/test_widget_speaker.h"
#include <power_management/test_power_management.h>
#include "data_transmission/test_data_transmission.h"
#include <camera_manager/test_camera_manager.h>

/* Private constants ---------------------------------------------------------*/

/* Private types -------------------------------------------------------------*/

/* Private values -------------------------------------------------------------*/

/* Private functions declaration ---------------------------------------------*/
static T_DjiReturnCode DjiTest_HighPowerApplyPinInit();
static T_DjiReturnCode DjiTest_WriteHighPowerApplyPin(E_DjiPowerManagementPinState pinState);

/* Exported functions definition ---------------------------------------------*/
int main(int argc, char **argv)
{
    Application application(argc, argv);
    char inputChar;
    T_DjiOsalHandler *osalHandler = DjiPlatform_GetOsalHandler();
    T_DjiReturnCode returnCode;
    T_DjiTestApplyHighPowerHandler applyHighPowerHandler;

start:
    std::cout
        << "\n"
        << "| Available commands:                                                                              |\n"
        << "| [0] Fc subscribe sample - subscribe quaternion and gps data                                      |\n"
        << "| [1] Flight controller sample - take off landing                                                  |\n"
        << "| [2] Flight controller sample - take off position ctrl landing                                    |\n"
        << "| [3] Flight controller sample - take off go home force landing                                    |\n"
        << "| [4] Flight controller sample - take off velocity ctrl landing                                    |\n"
        << "| [5] Flight controller sample - arrest flying                                                     |\n"
        << "| [6] Flight controller sample - set get parameters                                                |\n"
        << "| [7] Hms info sample - get health manger system info                                              |\n"
        << "| [8] Waypoint 2.0 sample - run airline mission by settings (only support on M300 RTK)             |\n"
        << "| [9] Waypoint 3.0 sample - run airline mission by kmz file (not support on M300 RTK)              |\n"
        << "| [a] Gimbal manager sample - rotate gimbal on free mode                                           |\n"
        << "| [b] Gimbal manager sample - rotate gimbal on yaw follow mode                                     |\n"
        << "| [c] Camera stream view sample - display the camera video stream                                  |\n"
        << "| [d] Stereo vision view sample - display the stereo image                                         |\n"
        << "| [e] Start camera all features sample - you can operate the camera on DJI Pilot                   |\n"
        << "| [f] Start gimbal all features sample - you can operate the gimbal on DJI Pilot                   |\n"
        << "| [g] Start widget all features sample - you can operate the widget on DJI Pilot                   |\n"
        << "| [h] Start widget speaker sample - you can operate the speaker on DJI Pilot2                      |\n"
        << "| [i] Start power management sample - you will see notification when aircraft power off            |\n"
        << "| [j] Start data transmission sample - you can send or recv custom data on MSDK demo               |\n"
        << "| [l] Run camera manager sample - shoot photo by the selected camera mounted position              |\n"
        << "| [m] Run camera manager download sample - download camera media file (only support on M3E/M3T)    |\n"
        << std::endl;

    std::cin >> inputChar;
    switch (inputChar) {
        case '7':
            DjiTest_HmsRunSample();
            break;
        case 'a':
            DjiTest_GimbalManagerRunSample(DJI_MOUNT_POSITION_PAYLOAD_PORT_NO1, DJI_GIMBAL_MODE_FREE);
            break;
        case 'b':
            DjiTest_GimbalManagerRunSample(DJI_MOUNT_POSITION_PAYLOAD_PORT_NO1, DJI_GIMBAL_MODE_YAW_FOLLOW);
            break;
        case 'c':
            DjiUser_RunCameraStreamViewSample();
            break;
        case 'd':
            DjiUser_RunStereoVisionViewSample();
            break;
        case 'e':
            returnCode = DjiTest_CameraEmuBaseStartService();
            if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
                USER_LOG_ERROR("camera emu common init error");
                break;
            }

            if (DjiPlatform_GetSocketHandler() != nullptr) {
                returnCode = DjiTest_CameraEmuMediaStartService();
                if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
                    USER_LOG_ERROR("camera emu media init error");
                    break;
                }
            }

            USER_LOG_INFO("Start camera all feautes sample successfully");
            break;
        case 'f':
            if (DjiTest_GimbalStartService() != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
                USER_LOG_ERROR("psdk gimbal init error");
                break;
            }

            USER_LOG_INFO("Start gimbal all feautes sample successfully");
            break;
        case 'g':
            returnCode = DjiTest_WidgetStartService();
            if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
                USER_LOG_ERROR("widget sample init error");
                break;
            }

            USER_LOG_INFO("Start widget all feautes sample successfully");
            break;
        case 'h':
            returnCode = DjiTest_WidgetSpeakerStartService();
            if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
                USER_LOG_ERROR("widget speaker test init error");
                break;
            }

            USER_LOG_INFO("Start widget speaker sample successfully");
            break;
        case 'i':
            applyHighPowerHandler.pinInit = DjiTest_HighPowerApplyPinInit;
            applyHighPowerHandler.pinWrite = DjiTest_WriteHighPowerApplyPin;

            returnCode = DjiTest_RegApplyHighPowerHandler(&applyHighPowerHandler);
            if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
                USER_LOG_ERROR("regsiter apply high power handler error");
                break;
            }

            returnCode = DjiTest_PowerManagementStartService();
            if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
                USER_LOG_ERROR("power management init error");
                break;
            }

            USER_LOG_INFO("Start power management sample successfully");
            break;
        case 'j':
            returnCode = DjiTest_DataTransmissionStartService();
            if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
                USER_LOG_ERROR("data transmission sample init error");
                break;
            }

            USER_LOG_INFO("Start data transmission sample successfully");
            break;
        case 'l':
            DjiTest_CameraManagerRunSample(DJI_MOUNT_POSITION_PAYLOAD_PORT_NO1,
                                           E_DJI_TEST_CAMERA_MANAGER_SAMPLE_SELECT_SET_CAMERA_ZOOM_PARAM);
            break;
        case 'm':
            DjiTest_CameraManagerRunSample(DJI_MOUNT_POSITION_PAYLOAD_PORT_NO1,
                                           E_DJI_TEST_CAMERA_MANAGER_SAMPLE_SELECT_DOWNLOAD_AND_DELETE_MEDIA_FILE);
            exit(1);
            break;
        default:
            break;
    }

    osalHandler->TaskSleepMs(2000);

    goto start;
}

/* Private functions definition-----------------------------------------------*/
static T_DjiReturnCode DjiTest_HighPowerApplyPinInit()
{
    return DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS;
}

static T_DjiReturnCode DjiTest_WriteHighPowerApplyPin(E_DjiPowerManagementPinState pinState)
{
    //attention: please pull up the HWPR pin state by hardware.
    return DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS;
}

/****************** (C) COPYRIGHT DJI Innovations *****END OF FILE****/
