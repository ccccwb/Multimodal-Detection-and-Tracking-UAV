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
#include <iostream>

#include "dji_logger.h"
// #include "dji_typedef.h"

#include "application.hpp"
#include "manager/manager.hpp"

/* Private constants ---------------------------------------------------------*/

/* Private types -------------------------------------------------------------*/

/* Private values -------------------------------------------------------------*/

/* Private functions declaration ---------------------------------------------*/

/* Exported functions definition ---------------------------------------------*/
int main(int argc, char **argv)
{
    Application application(argc, argv);
    char inputChar;
    Manager manager;
    T_DjiReturnCode returnCode;

    returnCode = manager.Start_Liveview_Task();
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("liveview start error");

    }
    USER_LOG_INFO("liveview start successfully");

    returnCode = manager.Start_MOP_Task();
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("mop start error");

    }
    USER_LOG_INFO("mop start successfully");

    returnCode = manager.Start_Algorithm_Task();
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Algorithm start error");

    }
    USER_LOG_INFO("Algorithm start successfully");

    returnCode = manager.Start_Camera_Control_Task();
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Camera control start error");

    }
    USER_LOG_INFO("Camera control start successfully");

    returnCode = manager.Start_Position_Compute_Task();
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Position compute start error");

    }
    USER_LOG_INFO("Position compute start successfully");

    while(1)
    {
        std::cout<<"程序已启动, q to quit"<<std::endl;
        std::cin >> inputChar;
        if (inputChar == 'q')
            break;
    }

}

/****************** (C) COPYRIGHT DJI Innovations *****END OF FILE****/
