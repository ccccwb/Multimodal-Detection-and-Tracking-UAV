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

/* Includes ------------------------------------------------------------------*/
#include "utils.hpp"

// 用来计算时间 gettimeofday
uint32_t Get_Duration_Time_ms(struct timeval t_start,struct timeval t_end)
{
    return ((t_start.tv_sec - t_end.tv_sec) * 1e6 + (t_start.tv_usec - t_end.tv_usec)) / 1000;
}

std::string Get_Current_File_DirPath(const std::string filePath)
{
    uint32_t i = filePath.length() - 1;
    while (filePath[i] != '/') //抄DJI sample的操作, 这要windows那种'\\'不就寄了
    {
        i--;
    }
    return filePath.substr(0, i + 1);
}
/************************ (C) COPYRIGHT DJI Innovations *******END OF FILE******/
