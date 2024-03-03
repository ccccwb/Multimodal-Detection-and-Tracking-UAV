/**
 ********************************************************************
 * @file    test_positioning.c
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
#include <fc_subscription/test_fc_subscription.h>
#include "test_positioning.h"
#include "dji_positioning.h"
#include "dji_logger.h"
#include "utils/util_misc.h"
#include "dji_platform.h"
#include "time_sync/test_time_sync.h"

/* Private constants ---------------------------------------------------------*/
#define POSITIONING_TASK_FREQ                     (1)
#define POSITIONING_TASK_STACK_SIZE               (1024)

#define DJI_TEST_POSITIONING_EVENT_COUNT          (2)
#define DJI_TEST_TIME_INTERVAL_AMONG_EVENTS_US    (200000)

/* Private types -------------------------------------------------------------*/


/* Private functions declaration ---------------------------------------------*/
static void *DjiTest_PositioningTask(void *arg);

/* Private variables ---------------------------------------------------------*/
static T_DjiTaskHandle s_userPositioningThread;
static int32_t s_eventIndex = 0;

/* Exported functions definition ---------------------------------------------*/
T_DjiReturnCode DjiTest_PositioningStartService(void)
{
    T_DjiReturnCode djiStat;
    T_DjiOsalHandler *osalHandler = DjiPlatform_GetOsalHandler();

    djiStat = DjiPositioning_Init();
    if (djiStat != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("positioning module init error.");
        return djiStat;
    }

    DjiPositioning_SetTaskIndex(0);

    if (osalHandler->TaskCreate("user_positioning_task", DjiTest_PositioningTask,
                                POSITIONING_TASK_STACK_SIZE, NULL, &s_userPositioningThread) !=
        DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("user positioning task create error.");
        return DJI_ERROR_SYSTEM_MODULE_CODE_UNKNOWN;
    }

    return DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS;
}

/* Private functions definition-----------------------------------------------*/
#ifndef __CC_ARM
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-noreturn"
#pragma GCC diagnostic ignored "-Wreturn-type"
#endif

static void *DjiTest_PositioningTask(void *arg)
{
    int32_t i = 0;
    T_DjiReturnCode djiStat;
    uint64_t ppsNewestTriggerTimeUs = 0;
    T_DjiPositioningEventInfo eventInfo[DJI_TEST_POSITIONING_EVENT_COUNT] = {0};
    T_DjiPositioningPositionInfo positionInfo[DJI_TEST_POSITIONING_EVENT_COUNT] = {0};
    T_DjiTimeSyncAircraftTime aircraftTime = {0};
    T_DjiOsalHandler *osalHandler = DjiPlatform_GetOsalHandler();
    uint8_t totalSatelliteNumber = 0;

    USER_UTIL_UNUSED(arg);

    while (1) {
        osalHandler->TaskSleepMs(1000 / POSITIONING_TASK_FREQ);

        djiStat = DjiTest_FcSubscriptionGetTotalSatelliteNumber(&totalSatelliteNumber);
        if (djiStat != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
            USER_LOG_ERROR("get total satellite number error: 0x%08llX.", djiStat);
            continue;
        }

        djiStat = DjiTest_TimeSyncGetNewestPpsTriggerLocalTimeUs(&ppsNewestTriggerTimeUs);
        if (djiStat != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
            USER_LOG_ERROR("get newest pps trigger time error: 0x%08llX.", djiStat);
            continue;
        }

        for (i = 0; i < DJI_TEST_POSITIONING_EVENT_COUNT; ++i) {
            eventInfo[i].eventSetIndex = s_eventIndex;
            eventInfo[i].targetPointIndex = i;

            djiStat = DjiTimeSync_TransferToAircraftTime(
                ppsNewestTriggerTimeUs - 1000000 - i * DJI_TEST_TIME_INTERVAL_AMONG_EVENTS_US, &aircraftTime);
            if (djiStat != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
                USER_LOG_ERROR("transfer to aircraft time error: 0x%08llX.", djiStat);
                continue;
            }

            eventInfo[i].eventTime = aircraftTime;
        }

        djiStat = DjiPositioning_GetPositionInformationSync(DJI_TEST_POSITIONING_EVENT_COUNT, eventInfo, positionInfo);
        if (djiStat != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
            USER_LOG_ERROR("get position information error.");
            continue;
        }

        USER_LOG_DEBUG("request position of target points success.");
        USER_LOG_DEBUG("detail position information:");
        for (i = 0; i < DJI_TEST_POSITIONING_EVENT_COUNT; ++i) {
            USER_LOG_DEBUG("position solution property: %d.", positionInfo[i].positionSolutionProperty);
            USER_LOG_DEBUG("pitchAttitudeAngle: %d\trollAttitudeAngle: %d\tyawAttitudeAngle: %d",
                           positionInfo[i].uavAttitude.pitch, positionInfo[i].uavAttitude.roll,
                           positionInfo[i].uavAttitude.yaw);
            USER_LOG_DEBUG("northPositionOffset: %d\tearthPositionOffset: %d\tdownPositionOffset: %d",
                           positionInfo[i].offsetBetweenMainAntennaAndTargetPoint.x,
                           positionInfo[i].offsetBetweenMainAntennaAndTargetPoint.y,
                           positionInfo[i].offsetBetweenMainAntennaAndTargetPoint.z);
            USER_LOG_DEBUG("longitude: %.8f\tlatitude: %.8f\theight: %.8f",
                           positionInfo[i].targetPointPosition.longitude,
                           positionInfo[i].targetPointPosition.latitude,
                           positionInfo[i].targetPointPosition.height);
            USER_LOG_DEBUG(
                "longStandardDeviation: %.8f\tlatStandardDeviation: %.8f\thgtStandardDeviation: %.8f",
                positionInfo[i].targetPointPositionStandardDeviation.longitude,
                positionInfo[i].targetPointPositionStandardDeviation.latitude,
                positionInfo[i].targetPointPositionStandardDeviation.height);
        }

        s_eventIndex++;
    }
}

#ifndef __CC_ARM
#pragma GCC diagnostic pop
#endif

/****************** (C) COPYRIGHT DJI Innovations *****END OF FILE****/
