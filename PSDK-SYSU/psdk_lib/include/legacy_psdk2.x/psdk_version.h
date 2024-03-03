/**
 ********************************************************************
 * @file    psdk_version.h
 * @brief   This is the header file for "psdk_version.c", defining the structure and
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
#ifndef PSDK_VERSION_H
#define PSDK_VERSION_H

/* Includes ------------------------------------------------------------------*/
#include <dji_version.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Exported constants --------------------------------------------------------*/
#define PSDK_VERSION_MAJOR     DJI_VERSION_MAJOR /*!< Payload SDK major version num, when have incompatible API changes. Range from 0 to 99. */
#define PSDK_VERSION_MINOR     DJI_VERSION_MINOR /*!< Payload SDK minor version num, when add functionality in a backwards compatible manner changes. Range from 0 to 99. */
#define PSDK_VERSION_MODIFY    DJI_VERSION_MODIFY /*!< Payload SDK modify version num, when have backwards compatible bug fixes changes. Range from 0 to 99. */
#define PSDK_VERSION_BETA      DJI_VERSION_BETA /*!< Payload SDK version beta info, release version will be 0, when beta version release changes. Range from 0 to 255. */
#define PSDK_VERSION_BUILD     DJI_VERSION_BUILD /*!< Payload SDK version build info, when jenkins trigger build changes. Range from 0 to 65535. */

/* Exported types ------------------------------------------------------------*/

/* Exported functions --------------------------------------------------------*/


#ifdef __cplusplus
}
#endif

#endif // PSDK_VERSION_H
/************************ (C) COPYRIGHT DJI Innovations *******END OF FILE******/
