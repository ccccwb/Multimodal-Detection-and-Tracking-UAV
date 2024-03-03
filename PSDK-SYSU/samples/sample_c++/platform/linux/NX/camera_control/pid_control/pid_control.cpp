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
#include "pid_control.hpp"


/* Private constants ---------------------------------------------------------*/

/* Private types -------------------------------------------------------------*/

/* Private values -------------------------------------------------------------*/

/* Private functions declaration ---------------------------------------------*/
dji_f32_t PID_Control::pid_update(int16_t error ){
    dji_f32_t CP = error;

    dji_f32_t deltaError = error - Pre_Error;
    dji_f32_t CD = (deltaError / DELTA_TIME);
    Sum_Error += error;
    dji_f32_t CI = Sum_Error*DELTA_TIME;
    Pre_Error = error; //更新
    
    return (KP*CP + KI*CI + KD*CD);


}
dji_f32_t PID_Control::pid_init(){
    Pre_Error = 0;
    Sum_Error = 0;

}
PID_Control::PID_Control(){

}
PID_Control::~PID_Control(){

}
PID_Control::PID_Control(dji_f32_t p, dji_f32_t i, dji_f32_t d){
    KP = p;
    KD = d;
    KI = i;
}



/* Exported functions definition ---------------------------------------------*/

/* Private functions definition-----------------------------------------------*/

/****************** (C) COPYRIGHT DJI Innovations *****END OF FILE****/
