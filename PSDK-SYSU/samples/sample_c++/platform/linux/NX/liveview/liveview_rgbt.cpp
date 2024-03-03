/**
 ********************************************************************
 * @file    test_liveview.cpp
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

#include "dji_logger.h"
#include "dji_camera_manager.h"

#include "liveview_rgbt.hpp"
#include "utils/utils.hpp"

/* Private constants ---------------------------------------------------------*/

/* Private types -------------------------------------------------------------*/

/* Private values -------------------------------------------------------------*/

/* Private functions declaration ---------------------------------------------*/

static DJICameraStreamDecoder* Stream_Decoder;
static void Liveview_Convert_H264ToRgb_Callback(E_DjiLiveViewCameraPosition position, const uint8_t *buf, uint32_t bufLen);

static cv::Rect* rect_tir;
static cv::Rect* rect_rgb;

/* Exported functions definition ---------------------------------------------*/
Liveview_RGBT::Liveview_RGBT()
{
    T_DjiReturnCode returnCode;

    returnCode = DjiLiveview_Init();
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        throw std::runtime_error("Liveview init failed");
    }
    Stream_Decoder = new DJICameraStreamDecoder();

    //xywh
    rect_tir = new cv::Rect(0, 156, 960, 768);//xywh
    rect_rgb = new cv::Rect(960, 156, 960, 768);
}

Liveview_RGBT::~Liveview_RGBT()
{
    T_DjiReturnCode returnCode;

    returnCode = DjiLiveview_StopH264Stream(DJI_LIVEVIEW_CAMERA_POSITION_NO_1, DJI_LIVEVIEW_CAMERA_SOURCE_H20T_IR);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        throw std::runtime_error("Liveview stop failed");
    }
    
    Stream_Decoder->cleanup();

    returnCode = DjiLiveview_Deinit();
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        throw std::runtime_error("Liveview deinit failed");
    }

    delete Stream_Decoder;
    delete rect_tir;
    delete rect_rgb;
}

void Liveview_RGBT::Write_RGBT_Callback(CameraRGBImage img, void *userData_)
{   
    T_ShareData* userData = (T_ShareData*) userData_;
    //红外960*768 + 可见光960*768 宽度拼接后居中于 1920*1080
    cv::Mat image(img.height, img.width, CV_8UC3, img.rawData.data());

    userData->osalHandler->SemaphoreWait(userData->Image_Sema);
    if (img.height != IMAGE_HEIGHT || img.width != IMAGE_WIDTH)
    {
        USER_LOG_ERROR("image size not (1920,1080) but (%d, %d) , img_ptr_* set to nullptr, may not set record mode !",img.height,img.width);
        userData->img_ptr_rgb = nullptr;
        userData->img_ptr_tir = nullptr;
        return;
    } 

    if (userData->img_ptr_tir != nullptr)
        delete userData->img_ptr_tir;
    if (userData->img_ptr_rgb != nullptr)
        delete userData->img_ptr_rgb;

    userData->img_ptr_tir = new cv::Mat(image, *rect_tir);
    userData->img_ptr_rgb = new cv::Mat(image, *rect_rgb);
    userData->New_Image = true;

    userData->osalHandler->SemaphorePost(userData->Image_Sema);

    // using for debug
    // cv::Mat temp_tir = userData->img_ptr_tir->clone();
    // cv::Mat temp_rgb = userData->img_ptr_rgb->clone();

    // cv::cvtColor(temp_tir, temp_tir, cv::COLOR_RGB2BGR);
    // cv::cvtColor(temp_rgb, temp_rgb, cv::COLOR_RGB2BGR);
    // cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    // cv::imshow("raw", image);
    // cv::imshow("TIR", temp_tir);
    // cv::imshow("RGB", temp_rgb);
    // cv::waitKey(1);

    return;
}

T_DjiReturnCode Liveview_RGBT::Start_Camera_Stream(void *userData)
{
    Stream_Decoder->init();
    Stream_Decoder->registerCallback(Write_RGBT_Callback, userData);

    T_DjiCameraManagerOpticalZoomParam opticalZoomParam;
    T_DjiReturnCode returnCode;
    dji_f32_t factor = 2.0;

    // returnCode = DjiCameraManager_GetOpticalZoomParam(DJI_MOUNT_POSITION_PAYLOAD_PORT_NO1, &opticalZoomParam);
    // if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS &&
    //     returnCode != DJI_ERROR_CAMERA_MANAGER_MODULE_CODE_UNSUPPORTED_COMMAND) {
    //     USER_LOG_ERROR("Get mounted position 01 camera's zoom param failed, error code :0x%08X",
    //                     returnCode);
    // }

    // if (opticalZoomParam.currentOpticalZoomFactor < factor)
    //     returnCode = DjiCameraManager_SetOpticalZoomParam(DJI_MOUNT_POSITION_PAYLOAD_PORT_NO1, DJI_CAMERA_ZOOM_DIRECTION_IN, factor);
    //     if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS &&
    //         returnCode != DJI_ERROR_CAMERA_MANAGER_MODULE_CODE_UNSUPPORTED_COMMAND) {
    //         USER_LOG_ERROR("Set mounted position 01 camera's zoom factor(%0.1f) failed, error code :0x%08X",
    //                     factor, returnCode);
    //     }
    // else if (opticalZoomParam.currentOpticalZoomFactor > factor)
    //     returnCode = DjiCameraManager_SetOpticalZoomParam(DJI_MOUNT_POSITION_PAYLOAD_PORT_NO1, DJI_CAMERA_ZOOM_DIRECTION_OUT, factor);
    //     if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS &&
    //         returnCode != DJI_ERROR_CAMERA_MANAGER_MODULE_CODE_UNSUPPORTED_COMMAND) {
    //         USER_LOG_ERROR("Set mounted position 01 camera's zoom factor(%0.1f) failed, error code :0x%08X",
    //                     factor, returnCode);
    //     }
    // else
    // {
    //     //do nothing
    // }

    //设置录像模式, 应当由遥控器切换来作保证
    returnCode = DjiCameraManager_SetMode(DJI_MOUNT_POSITION_PAYLOAD_PORT_NO1, DJI_CAMERA_MANAGER_WORK_MODE_RECORD_VIDEO);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS &&
        returnCode != DJI_ERROR_CAMERA_MANAGER_MODULE_CODE_UNSUPPORTED_COMMAND) 
    {
        USER_LOG_ERROR("set mounted position 01 camera's work mode as shoot-photo mode failed,"
                    " error code :0x%08X", returnCode);
        return returnCode;
    }
    //DJI_LIVEVIEW_CAMERA_SOURCE_H20T_IR 选取IR流, 由遥控器切换到分屏模式
    return DjiLiveview_StartH264Stream(DJI_LIVEVIEW_CAMERA_POSITION_NO_1, DJI_LIVEVIEW_CAMERA_SOURCE_H20T_IR,
                                        Liveview_Convert_H264ToRgb_Callback);

}


/* Private functions definition-----------------------------------------------*/
static void Liveview_Convert_H264ToRgb_Callback(E_DjiLiveViewCameraPosition position, const uint8_t *buf, uint32_t bufLen)
{
    Stream_Decoder->decodeBuffer(buf, bufLen);
}

/****************** (C) COPYRIGHT DJI Innovations *****END OF FILE****/
