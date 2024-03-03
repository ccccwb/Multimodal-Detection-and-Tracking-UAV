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

#include "algorithm.hpp"
#include "manager/manager.hpp"

/* Private constants ---------------------------------------------------------*/

/* Private types -------------------------------------------------------------*/

/* Private values -------------------------------------------------------------*/

/* Private functions declaration ---------------------------------------------*/

/* Exported functions definition ---------------------------------------------*/
Algorithm::Algorithm()
{
}

Algorithm::~Algorithm()
{
}

void *Algorithm::Algorithm_Task(void *arg_manager)
{
    Manager* arg = (Manager*) arg_manager;
    struct timeval start_time, stop_time;//判断是否需要Sleep一下
    bool start_duration = true;//判断是否启动计时
    uint8_t flag = 0;
    cv::Mat *image_ptr_rgb = nullptr;
    cv::Mat *image_ptr_tir = nullptr;
    
    while(1)
    {
        if(start_duration)
        {
            gettimeofday(&start_time, NULL);
            start_duration = false;//开始计时后直到完成计算前都不再次计时
        }

        arg->Share_Data.osalHandler->SemaphoreWait(arg->Share_Data.Image_Sema);
        
        if(!arg->Share_Data.New_Image || arg->Share_Data.img_ptr_rgb == nullptr) //判断图片是否空, rgb和tir应该是一起操作的
        {
            arg->Share_Data.osalHandler->SemaphorePost(arg->Share_Data.Image_Sema);
            arg->Share_Data.osalHandler->TaskSleepMs(10);//强制停一下等新图, 不能太长时间
        }
        else
        {
            image_ptr_rgb = arg->Share_Data.img_ptr_rgb;
            image_ptr_tir = arg->Share_Data.img_ptr_tir;
            arg->Share_Data.img_ptr_rgb = nullptr;
            arg->Share_Data.img_ptr_tir = nullptr;
            arg->Share_Data.osalHandler->SemaphorePost(arg->Share_Data.Image_Sema);
            

            arg->Share_Data.osalHandler->SemaphoreWait(arg->Share_Data.Flag_Bbox_Recv_Sema);
            flag = arg->Share_Data.flag_bbox_recv[0];
            cv::Rect roi(arg->Share_Data.flag_bbox_recv[1],
                    arg->Share_Data.flag_bbox_recv[2],
                    arg->Share_Data.flag_bbox_recv[3],
                    arg->Share_Data.flag_bbox_recv[4]);//0是判断位
            arg->Share_Data.osalHandler->SemaphorePost(arg->Share_Data.Flag_Bbox_Recv_Sema);

            switch (flag)
            {
            case 0:
                //吃瓜!
                arg->on_tracking = false;
                break;
            case 1:
                //检测
                // std::cout<<"执行检测"<<std::endl;
                arg->on_tracking = false;
                arg->algorithm.algorithm_detector.det(*image_ptr_rgb, &arg->Share_Data);//暂时只有可见光
                break;
            case 2:
                //跟踪
                // std::cout<<"执行跟踪"<<std::endl;
                if (arg->on_tracking)
                    arg->algorithm.algorithm_tracker.update(*image_ptr_rgb, &arg->Share_Data);
                else
                {
                    arg->algorithm.algorithm_tracker.init(*image_ptr_rgb, roi);//暂时只有可见光
                    arg->on_tracking = true;
                }              
                break;
            default:
                arg->on_tracking = false;
                USER_LOG_ERROR("判断位出现未知量, 按0处理");
                break;
            }

            // using for debug
            // cv::Mat temp_rgb = image_ptr_rgb->clone();
            // cv::cvtColor(temp_rgb, temp_rgb, cv::COLOR_RGB2BGR);
            // cv::rectangle(temp_rgb, cv::Rect(arg->Share_Data.bbox_send[0][0],arg->Share_Data.bbox_send[0][1],
            //                                     arg->Share_Data.bbox_send[0][2],arg->Share_Data.bbox_send[0][3]), cv::Scalar(255, 0, 0), 2);
            // cv::imshow("RGB", temp_rgb);
            // cv::waitKey(1);

            delete image_ptr_rgb;
            delete image_ptr_tir;

            gettimeofday(&stop_time, NULL);
            
            uint32_t duration = Get_Duration_Time_ms(start_time,stop_time); //ms
            // USER_LOG_INFO("Detection Time%f", duration);//ms

            int32_t sleep_time = 1000 / ALGORITHM_RUN_FREQ - duration - 1;//假定判断是否sleep的操作要1ms hhh
            start_duration = true;//重新开始下一阶段的计时

            if (sleep_time > 0)
                arg->Share_Data.osalHandler->TaskSleepMs(sleep_time);
        }


    }

}

/****************** (C) COPYRIGHT DJI Innovations *****END OF FILE****/