/**
 ********************************************************************
 * @file    dji_camera_stream_decoder.cpp
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
#include "dji_camera_stream_decoder.hpp"
#include "unistd.h"
#include "pthread.h"
#include "dji_logger.h"
// #include <iostream>
/* Private constants ---------------------------------------------------------*/

/* Private types -------------------------------------------------------------*/

/* Private values -------------------------------------------------------------*/

/* Private functions declaration ---------------------------------------------*/

/* Exported functions definition ---------------------------------------------*/
DJICameraStreamDecoder::DJICameraStreamDecoder()
    : initSuccess(false),
      cbThreadIsRunning(false),
      cbThreadStatus(-1),
      cb(nullptr),
      cbUserParam(nullptr),
      pCodecCtx(nullptr),
      pCodec(nullptr),
      pCodecParserCtx(nullptr),
      pSwsCtx(nullptr),
      pFrameYUV(nullptr),
      pFrameRGB(nullptr),
      rgbBuf(nullptr),
      bufSize(0)
{
    pthread_mutex_init(&decodemutex, nullptr);
}

DJICameraStreamDecoder::~DJICameraStreamDecoder()
{
    pthread_mutex_destroy(&decodemutex);
    if(cb)
    {
        registerCallback(nullptr, nullptr);
    }

    cleanup();
}

bool DJICameraStreamDecoder::init()
{
    pthread_mutex_lock(&decodemutex);

    // 防止重复初始化？ lz写的不是很懂
    if (true == initSuccess) {
        USER_LOG_INFO("Decoder already initialized.\n");
        return true;
    }

    // 注册所有解码器  //这两个应该是一样的吧？
    av_register_all();
    avcodec_register_all();

    // Idr_flag 用来判断要不要 转成Idr格式， H20输出的H264是BDR格式，硬解码的解码器没办法直接用，需要先转成IDR格式
    Idr_flag = 1;
    // AV_CODEC_ID_H264 是软解码器， h264_nvmpi是硬解码器， 一个大佬写的（好像是英伟达员工）
    pCodec = avcodec_find_decoder(AV_CODEC_ID_H264);
    // pCodec = avcodec_find_decoder_by_name("h264_nvmpi"); //27?
    if(!pCodec){
        USER_LOG_ERROR("InitVideoCodec 获取解码器出错");
        return false;
    }

    // 解码器上下文环境
    pCodecCtx = avcodec_alloc_context3(pCodec);
    if (!pCodecCtx) {
        return false;
    }
    pCodecCtx->thread_count = 4;//// 这个必须要， 不然画面会很卡！！！！！！！！！！！

    // 因为我们是从 databuff直接解码，而不是流数据，所以自己来创建一个视频流用来初始化编码器格式
    AVFormatContext* pFormatCtx = avformat_alloc_context();
    if (!pFormatCtx) {
        printf("Failed to allocate format context\n");
        return false;
    }
    // Create video stream and set codec parameters
    AVStream* pStream = avformat_new_stream(pFormatCtx, NULL);
    if (!pStream) {
        printf("Failed to create stream\n");
        return false;
    }
    // 设置参数， 注意图像的尺寸
    AVCodecParameters* pCodecParams = pStream->codecpar;
    pCodecParams->codec_type = AVMEDIA_TYPE_VIDEO;
    pCodecParams->codec_id = AV_CODEC_ID_H264;
    pCodecParams->width = 1920;
    pCodecParams->height = 1080;
    // Set other codec parameters as needed
    if (avcodec_parameters_to_context(pCodecCtx, pCodecParams) < 0) {
        printf("Failed to set codec parameters\n");
        return -1;
    }

    // 打开解码器
    if (!pCodec || avcodec_open2(pCodecCtx, pCodec, NULL) < 0) {
        USER_LOG_ERROR("打开解码器失败");
        return false;
    }

    // av_parser_init初始化AVCodecParserContext, AVCodecParser用于解析输入的数据流并把它们分成一帧一帧的压缩编码数据
    pCodecParserCtx = av_parser_init(pCodec->id);
    if (!pCodecParserCtx) {
        USER_LOG_ERROR("初始化AVCodecParserContex 失败");
        return false;
    }

    pFrameYUV = av_frame_alloc();
    if (!pFrameYUV) {
        return false;
    }

    pFrameRGB = av_frame_alloc();
    if (!pFrameRGB) {
        return false;
    }

    pSwsCtx = nullptr;

    pCodecCtx->flags2 |= AV_CODEC_FLAG2_SHOW_ALL;
    initSuccess = true;

    pthread_mutex_unlock(&decodemutex);

    return true;
}

void DJICameraStreamDecoder::cleanup()
{
    pthread_mutex_lock(&decodemutex);

    initSuccess = false;
    if (nullptr != pSwsCtx) {
        sws_freeContext(pSwsCtx);
        pSwsCtx = nullptr;
    }

    if (nullptr != pFrameYUV) {
        av_free(pFrameYUV);
        pFrameYUV = nullptr;
    }

    if (nullptr != pCodecParserCtx) {
        av_parser_close(pCodecParserCtx);
        pCodecParserCtx = nullptr;
    }

    if (nullptr != pCodec) {
        avcodec_close(pCodecCtx);
        pCodec = nullptr;
    }

    if (nullptr != pCodecCtx) {
        av_free(pCodecCtx);
        pCodecCtx = nullptr;
    }

    if (nullptr != rgbBuf) {
        av_free(rgbBuf);
        rgbBuf = nullptr;
    }

    if (nullptr != pFrameRGB) {
        av_free(pFrameRGB);
        pFrameRGB = nullptr;
    }

    pthread_mutex_unlock(&decodemutex);
}

void *DJICameraStreamDecoder::callbackThreadEntry(void *p)
{
    //DSTATUS_PRIVATE("****** Decoder Callback Thread Start ******\n");
    usleep(50 * 1000);
    static_cast<DJICameraStreamDecoder *>(p)->callbackThreadFunc();
    return nullptr;
}

void DJICameraStreamDecoder::callbackThreadFunc()
{
    while (cbThreadIsRunning) {
        CameraRGBImage copyOfImage;
        if (!decodedImageHandler.getNewImageWithLock(copyOfImage, 1000)) {
            //DDEBUG_PRIVATE("Decoder Callback Thread: Get image time out\n");
            continue;
        }

        if (cb) {
            (*cb)(copyOfImage, cbUserParam);
        }
    }
}

//试一下移除SEI帧来避免ffmpeg的报错,  照抄大疆创新论坛
int DJICameraStreamDecoder::RemoveSEINal(uint8_t *pData, const uint8_t *buf, int bufLen, bool *IsKeyFrame)
{
    //没太能理解KeyFrame用法, 放弃使用..., 看论坛上另一个文章（同作者好像）是用来队列清空缓存用的, 因为关键帧(IDR)吧
    unsigned char naluType;
    uint32_t datalen = 0;
    bool drop = false;
    // int NalNum = 0;
    // std::cout<< bufLen << std::endl;
    for(int i = 0; i < bufLen; i++)
    {
        //找一个新的NALU单元就判断他的类型, 三字节起始码判断不会被这个drop影响(一帧的第一个slice也是四字节的)
        if (buf[i]==0 && buf[i+1] == 0 && buf[i+2] == 0 && buf[i+3] == 1)//0x00 00 00 01 是 SEI SPS PPS 等NALU单元的起始码
        {
            naluType = buf[i+4] & 0x1F;//后5位用于判断type
            // std::cout<<"test RemoveSEINal:naluType"<<naluType<<std::endl;
            switch (naluType)
            {
            case 0x06://SEI
                drop = true;
                break;
            // case 0x07://SPS
            // case 0x08://PPS
            //     // std::cout<< i << " NalNum++" << std::endl;
            //     NalNum++;//具我查到的资料所知, SPS和PPS好像在IDR帧之前, GOP开始, 是一个编码序列的头, 所以找到他们就是关键帧???
            //     drop = false;
            //     break;
            default:
                drop = false;
                break;
            }
        }

        if(drop)
            continue;
        else
        {
        //     if (NalNum > 0)
        //     {
        //         *IsKeyFrame = true;
        //     }
        //     else
        //         *IsKeyFrame = false;
            *pData++ = buf[i];
            datalen++;
        }
    }
    return datalen;
}

void DJICameraStreamDecoder::decodeBuffer(const uint8_t *buf, int bufLen)
{
    struct timeval start_time, stop_time;//判断是否需要Sleep一下
    gettimeofday(&start_time, NULL);



    uint8_t pData_array[bufLen] = {0};
    bool IsKeyFrame = false; //判断是否关键帧, 暂时用不上
    int remainingLen = RemoveSEINal(pData_array, buf, bufLen, &IsKeyFrame);
    
    const uint8_t *pData = pData_array;
    
    int processedLen = 0;

    pthread_mutex_lock(&decodemutex);
    
    AVPacket pkt;
    av_init_packet(&pkt);
    if(Idr_flag){
        const uint8_t *idrFrameDataPtr;
	    uint32_t idrFrameDataLen;
		//Need push idr frame to hardware decoder to identify h264 stream, because dji camera h264 steam use GDR
		if (!H264IdrFrame_GetData(IDR_FRAME_TYPE_H20_RECORD_VIDEO, idrFrameDataPtr, idrFrameDataLen)) {
			USER_LOG_ERROR("Can't get IDR Frame, resolution enum value = %lld.", IDR_FRAME_TYPE_H20_RECORD_VIDEO);
			return;
		}
        Idr_flag = 0;
        pData = idrFrameDataPtr;
        remainingLen = idrFrameDataLen;
	} 

    while (remainingLen > 0) {
        if (!pCodecParserCtx || !pCodecCtx) {
            USER_LOG_ERROR("Invalid decoder ctx.");
            break;
        }
        //av_parser_parse2 解析数据获得一个Packet， 从输入的数据流中分离出一帧一帧的压缩编码数据。
        processedLen = av_parser_parse2(pCodecParserCtx, pCodecCtx,
                                        &pkt.data, &pkt.size,
                                        pData, remainingLen,
                                        AV_NOPTS_VALUE, AV_NOPTS_VALUE, AV_NOPTS_VALUE);
        // USER_LOG_INFO("aprocessedLen %d %d %d", processedLen, pkt.size, remainingLen);
        remainingLen -= processedLen;
        pData += processedLen;
        if (pkt.size > 0) {

            
            // 输入一个压缩编码的结构体AVPacket，输出一个解码后的结构体AVFrame
            int gotPicture = 0;
            avcodec_decode_video2(pCodecCtx, pFrameYUV, &gotPicture, &pkt);

            if (!gotPicture) {
                // USER_LOG_ERROR("Got Frame, but no picture\n");
                continue;
            } else {
                // USER_LOG_INFO("Got picture\n");
                int w = pFrameYUV->width;
                int h = pFrameYUV->height;
                // USER_LOG_INFO("width%d height %d", w, h);

                ////DSTATUS_PRIVATE("Got picture! size=%dx%d\n", w, h);

                if (nullptr == pSwsCtx) {
                    pSwsCtx = sws_getContext(w, h, pCodecCtx->pix_fmt,
                                             w, h, AV_PIX_FMT_RGB24,
                                             4, nullptr, nullptr, nullptr);
                }

                if (nullptr == rgbBuf) {
                    bufSize = avpicture_get_size(AV_PIX_FMT_RGB24, w, h);
                    rgbBuf = (uint8_t *) av_malloc(bufSize);
                    avpicture_fill((AVPicture *) pFrameRGB, rgbBuf, AV_PIX_FMT_RGB24, w, h);
                }

                if (nullptr != pSwsCtx && nullptr != rgbBuf) {
                    sws_scale(pSwsCtx,
                              (uint8_t const *const *) pFrameYUV->data, pFrameYUV->linesize, 0, pFrameYUV->height,
                              pFrameRGB->data, pFrameRGB->linesize);

                    pFrameRGB->height = h;
                    pFrameRGB->width = w;

                    decodedImageHandler.writeNewImageWithLock(pFrameRGB->data[0], bufSize, w, h);
                }
            }
        }
    }
    pthread_mutex_unlock(&decodemutex);
    av_free_packet(&pkt);


    // for debug
    // gettimeofday(&stop_time, NULL);
    // duration = (duration + Get_Duration_Time_ms(start_time, stop_time));
    // duration_times++;
    // if(duration_times == 29)
    // {
    //     duration = 0;
    //     duration_times = 0;
    //     USER_LOG_INFO("Decoder Time%f", duration/30);//ms
    // }
}

bool DJICameraStreamDecoder::registerCallback(CameraImageCallback f, void *param)
{
    cb = f;
    cbUserParam = param;

    /* When users register a non-nullptr callback, we will start the callback thread. */
    if (nullptr != cb) {
        if (!cbThreadIsRunning) {
            cbThreadStatus = pthread_create(&callbackThread, nullptr, callbackThreadEntry, this);
            if (0 == cbThreadStatus) {
                //DSTATUS_PRIVATE("User callback thread created successfully!\n");
                cbThreadIsRunning = true;
                return true;
            } else {
                //DERROR_PRIVATE("User called thread creation failed!\n");
                cbThreadIsRunning = false;
                return false;
            }
        } else {
            //DERROR_PRIVATE("Callback thread already running!\n");
            return true;
        }
    } else {
        if (cbThreadStatus == 0) {
            cbThreadIsRunning = false;
            pthread_join(callbackThread, nullptr);
            cbThreadStatus = -1;
        }
        return true;
    }
}

/* Private functions definition-----------------------------------------------*/

/****************** (C) COPYRIGHT DJI Innovations *****END OF FILE****/






