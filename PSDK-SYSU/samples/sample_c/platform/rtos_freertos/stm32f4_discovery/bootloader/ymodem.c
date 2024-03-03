/**
  ******************************************************************************
  * @file    IAP/IAP_Main/Src/ymodem.c 
  * @author  MCD Application Team
  * @brief   This file provides all the software functions related to the ymodem 
  *          protocol.
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2017 STMicroelectronics International N.V. 
  * All rights reserved.</center></h2>
  *
  * Redistribution and use in source and binary forms, with or without 
  * modification, are permitted, provided that the following conditions are met:
  *
  * 1. Redistribution of source code must retain the above copyright notice, 
  *    this list of conditions and the following disclaimer.
  * 2. Redistributions in binary form must reproduce the above copyright notice,
  *    this list of conditions and the following disclaimer in the documentation
  *    and/or other materials provided with the distribution.
  * 3. Neither the name of STMicroelectronics nor the names of other 
  *    contributors to this software may be used to endorse or promote products 
  *    derived from this software without specific written permission.
  * 4. This software, including modifications and/or derivative works of this 
  *    software, must execute solely and exclusively on microcontroller or
  *    microprocessor devices manufactured by or for STMicroelectronics.
  * 5. Redistribution and use of this software other than as permitted under 
  *    this license is void and will automatically terminate your rights under 
  *    this license. 
  *
  * THIS SOFTWARE IS PROVIDED BY STMICROELECTRONICS AND CONTRIBUTORS "AS IS" 
  * AND ANY EXPRESS, IMPLIED OR STATUTORY WARRANTIES, INCLUDING, BUT NOT 
  * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
  * PARTICULAR PURPOSE AND NON-INFRINGEMENT OF THIRD PARTY INTELLECTUAL PROPERTY
  * RIGHTS ARE DISCLAIMED TO THE FULLEST EXTENT PERMITTED BY LAW. IN NO EVENT 
  * SHALL STMICROELECTRONICS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, 
  * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
  * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
  * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
  * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  *
  * @statement DJI has modified some symbols' name.
  *
  ******************************************************************************
  */
/** @addtogroup STM32F4xx_IAP_Main
  * @{
  */

/* Includes ------------------------------------------------------------------*/
#include "flash_if.h"
#include "common.h"
#include "string.h"
#include "main.h"
#include "menu.h"
#include "uart.h"
#include "osal.h"

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Packet structure defines */
#define PACKET_HEADER_SIZE      ((uint32_t)3)
#define PACKET_DATA_INDEX       ((uint32_t)4)
#define PACKET_START_INDEX      ((uint32_t)1)
#define PACKET_NUMBER_INDEX     ((uint32_t)2)
#define PACKET_CNUMBER_INDEX    ((uint32_t)3)
#define PACKET_TRAILER_SIZE     ((uint32_t)2)
#define PACKET_OVERHEAD_SIZE    (PACKET_HEADER_SIZE + PACKET_TRAILER_SIZE - 1)
#define PACKET_SIZE             ((uint32_t)128)
#define PACKET_1K_SIZE          ((uint32_t)1024)

/* /-------- Packet in IAP memory ------------------------------------------\
 * | 0      |  1    |  2     |  3   |  4      | ... | n+4     | n+5  | n+6  |
 * |------------------------------------------------------------------------|
 * | unused | start | number | !num | data[0] | ... | data[n] | crc0 | crc1 |
 * \------------------------------------------------------------------------/
 * the first byte is left unused for memory alignment reasons                 */
#define FILE_SIZE_LENGTH        ((uint32_t)16)

#define SOH                     ((uint8_t)0x01)  /* start of 128-byte data packet */
#define STX                     ((uint8_t)0x02)  /* start of 1024-byte data packet */
#define EOT                     ((uint8_t)0x04)  /* end of transmission */
#define ACK                     ((uint8_t)0x06)  /* acknowledge */
#define NAK                     ((uint8_t)0x15)  /* negative acknowledge */
#define CA                      ((uint32_t)0x18) /* two of these in succession aborts transfer */
#define NEGATIVE_BYTE           ((uint8_t)0xFF)

#define ABORT1                  ((uint8_t)0x41)  /* 'A' == 0x41, abort by user */
#define ABORT2                  ((uint8_t)0x61)  /* 'a' == 0x61, abort by user */

#define NAK_TIMEOUT             ((uint32_t)0x100000)
#define DOWNLOAD_TIMEOUT        ((uint32_t)5000) /* Five second retry delay */
#define MAX_ERRORS              ((uint32_t)5)
#define CRC16_F       /* activate the CRC16 integrity */
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
__IO uint32_t flashdestination;
/* @note ATTENTION - please keep this variable 32bit alligned */
uint8_t aPacketData[PACKET_1K_SIZE + PACKET_DATA_INDEX + PACKET_TRAILER_SIZE];

/* Private function prototypes -----------------------------------------------*/
static void PrepareIntialPacket(uint8_t *p_data, const uint8_t *p_file_name, uint32_t length);
static void PreparePacket(uint8_t *p_source, uint8_t *p_packet, uint8_t pkt_nr, uint32_t size_blk);
static HAL_StatusTypeDef ReceivePacket(uint8_t *p_data, uint32_t *p_length, uint32_t timeout);
uint16_t UpdateCRC16(uint16_t crc_in, uint8_t byte);
uint16_t Cal_CRC16(const uint8_t *p_data, uint32_t size);
uint8_t CalcChecksum(const uint8_t *p_data, uint32_t size);

HAL_StatusTypeDef Uart_ReadWithTimeOut(E_UartNum uartNum, uint8_t *data, uint16_t len, uint32_t timeOut);
HAL_StatusTypeDef Uart_WriteWithTimeOut(E_UartNum uartNum, uint8_t *data, uint16_t len, uint32_t timeOut);

/* Private functions ---------------------------------------------------------*/

/**
  * @brief  Receive a packet from sender
  * @param  data
  * @param  length
  *     0: end of transmission
  *     2: abort by sender
  *    >0: packet length
  * @param  timeout
  * @retval HAL_OK: normally return
  *         HAL_BUSY: abort by user
  */
static HAL_StatusTypeDef ReceivePacket(uint8_t *p_data, uint32_t *p_length, uint32_t timeout)
{
    uint32_t crc;
    uint32_t packet_size = 0;
    HAL_StatusTypeDef status;
    uint8_t char1;

    *p_length = 0;
    status = Uart_ReadWithTimeOut(PSDK_CONSOLE_UART_NUM, &char1, 1, timeout);

    if (status == HAL_OK) {
        switch (char1) {
            case SOH:
                packet_size = PACKET_SIZE;
                break;
            case STX:
                packet_size = PACKET_1K_SIZE;
                break;
            case EOT:
                break;
            case CA:
                if ((Uart_ReadWithTimeOut(PSDK_CONSOLE_UART_NUM, &char1, 1, timeout) == HAL_OK) && (char1 == CA)) {
                    packet_size = 2;
                } else {
                    status = HAL_ERROR;
                }
                break;
            case ABORT1:
            case ABORT2:
                status = HAL_BUSY;
                break;
            default:
                status = HAL_ERROR;
                break;
        }
        *p_data = char1;

        if (packet_size >= PACKET_SIZE) {
            status = Uart_ReadWithTimeOut(PSDK_CONSOLE_UART_NUM, &p_data[PACKET_NUMBER_INDEX],
                                          packet_size + PACKET_OVERHEAD_SIZE,
                                          timeout);

            /* Simple packet sanity check */
            if (status == HAL_OK) {
                if (p_data[PACKET_NUMBER_INDEX] != ((p_data[PACKET_CNUMBER_INDEX]) ^ NEGATIVE_BYTE)) {
                    packet_size = 0;
                    status = HAL_ERROR;
                } else {
                    /* Check packet CRC */
                    crc = p_data[packet_size + PACKET_DATA_INDEX] << 8;
                    crc += p_data[packet_size + PACKET_DATA_INDEX + 1];
                    if (Cal_CRC16(&p_data[PACKET_DATA_INDEX], packet_size) != crc) {
                        packet_size = 0;
                        status = HAL_ERROR;
                    }
                }
            } else {
                packet_size = 0;
            }
        }
    }
    *p_length = packet_size;
    return status;
}

/**
  * @brief  Prepare the first block
  * @param  p_data:  output buffer
  * @param  p_file_name: name of the file to be sent
  * @param  length: length of the file to be sent in bytes
  * @retval None
  */
static void PrepareIntialPacket(uint8_t *p_data, const uint8_t *p_file_name, uint32_t length)
{
    uint32_t i, j = 0;
    uint8_t astring[10];

    /* first 3 bytes are constant */
    p_data[PACKET_START_INDEX] = SOH;
    p_data[PACKET_NUMBER_INDEX] = 0x00;
    p_data[PACKET_CNUMBER_INDEX] = 0xff;

    /* Filename written */
    for (i = 0; (p_file_name[i] != '\0') && (i < FILE_NAME_LENGTH); i++) {
        p_data[i + PACKET_DATA_INDEX] = p_file_name[i];
    }

    p_data[i + PACKET_DATA_INDEX] = 0x00;

    /* file size written */
    Int2Str(astring, length);
    i = i + PACKET_DATA_INDEX + 1;
    while (astring[j] != '\0') {
        p_data[i++] = astring[j++];
    }

    /* padding with zeros */
    for (j = i; j < PACKET_SIZE + PACKET_DATA_INDEX; j++) {
        p_data[j] = 0;
    }
}

/**
  * @brief  Prepare the data packet
  * @param  p_source: pointer to the data to be sent
  * @param  p_packet: pointer to the output buffer
  * @param  pkt_nr: number of the packet
  * @param  size_blk: length of the block to be sent in bytes
  * @retval None
  */
static void PreparePacket(uint8_t *p_source, uint8_t *p_packet, uint8_t pkt_nr, uint32_t size_blk)
{
    uint8_t *p_record;
    uint32_t i, size, packet_size;

    /* Make first three packet */
    packet_size = size_blk >= PACKET_1K_SIZE ? PACKET_1K_SIZE : PACKET_SIZE;
    size = size_blk < packet_size ? size_blk : packet_size;
    if (packet_size == PACKET_1K_SIZE) {
        p_packet[PACKET_START_INDEX] = STX;
    } else {
        p_packet[PACKET_START_INDEX] = SOH;
    }
    p_packet[PACKET_NUMBER_INDEX] = pkt_nr;
    p_packet[PACKET_CNUMBER_INDEX] = (~pkt_nr);
    p_record = p_source;

    /* Filename packet has valid data */
    for (i = PACKET_DATA_INDEX; i < size + PACKET_DATA_INDEX; i++) {
        p_packet[i] = *p_record++;
    }
    if (size <= packet_size) {
        for (i = size + PACKET_DATA_INDEX; i < packet_size + PACKET_DATA_INDEX; i++) {
            p_packet[i] = 0x1A; /* EOF (0x1A) or 0x00 */
        }
    }
}

/**
  * @brief  Update CRC16 for input byte
  * @param  crc_in input value 
  * @param  input byte
  * @retval None
  */
uint16_t UpdateCRC16(uint16_t crc_in, uint8_t byte)
{
    uint32_t crc = crc_in;
    uint32_t in = byte | 0x100;

    do {
        crc <<= 1;
        in <<= 1;
        if (in & 0x100)
            ++crc;
        if (crc & 0x10000)
            crc ^= 0x1021;
    } while (!(in & 0x10000));

    return crc & 0xffffu;
}

/**
  * @brief  Cal CRC16 for YModem Packet
  * @param  data
  * @param  length
  * @retval None
  */
uint16_t Cal_CRC16(const uint8_t *p_data, uint32_t size)
{
    uint32_t crc = 0;
    const uint8_t *dataEnd = p_data + size;

    while (p_data < dataEnd)
        crc = UpdateCRC16(crc, *p_data++);

    crc = UpdateCRC16(crc, 0);
    crc = UpdateCRC16(crc, 0);

    return crc & 0xffffu;
}

/**
  * @brief  Calculate Check sum for YModem Packet
  * @param  p_data Pointer to input data
  * @param  size length of input data
  * @retval uint8_t checksum value
  */
uint8_t CalcChecksum(const uint8_t *p_data, uint32_t size)
{
    uint32_t sum = 0;
    const uint8_t *p_data_end = p_data + size;

    while (p_data < p_data_end) {
        sum += *p_data++;
    }

    return (sum & 0xffu);
}

/* Public functions ---------------------------------------------------------*/
/**
  * @brief  Receive a file using the ymodem protocol with CRC16.
  * @param  p_size The size of the file.
  * @retval COM_StatusTypeDef result of reception/programming
  */
COM_StatusTypeDef Ymodem_Receive(uint32_t *p_size)
{
    uint32_t i, packet_length, session_done = 0, file_done, errors = 0, session_begin = 0;
    // uint32_t flashdestination;
    uint32_t ramsource, filesize, packets_received;
    uint8_t *file_ptr;
    uint8_t file_size[FILE_SIZE_LENGTH], tmp;
    COM_StatusTypeDef result = COM_OK;

    /* Initialize flashdestination variable */
    flashdestination = APPLICATION_ADDRESS;

    while ((session_done == 0) && (result == COM_OK)) {
        packets_received = 0;
        file_done = 0;
        while ((file_done == 0) && (result == COM_OK)) {
            switch (ReceivePacket(aPacketData, &packet_length, DOWNLOAD_TIMEOUT)) {
                case HAL_OK:
                    errors = 0;
                    switch (packet_length) {
                        case 2:
                            /* Abort by sender */
                            Serial_PutByte(ACK);
                            result = COM_ABORT;
                            break;
                        case 0:
                            /* End of transmission */
                            Serial_PutByte(ACK);
                            file_done = 1;
                            break;
                        default:
                            /* Normal packet */
                            if (aPacketData[PACKET_NUMBER_INDEX] != (uint8_t) packets_received) {
                                Serial_PutByte(NAK);
                            } else {
                                if (packets_received == 0) {
                                    /* File name packet */
                                    if (aPacketData[PACKET_DATA_INDEX] != 0) {
                                        /* File name extraction */
                                        i = 0;
                                        file_ptr = aPacketData + PACKET_DATA_INDEX;
                                        while ((*file_ptr != 0) && (i < FILE_NAME_LENGTH)) {
                                            aFileName[i++] = *file_ptr++;
                                        }

                                        /* File size extraction */
                                        aFileName[i++] = '\0';
                                        i = 0;
                                        file_ptr++;
                                        while ((*file_ptr != ' ') && (i < FILE_SIZE_LENGTH)) {
                                            file_size[i++] = *file_ptr++;
                                        }
                                        file_size[i++] = '\0';
                                        Str2Int(file_size, &filesize);

                                        /* Test the size of the image to be sent */
                                        /* Image size is greater than Flash size */
                                        if (*p_size > (APPLICATION_FLASH_SIZE + 1)) {
                                            /* End session */
                                            tmp = CA;
                                            Uart_WriteWithTimeOut(PSDK_CONSOLE_UART_NUM, &tmp, 1, NAK_TIMEOUT);
                                            Uart_WriteWithTimeOut(PSDK_CONSOLE_UART_NUM, &tmp, 1, NAK_TIMEOUT);
                                            result = COM_LIMIT;
                                        }
                                        /* erase user application area */
                                        FLASH_If_Erase(APPLICATION_ADDRESS, APPLICATION_ADDRESS_END);
                                        *p_size = filesize;

                                        Serial_PutByte(ACK);
                                        Serial_PutByte(CRC16);
                                    }
                                        /* File header packet is empty, end session */
                                    else {
                                        Serial_PutByte(ACK);
                                        file_done = 1;
                                        session_done = 1;
                                        break;
                                    }
                                } else /* Data packet */
                                {
                                    ramsource = (uint32_t) &aPacketData[PACKET_DATA_INDEX];
                                    /* Write received data in Flash */
                                    if (FLASH_If_Write(flashdestination, (uint8_t *) ramsource, packet_length) ==
                                        FLASHIF_OK) {
                                        flashdestination += packet_length;
                                        Serial_PutByte(ACK);
                                    } else /* An error occurred while writing to Flash memory */
                                    {
                                        /* End session */
                                        Serial_PutByte(CA);
                                        Serial_PutByte(CA);
                                        result = COM_DATA;
                                    }
                                }
                                packets_received++;
                                session_begin = 1;
                            }
                            break;
                    }
                    break;
                case HAL_BUSY: /* Abort actually */
                    Serial_PutByte(CA);
                    Serial_PutByte(CA);
                    result = COM_ABORT;
                    break;
                default:
                    if (session_begin > 0) {
                        errors++;
                    }
                    if (errors > MAX_ERRORS) {
                        /* Abort communication */
                        Serial_PutByte(CA);
                        Serial_PutByte(CA);
                    } else {
                        Serial_PutByte(CRC16); /* Ask for a packet */
                    }
                    break;
            }
        }
    }
    return result;
}

/**
  * @brief  Transmit a file using the ymodem protocol
  * @param  p_buf: Address of the first byte
  * @param  p_file_name: Name of the file sent
  * @param  file_size: Size of the transmission
  * @retval COM_StatusTypeDef result of the communication
  */
COM_StatusTypeDef Ymodem_Transmit(uint8_t *p_buf, const uint8_t *p_file_name, uint32_t file_size)
{
    uint32_t errors = 0, ack_recpt = 0, size = 0, pkt_size;
    uint8_t *p_buf_int;
    COM_StatusTypeDef result = COM_OK;
    uint32_t blk_number = 1;
    uint8_t a_rx_ctrl[2];
    uint8_t i;
#ifdef CRC16_F
    uint32_t temp_crc;
#else /* CRC16_F */   
    uint8_t temp_chksum;
#endif /* CRC16_F */

    /* Prepare first block - header */
    PrepareIntialPacket(aPacketData, p_file_name, file_size);

    while ((!ack_recpt) && (result == COM_OK)) {
        /* Send Packet */
        Uart_WriteWithTimeOut(PSDK_CONSOLE_UART_NUM, &aPacketData[PACKET_START_INDEX], PACKET_SIZE + PACKET_HEADER_SIZE,
                              NAK_TIMEOUT);

        /* Send CRC or Check Sum based on CRC16_F */
#ifdef CRC16_F
        temp_crc = Cal_CRC16(&aPacketData[PACKET_DATA_INDEX], PACKET_SIZE);
        Serial_PutByte(temp_crc >> 8);
        Serial_PutByte(temp_crc & 0xFF);
#else /* CRC16_F */   
        temp_chksum = CalcChecksum (&aPacketData[PACKET_DATA_INDEX], PACKET_SIZE);
        Serial_PutByte(temp_chksum);
#endif /* CRC16_F */

        /* Wait for Ack and 'C' */
        if (Uart_ReadWithTimeOut(PSDK_CONSOLE_UART_NUM, &a_rx_ctrl[0], 1, NAK_TIMEOUT) == HAL_OK) {
            if (a_rx_ctrl[0] == ACK) {
                ack_recpt = 1;
            } else if (a_rx_ctrl[0] == CA) {
                if ((Uart_ReadWithTimeOut(PSDK_CONSOLE_UART_NUM, &a_rx_ctrl[0], 1, NAK_TIMEOUT) == HAL_OK) &&
                    (a_rx_ctrl[0] == CA)) {
                    HAL_Delay(2);
                    result = COM_ABORT;
                }
            }
        } else {
            errors++;
        }
        if (errors >= MAX_ERRORS) {
            result = COM_ERROR;
        }
    }

    p_buf_int = p_buf;
    size = file_size;

    /* Here 1024 bytes length is used to send the packets */
    while ((size) && (result == COM_OK)) {
        /* Prepare next packet */
        PreparePacket(p_buf_int, aPacketData, blk_number, size);
        ack_recpt = 0;
        a_rx_ctrl[0] = 0;
        errors = 0;

        /* Resend packet if NAK for few times else end of communication */
        while ((!ack_recpt) && (result == COM_OK)) {
            /* Send next packet */
            if (size >= PACKET_1K_SIZE) {
                pkt_size = PACKET_1K_SIZE;
            } else {
                pkt_size = PACKET_SIZE;
            }

            Uart_WriteWithTimeOut(PSDK_CONSOLE_UART_NUM, &aPacketData[PACKET_START_INDEX],
                                  pkt_size + PACKET_HEADER_SIZE,
                                  NAK_TIMEOUT);

            /* Send CRC or Check Sum based on CRC16_F */
#ifdef CRC16_F
            temp_crc = Cal_CRC16(&aPacketData[PACKET_DATA_INDEX], pkt_size);
            Serial_PutByte(temp_crc >> 8);
            Serial_PutByte(temp_crc & 0xFF);
#else /* CRC16_F */   
            temp_chksum = CalcChecksum (&aPacketData[PACKET_DATA_INDEX], pkt_size);
            Serial_PutByte(temp_chksum);
#endif /* CRC16_F */

            /* Wait for Ack */
            if ((Uart_ReadWithTimeOut(PSDK_CONSOLE_UART_NUM, &a_rx_ctrl[0], 1, NAK_TIMEOUT) == HAL_OK) &&
                (a_rx_ctrl[0] == ACK)) {
                ack_recpt = 1;
                if (size > pkt_size) {
                    p_buf_int += pkt_size;
                    size -= pkt_size;
                    if (blk_number == (APPLICATION_FLASH_SIZE / PACKET_1K_SIZE)) {
                        result = COM_LIMIT; /* boundary error */
                    } else {
                        blk_number++;
                    }
                } else {
                    p_buf_int += pkt_size;
                    size = 0;
                }
            } else {
                errors++;
            }

            /* Resend packet if NAK  for a count of 10 else end of communication */
            if (errors >= MAX_ERRORS) {
                result = COM_ERROR;
            }
        }
    }

    /* Sending End Of Transmission char */
    ack_recpt = 0;
    a_rx_ctrl[0] = 0x00;
    errors = 0;
    while ((!ack_recpt) && (result == COM_OK)) {
        Serial_PutByte(EOT);

        /* Wait for Ack */
        if (Uart_ReadWithTimeOut(PSDK_CONSOLE_UART_NUM, &a_rx_ctrl[0], 1, NAK_TIMEOUT) == HAL_OK) {
            if (a_rx_ctrl[0] == ACK) {
                ack_recpt = 1;
            } else if (a_rx_ctrl[0] == CA) {
                if ((Uart_ReadWithTimeOut(PSDK_CONSOLE_UART_NUM, &a_rx_ctrl[0], 1, NAK_TIMEOUT) == HAL_OK) &&
                    (a_rx_ctrl[0] == CA)) {
                    HAL_Delay(2);
                    result = COM_ABORT;
                }
            }
        } else {
            errors++;
        }

        if (errors >= MAX_ERRORS) {
            result = COM_ERROR;
        }
    }

    /* Empty packet sent - some terminal emulators need this to close session */
    if (result == COM_OK) {
        /* Preparing an empty packet */
        aPacketData[PACKET_START_INDEX] = SOH;
        aPacketData[PACKET_NUMBER_INDEX] = 0;
        aPacketData[PACKET_CNUMBER_INDEX] = 0xFF;
        for (i = PACKET_DATA_INDEX; i < (PACKET_SIZE + PACKET_DATA_INDEX); i++) {
            aPacketData[i] = 0x00;
        }

        /* Send Packet */
        Uart_WriteWithTimeOut(PSDK_CONSOLE_UART_NUM, &aPacketData[PACKET_START_INDEX], PACKET_SIZE + PACKET_HEADER_SIZE,
                              NAK_TIMEOUT);

        /* Send CRC or Check Sum based on CRC16_F */
#ifdef CRC16_F
        temp_crc = Cal_CRC16(&aPacketData[PACKET_DATA_INDEX], PACKET_SIZE);
        Serial_PutByte(temp_crc >> 8);
        Serial_PutByte(temp_crc & 0xFF);
#else /* CRC16_F */   
        temp_chksum = CalcChecksum (&aPacketData[PACKET_DATA_INDEX], PACKET_SIZE);
        Serial_PutByte(temp_chksum);
#endif /* CRC16_F */

        /* Wait for Ack and 'C' */
        if (Uart_ReadWithTimeOut(PSDK_CONSOLE_UART_NUM, &a_rx_ctrl[0], 1, NAK_TIMEOUT) == HAL_OK) {
            if (a_rx_ctrl[0] == CA) {
                HAL_Delay(2);
                result = COM_ABORT;
            }
        }
    }

    return result; /* file transmitted successfully */
}

HAL_StatusTypeDef Uart_ReadWithTimeOut(E_UartNum uartNum, uint8_t *data, uint16_t len, uint32_t timeOut)
{
    int res;
    uint16_t alreadyReadLen = 0;
		uint32_t loop_count = 0;
	
    while (loop_count <= timeOut) {
        res = UART_Read(uartNum, data + alreadyReadLen, len - alreadyReadLen);
        if (res > 0) {
            alreadyReadLen += res;
        }
        if (alreadyReadLen == len) {
            return HAL_OK;
        }
        Osal_TaskSleepMs(1);
				loop_count++;
    }

    return HAL_TIMEOUT;
}

HAL_StatusTypeDef Uart_WriteWithTimeOut(E_UartNum uartNum, uint8_t *data, uint16_t len, uint32_t timeOut)
{
    int res;

    res = UART_Write(uartNum, data, len);
    if (res == len) {
        return HAL_OK;
    } else {
        return HAL_ERROR;
    }
}
/**
  * @}
  */

/*******************(C)COPYRIGHT 2016 STMicroelectronics *****END OF FILE****/
