#pragma once
#define NOMINMAX
#include "stdint.h"
#include <winsock2.h>
#pragma comment(lib, "Ws2_32.lib")
#include <commctrl.h>
#pragma comment(lib, "Comctl32.lib")
#include <ws2tcpip.h>
#include <mswsock.h>
#include "windows.h"
#include "string"
#undef ERROR
#define BUFFER_DEFAULT_SIZE 8192

typedef struct comms_payload_t {
	uint16_t msg_id;
	uint32_t crc_32;
	uint16_t data_size;
	uint8_t data[BUFFER_DEFAULT_SIZE];
} comms_payload_t;

//////// CRC32 ///////
class Crc32 {

public:
	Crc32();
	uint32_t Calculate(comms_payload_t* payload);
	uint32_t CalculateRaw(uint8_t * data, size_t size);
	bool Check(comms_payload_t* payload);
	uint8_t CalculateCS(const uint8_t* data, uint8_t len);

private:
	uint32_t DefaultPolynomial;
	uint32_t DefaultSeed;
	uint32_t table[256];
	uint32_t seed;
	uint32_t hash;
	uint32_t CalculateHash(uint8_t buffer[], int start, int size);
	void InitializeTable();
	uint32_t Compute(uint8_t buffer[], size_t size);

};
//////////////////////

//////// Frame ///////
int DecodeFrame(uint8_t* frame, int frame_len, comms_payload_t* payload);
int EncodeFrame(comms_payload_t* payload, uint8_t* frame);
//////////////////////

//////// COM ///////
bool COMInit(HANDLE& husb, const wchar_t* comName);
void COMDeInit(HANDLE& husb);
int COMSend(HANDLE husb, const uint8_t* msg, size_t len);
int COMReceive(HANDLE husb, uint32_t timeout, uint8_t* msg);
bool COMReset(HANDLE& husb, DWORD timeout_ms);
int COMSendMsg(HANDLE hport, uint16_t msgid, uint8_t * msg, uint16_t msg_size);
bool COMGetPidVid(HANDLE hCom, WORD* outVid, WORD* outPid);
bool COMConnectPidVid(HANDLE* inoutPortHandle, WORD vid, WORD pid);
//////////////////////

//////// UDP ///////
bool UDPInit(SOCKET * udph);
void UDPDeInit(SOCKET * udph);
int UDPSend(SOCKET  udph, const std::wstring& ip, uint16_t port, const uint8_t* msg, size_t msg_len);
int UDPReceive(SOCKET udph, uint8_t* out, size_t out_len);

int UDPSendMsg(SOCKET hsocket, std::wstring& ip, uint16_t port, uint16_t msgid, uint8_t* msg, uint16_t msg_size);

//////////////////////
