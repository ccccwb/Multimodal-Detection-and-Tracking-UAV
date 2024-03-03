#ifndef CLASS_YOLOV5_H_
#define CLASS_YOLOV5_H_
#include "yolo.h"
#include <thread>
#include <mutex>
#include <condition_variable>

class YoloV5 :public Yolo
{
public:
	YoloV5(
		const NetworkInfo &network_info_,
		const InferParams &infer_params_);

	BBox convert_bbox_res(const float& bx, const float& by, const float& bw, const float& bh,
		const uint32_t& stride_h_, const uint32_t& stride_w_, const uint32_t& netW, const uint32_t& netH)
	{
		BBox b;
		// Restore coordinates to network input resolution
		float x = bx * stride_w_;
		float y = by * stride_h_;

		b.x1 = x - bw / 2;
		b.x2 = x + bw / 2;

		b.y1 = y - bh / 2;
		b.y2 = y + bh / 2;

		b.x1 = clamp(b.x1, 0, netW);
		b.x2 = clamp(b.x2, 0, netW);
		b.y1 = clamp(b.y1, 0, netH);
		b.y2 = clamp(b.y2, 0, netH);

		return b;
	}

	
private:
	std::vector<BBoxInfo> decodeTensor(const int imageIdx,
		const int imageH,
		const int imageW,
		const TensorInfo& tensor) override;


};

struct ProcessThreadParams {
	uint32_t h_id;
	const TensorInfo& tensor;
	const float* detections;
	const float m_ProbThresh;
	const int imageH;
	const int imageW;
	std::vector<BBoxInfo>& binfo;
};
void processThread(ProcessThreadParams* params);


#endif