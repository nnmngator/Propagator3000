#pragma once
#include<opencv2\opencv.hpp>


namespace math
{
	cv::Vec2f Mul(const cv::Vec2f& a, const cv::Vec2f& b);

	cv::Vec2f Mul(const cv::Vec2f& a, float b);

	cv::Vec2f Exp(const cv::Vec2f& a);
}

namespace optics
{
	cv::Vec2f TransferFunction(float z, float lambda, float px, float py, int row, int col, int rows, int cols);

	cv::Vec2f Lens(float f, float lambda, float px, float py, int row, int col, int rows, int cols);
}
