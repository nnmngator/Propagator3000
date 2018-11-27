#include "GatorUtils.h"

cv::Vec2f Mul(const cv::Vec2f & a, const cv::Vec2f & b)
{
	return cv::Vec2f(a[0] * b[0] - a[1] * b[1], a[1] * b[0] + a[0] * b[1]);
}


cv::Vec2f Exp(const cv::Vec2f & cplx)
{
	auto re = cplx[0];
	auto im = cplx[1];
	auto a = std::expf(re);
	return cv::Vec2f(a * std::cosf(im), a * std::sinf(im));
}

cv::Vec2f TransferFunction(float z, float lambda, float px, float py, int row, int col, int rows, int cols)
{
	float fy = (row - rows / 2) / (rows * py);
	float fx = (col - cols / 2) / (cols * px);
	auto a = Exp({ 0.f,-static_cast<float>(CV_PI) * lambda*z*(fy*fy + fx*fx) });
	auto b = Exp({ 0.f, 2.0f * static_cast<float>(CV_PI) * z / lambda });
	return Mul(a, b);
}