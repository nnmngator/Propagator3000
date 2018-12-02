#include "GatorUtils.h"

namespace math
{
	cv::Vec2f Add(const cv::Vec2f & a, const cv::Vec2f & b)
	{
		return cv::Vec2f(a[0] + b[0], a[1] + b[1]);
	}
	
	cv::Vec2f Add(const cv::Vec2f & a, float b)
	{
		return cv::Vec2f(a[0] + b, a[1]);
	}
	

	cv::Vec2f Sub(const cv::Vec2f & a, const cv::Vec2f & b)
	{
		return cv::Vec2f(a[0] - b[0], a[1] - b[1]);
	}
	
	cv::Vec2f Sub(const cv::Vec2f & a, float b)
	{
		return cv::Vec2f(a[0] - b, a[1]);
	}
	

	cv::Vec2f Mul(const cv::Vec2f & a, const cv::Vec2f & b)
	{
		return cv::Vec2f(a[0] * b[0] - a[1] * b[1], a[1] * b[0] + a[0] * b[1]);
	}

	cv::Vec2f Mul(const cv::Vec2f & a, float b)
	{
		return cv::Vec2f(a[0] * b, a[1] * b);
	}


	cv::Vec2f Div(const cv::Vec2f & A, const cv::Vec2f & B)
	{
		float a = A[0], b = A[1], c = B[0], d = B[1];
		float denum = c*c + d*d;
		float re = a*c + b*d;
		float im = b*c - a*d;
		return cv::Vec2f(re / denum, im / denum);
	}

	cv::Vec2f Div(const cv::Vec2f & a, float b)
	{
		return cv::Vec2f(a[0] / b, a[1] / b);
	}


	cv::Vec2f Exp(const cv::Vec2f & cplx)
	{
		auto re = cplx[0];
		auto im = cplx[1];
		auto a = std::expf(re);
		return cv::Vec2f(a * std::cosf(im), a * std::sinf(im));
	}
}

namespace optics
{
	using namespace math;

	cv::Vec2f TransferFunction(float z, float lambda, float px, float py, int row, int col, int rows, int cols)
	{
		float fy = (row - rows / 2) / (rows * py);
		float fx = (col - cols / 2) / (cols * px);
		auto a = Exp({ 0.f,-static_cast<float>(CV_PI) * lambda*z*(fy*fy + fx*fx) });
		auto b = Exp({ 0.f, 2.0f * static_cast<float>(CV_PI) * z / lambda });
		return Mul(a, b);
	}

	cv::Vec2f Lens(float f, float lambda, float px, float py, int row, int col, int rows, int cols)
	{
		float y = (row - rows / 2) * py;
		float x = (col - cols / 2) * px;
		return Exp({ 0.f, -static_cast<float>(CV_PI) * (x*x + y*y) / (lambda * f) });
	}
}

namespace misc
{
	std::string FieldTypeToString(FieldType fieldType)
	{
		switch(fieldType)
		{
		case Re: return "Re";
		case Im: return "Im";
		case Amplitude: return "Amplitude";
		case Phase: return "Phase";
		case Intensity: return "Intensity";
		case LogIntensity: return "LogIntensity";
		}
	}
}
