#pragma once
#include<opencv2\opencv.hpp>

namespace math
{
	cv::Vec2f Add(const cv::Vec2f& a, const cv::Vec2f& b);

	cv::Vec2f Add(const cv::Vec2f& a, float b);

	cv::Vec2f Sub(const cv::Vec2f& a, const cv::Vec2f& b);

	cv::Vec2f Sub(const cv::Vec2f& a, float b);

	cv::Vec2f Mul(const cv::Vec2f& a, const cv::Vec2f& b);

	cv::Vec2f Mul(const cv::Vec2f& a, float b);

	cv::Vec2f Div(const cv::Vec2f& A, const cv::Vec2f& B);

	cv::Vec2f Div(const cv::Vec2f& a, float b);

	cv::Vec2f Exp(const cv::Vec2f& a);
}

namespace optics 
{
	/// <summary>
	/// Available optical field types
	/// </summary>
	enum FieldType
	{
		Re,
		Im,
		Phase,
		Amplitude,
		Intensity,
		LogIntensity
	};

	cv::Vec2f TransferFunction(float z, float lambda, float px, float py, int row, int col, int rows, int cols);

	cv::Vec2f Lens(float f, float lambda, float px, float py, int row, int col, int rows, int cols);
}

namespace misc
{
	// To use FieldType w/o namespace qualifier
	using namespace optics;

	/// <summary>
	/// Directions for propagation and FFts
	/// </summary>
	enum Direction
	{
		Y,
		X
	};

	std::string FieldTypeToString(FieldType fieldType);
}