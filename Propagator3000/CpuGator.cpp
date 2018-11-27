#include "CpuGator.h"
#include "GatorUtils.h"
#include <mutex>

using namespace math;
using namespace optics;

CpuGator::CpuGator(int rows, int cols) :
	PitchX(10e-6), PitchY(10e-6), Wavelength(632.8e-9), m_data(rows, cols, CV_32FC2, cv::Scalar(1))
{}

CpuGator::CpuGator(const cv::Mat & image, FieldType fieldType) :
	PitchX(10e-6), PitchY(10e-6), Wavelength(632.8e-9)
{
	SetImage(image, fieldType);
}

void CpuGator::SetImage(const cv::Mat & image, FieldType fieldType)
{
	m_data.release();
	m_data.create(image.rows, image.cols, CV_32FC2);
	m_data = cv::Scalar(1, 1);
	m_data.forEach<cv::Vec2f>([&](cv::Vec2f& pix, const int *pos)-> void {
		int row = pos[0], col = pos[1];
		switch(fieldType)
		{
		case CpuGator::Re:
			pix[0] = image.at<float>(row, col);
			break;
		case CpuGator::Im:
			pix[1] = image.at<float>(row, col);
			break;
		case CpuGator::Phase:
			//todo xD
			break;
		case CpuGator::Amplitude:
			//todo xD
			break;
		case CpuGator::Intensity:
			//todo xD
			break;
		case CpuGator::LogIntensity:
			//todo xD
			break;
		default:
			break;
		}
	});

}

void CpuGator::FFT()
{
	cv::dft(m_data, m_data, cv::DFT_SCALE);
}

void CpuGator::FFT(char axis)
{
	if(axis == 'x')
	{
		cv::dft(m_data, m_data, cv::DFT_ROWS);
	}
	else if(axis == 'y')
	{
		cv::rotate(m_data, m_data, cv::ROTATE_90_CLOCKWISE);
		cv::dft(m_data, m_data, cv::DFT_ROWS);
		cv::rotate(m_data, m_data, cv::ROTATE_90_COUNTERCLOCKWISE);
	}
}

void CpuGator::IFFT()
{
	cv::dft(m_data, m_data, cv::DFT_INVERSE);
}

void CpuGator::IFFT(char axis)
{
	if(axis == 'x')
	{
		cv::dft(m_data, m_data, cv::DFT_ROWS | cv::DFT_INVERSE);
	}
	else if(axis == 'y')
	{
		cv::rotate(m_data, m_data, cv::ROTATE_90_CLOCKWISE);
		cv::dft(m_data, m_data, cv::DFT_ROWS | cv::DFT_INVERSE);
		cv::rotate(m_data, m_data, cv::ROTATE_90_COUNTERCLOCKWISE);
	}
}

void CpuGator::FFTShift()
{
	m_data.forEach<cv::Vec2f>([&](auto &pix, const int *pos)->void {
		int row = pos[0], col = pos[1];

		if(row >= m_data.rows / 2) return;

		// 2nd (top left) quarter
		if(col < m_data.cols / 2)
		{
			auto& other_pix = m_data.at<cv::Vec2f>(row + m_data.rows / 2, col + m_data.cols / 2);
			std::swap(other_pix, pix);
		}
		// 1st quareter (top right)
		else
		{
			auto& other_pix = m_data.at<cv::Vec2f>(row + m_data.rows / 2, col - m_data.cols / 2);
			std::swap(other_pix, pix);
		}
	});
}

inline void CpuGator::FFTShifted()
{
	FFTShift();
	FFT();
	FFTShift();
}

inline void CpuGator::IFFTShifted()
{
	FFTShift();
	IFFT();
	FFTShift();
}

void CpuGator::CplxToExp()
{
	m_data.forEach<cv::Vec2f>([&](cv::Vec2f &pix, const int *pos)->void {
		float A = std::sqrtf(pix[0] * pix[0] + pix[1] * pix[1]);
		float Fi = atan2f(pix[1], pix[0]);
		pix = cv::Vec2f(A, Fi);
	});
}

void CpuGator::ExpToCplx()
{
	m_data.forEach<cv::Vec2f>([&](cv::Vec2f &pix, const int *pos)->void {
		float A = pix[0] * cosf(pix[1]);
		float B = pix[0] * sinf(pix[1]);
		pix = cv::Vec2f(A, B);
	});
}

void CpuGator::IntNormCplx()
{
	CplxToExp();
	IntNormExp();
	ExpToCplx();
}

void CpuGator::IntNormExp()
{
	std::vector<cv::Mat> chs(2);
	cv::split(m_data, chs);
	cv::normalize(chs[0], chs[0], 0, 1, cv::NORM_MINMAX);
	cv::merge(chs, m_data);
}

void CpuGator::PhaseBinExp(float threshold)
{
	std::vector<cv::Mat> chs(2);
	cv::split(m_data, chs);
	cv::threshold(chs[1], chs[1], threshold, 2 * CV_PI, cv::THRESH_BINARY);
	cv::merge(chs, m_data);
}

void CpuGator::PhaseBinCplx(float threshold )
{
	CplxToExp();
	PhaseBinExp();
	ExpToCplx();
}

void CpuGator::MulTransferFunction(float distance)
{
	m_data.forEach<cv::Vec2f>([&](cv::Vec2f &pix, const int *pos)->void {
		auto tf = TransferFunction(distance, Wavelength, PitchX, PitchY, pos[0], pos[1], m_data.rows, m_data.cols);
		pix = Mul(tf, pix);
	});
}

void CpuGator::MulLens(float focal)
{
	m_data.forEach<cv::Vec2f>([&](cv::Vec2f &pix, const int *pos)->void {
		auto lens = Lens(focal, Wavelength, PitchX, PitchY, pos[0], pos[1], m_data.rows, m_data.cols);
		pix = Mul(lens, pix);
	});
}

void CpuGator::Propagate(float distance)
{
	FFTShifted();
	MulTransferFunction(distance);
	IFFTShifted();
}

void CpuGator::Propagate1D(float distance, char axis)
{
	FFTShift();
	FFT(axis);
	FFTShift();
	MulTransferFunction(distance);
	FFTShift();
	IFFT(axis);
	FFTShift();
}

void CpuGator::Show(FieldType fieldType)
{
	cv::Mat result(m_data.rows, m_data.cols, CV_32FC1);
	m_data.forEach<cv::Vec2f>([&](auto &pix, const int *pos)->void {
		int row = pos[0], col = pos[1];
		auto &pix2 = result.at<float>(row, col);
		switch(fieldType)
		{
		case CpuGator::Re:
			pix2 = pix[0];
			break;
		case CpuGator::Im:
			pix2 = pix[1];
			break;
		case CpuGator::Phase:
			pix2 = std::atan2f(pix[1], pix[0]);
			break;
		case CpuGator::Amplitude:
			pix2 = std::sqrtf(pix[1] * pix[1] + pix[0] * pix[0]);
			break;
		case CpuGator::Intensity:
			pix2 = pix[1] * pix[1] + pix[0] * pix[0];
			break;
		case CpuGator::LogIntensity:
			pix2 = std::logf(1.0f + pix[1] * pix[1] + pix[0] * pix[0]);
			//todo xD
			break;
		default:
			break;
		}
	});
	cv::normalize(result, result, 0, 1, cv::NORM_MINMAX);
	cv::namedWindow("a", cv::WINDOW_NORMAL);
	cv::imshow("a", result);
	cv::waitKey();
	cv::destroyAllWindows();
}

