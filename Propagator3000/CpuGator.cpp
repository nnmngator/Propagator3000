#include "CpuGator.h"
#include "GatorUtils.h"
#include <mutex>

CpuGator::CpuGator(int rows, int cols) :
	PitchX(10e-6), PitchY(10e-6), Wavelength(632.8e-9), m_data(rows, cols, CV_32FC2)
{
}

CpuGator::CpuGator(const cv::Mat & image, FieldType fieldType) :
	PitchX(10e-6), PitchY(10e-6), Wavelength(632.8e-9)
{
	SetImage(image, fieldType);
}

void CpuGator::SetImage(const cv::Mat & image, FieldType fieldType)
{
	m_data.release();
	m_data.create(image.rows, image.cols, CV_32FC2);
	m_data.forEach<cv::Vec2f>([&](cv::Vec2f& pix, const int *pos)-> void {
		int row = pos[0], col = pos[1];
		switch (fieldType)
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

void CpuGator::FFT2()
{
	cv::dft(m_data, m_data);
}

void CpuGator::FFT1rows()
{
	cv::dft(m_data, m_data, cv::DFT_ROWS);
}

void CpuGator::FFT1cols()
{
	cv::rotate(m_data, m_data, cv::ROTATE_90_CLOCKWISE);
	FFT1rows();
	cv::rotate(m_data, m_data, cv::ROTATE_90_COUNTERCLOCKWISE);
}

void CpuGator::IFFT2()
{
	cv::dft(m_data, m_data, cv::DFT_INVERSE);
}

void CpuGator::FFTShift()
{
	m_data.forEach<cv::Vec2f>([&](auto &pix, const int *pos)->void {
		int row = pos[0], col = pos[1];

		if (row >= m_data.rows / 2) return;

		// 2nd (top left) quarter
		if (col < m_data.cols / 2) {
			auto& other_pix = m_data.at<cv::Vec2f>(row + m_data.rows / 2, col + m_data.cols / 2);
			std::swap(other_pix, pix);
		}
		// 1st quareter (top right)
		else {
			auto& other_pix = m_data.at<cv::Vec2f>(row + m_data.rows / 2, col - m_data.cols / 2);
			std::swap(other_pix, pix);
		}
	});
}

void CpuGator::MulTransferFunction(float distance)
{
	m_data.forEach<cv::Vec2f>([&](cv::Vec2f &pix, const int *pos)->void {
		auto tf = TransferFunction(distance, Wavelength, PitchX, PitchY, pos[0], pos[1], m_data.rows, m_data.cols);
		pix = Mul(tf, pix);
	});
}

void CpuGator::Propagate(float distance)
{
	FFTShift();
	cv::dft(m_data, m_data);
	FFTShift();
	m_data.forEach<cv::Vec2f>([&](cv::Vec2f& imgPix, const int * pos) -> void {
		int row = pos[0], col = pos[1];
		auto tfPix = TransferFunction(distance, Wavelength, PitchX, PitchY, row, col, m_data.rows, m_data.cols);
		imgPix = Mul(imgPix, tfPix);
	});
	FFTShift();
	cv::dft(m_data, m_data, cv::DFT_INVERSE);
	FFTShift();
}


void CpuGator::Show(FieldType fieldType)
{
	cv::Mat result(m_data.rows, m_data.cols, CV_32FC1);
	m_data.forEach<cv::Vec2f>([&](auto &pix, const int *pos)->void {
		int row = pos[0], col = pos[1];
		auto &pix2 = result.at<float>(row, col);
		switch (fieldType)
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
			//todo xD
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

