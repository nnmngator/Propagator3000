#include "CpuGator.h"
#include "GatorUtils.h"
#include <mutex>

using namespace math;
using namespace optics;

CpuGator::CpuGator(int rows, int cols) :
	PitchX(10e-6), PitchY(10e-6), Wavelength(632.8e-9),
	m_data(rows, cols, CV_32FC2, cv::Scalar(1)),
	m_fieldType(CpuGator::ReIm)
{}

CpuGator::CpuGator(const cv::Mat & image, FieldType fieldType) :
	PitchX(10e-6), PitchY(10e-6), Wavelength(632.8e-9), m_fieldType(CpuGator::ReIm)
{
	SetImage(image, fieldType);
}

void CpuGator::SetImage(const cv::Mat & image, FieldType fieldType)
{
	m_data.release();
	m_data.create(image.rows, image.cols, CV_32FC2);
	m_data = cv::Scalar(1, 1);
	m_fieldType = (fieldType & CpuGator::ReIm) ? CpuGator::ReIm : CpuGator::AmpPhase;
	m_data.forEach<cv::Vec2f>([&](cv::Vec2f& pix, const int *pos)-> void {
		int row = pos[0], col = pos[1];
		float value = image.at<float>(row, col);
		SetPixelValue(pix, value, fieldType);
	});
}

void CpuGator::FFT()
{
	ToggleFieldType(CpuGator::ReIm);
	cv::dft(m_data, m_data, cv::DFT_SCALE);
}

void CpuGator::FFT(char axis)
{
	ToggleFieldType(CpuGator::ReIm);
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
	ToggleFieldType(CpuGator::ReIm);
	cv::dft(m_data, m_data, cv::DFT_INVERSE);
}

void CpuGator::IFFT(char axis)
{
	ToggleFieldType(CpuGator::ReIm);
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

void CpuGator::FFTShifted()
{
	ToggleFieldType(CpuGator::ReIm);
	FFTShift();
	FFT();
	FFTShift();
}

void CpuGator::FFTShifted(char axis)
{
	ToggleFieldType(CpuGator::ReIm);
	if(axis == 'x')
	{
		FFTShift();
		cv::dft(m_data, m_data, cv::DFT_ROWS);
		FFTShift();
	}
	else if(axis == 'y')
	{
		cv::rotate(m_data, m_data, cv::ROTATE_90_CLOCKWISE);
		FFTShift();
		cv::dft(m_data, m_data, cv::DFT_ROWS);
		FFTShift();
		cv::rotate(m_data, m_data, cv::ROTATE_90_COUNTERCLOCKWISE);
	}
}

void CpuGator::IFFTShifted()
{
	ToggleFieldType(CpuGator::ReIm);
	FFTShift();
	IFFT();
	FFTShift();
}

void CpuGator::IFFTShifted(char axis)
{
	ToggleFieldType(CpuGator::ReIm);
	if(axis == 'x')
	{
		FFTShift();
		cv::dft(m_data, m_data, cv::DFT_ROWS);
		FFTShift();
	}
	else if(axis == 'y')
	{
		cv::rotate(m_data, m_data, cv::ROTATE_90_CLOCKWISE);
		FFTShift();
		cv::dft(m_data, m_data, cv::DFT_ROWS | cv::DFT_INVERSE);
		FFTShift();
		cv::rotate(m_data, m_data, cv::ROTATE_90_COUNTERCLOCKWISE);
	}
}

void CpuGator::CplxToExp()
{
	// m_data already holds AmpPhase values
	if(m_fieldType & CpuGator::AmpPhase) return;

	m_fieldType = CpuGator::AmpPhase;
	m_data.forEach<cv::Vec2f>([&](cv::Vec2f &pix, const int *pos)->void {
		float A = std::sqrtf(pix[0] * pix[0] + pix[1] * pix[1]);
		float Fi = atan2f(pix[1], pix[0]);
		pix = cv::Vec2f(A, Fi);
	});
}

void CpuGator::ExpToCplx()
{
	// m_data already holds ReIm values
	if(m_fieldType & CpuGator::ReIm) return;

	m_fieldType = CpuGator::ReIm;
	m_data.forEach<cv::Vec2f>([&](cv::Vec2f &pix, const int *pos)->void {
		float A = pix[0] * cosf(pix[1]);
		float B = pix[0] * sinf(pix[1]);
		pix = cv::Vec2f(A, B);
	});
}

void CpuGator::IntensityNorm()
{
	ToggleFieldType(CpuGator::AmpPhase);
	std::vector<cv::Mat> chs(2);
	cv::split(m_data, chs);
	cv::normalize(chs[0], chs[0], 0, 1, cv::NORM_MINMAX);
	cv::merge(chs, m_data);
}

void CpuGator::PhaseBinarization(float threshold)
{
	ToggleFieldType(CpuGator::AmpPhase);
	std::vector<cv::Mat> chs(2);
	cv::split(m_data, chs);
	cv::threshold(chs[1], chs[1], threshold, CV_PI, cv::THRESH_BINARY);
	cv::merge(chs, m_data);
}

void CpuGator::Sobel(char axis, int kernelSize)
{
	ToggleFieldType(CpuGator::ReIm);
	bool x = axis == 'x' ? 1 : 0;
	bool y = axis == 'y' ? 1 : 0;
	cv::Sobel(m_data, m_data, CV_32F, x, y, kernelSize);
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
	ToggleFieldType(InternalFieldType::ReIm);
	FFTShifted();
	MulTransferFunction(distance);
	IFFTShifted();
}

void CpuGator::Propagate1D(float distance, char axis)
{
	ToggleFieldType(InternalFieldType::ReIm);
	FFTShifted(axis);
	MulTransferFunction(distance);
	IFFTShifted(axis);
}

void CpuGator::Show(FieldType fieldType)
{
	// Bo czasami nie wiem na co pacze
	static auto fieldTypeToString = [&](FieldType ft) -> std::string {
		switch(ft)
		{
		case CpuGator::Re: return "Re";
		case CpuGator::Im: return "Im";
		case CpuGator::Amplitude: return "Amplitude";
		case CpuGator::Phase: return "Phase";
		case CpuGator::Intensity: return "Intensity";
		case CpuGator::LogIntensity: return "LogIntensity";
		}
	};

	cv::Mat result(m_data.rows, m_data.cols, CV_32FC1);
	ToggleFieldType(fieldType);
	m_data.forEach<cv::Vec2f>([&](auto &pix, const int *pos)->void {
		int row = pos[0], col = pos[1];
		auto &pix2 = result.at<float>(row, col);
		pix2 = GetPixelValue(pix, fieldType);
	});
	std::string winname = fieldTypeToString(fieldType);
	cv::normalize(result, result, 0, 1, cv::NORM_MINMAX);
	cv::namedWindow(winname, cv::WINDOW_NORMAL);
	cv::resizeWindow(winname, 1024, 1024);
	cv::moveWindow(winname, 768,0);

	//ROI
	int w = 768, h = 768;
	int x = result.cols / 2 - w / 2;
	int y = result.rows / 2 - h / 2;

	cv::imshow(winname, result(cv::Rect(x, y, w, h)));
	cv::waitKey();
	cv::destroyWindow(winname);
}

void CpuGator::ShowAll()
{
	Show(CpuGator::Re);
	Show(CpuGator::Im);
	Show(CpuGator::Phase);
	Show(CpuGator::Intensity);
	Show(CpuGator::LogIntensity);
}

void CpuGator::Save(const std::string & filename, FieldType fieldType)
{
	cv::Mat result(m_data.rows, m_data.cols, CV_32FC1);
	ToggleFieldType(fieldType);
	m_data.forEach<cv::Vec2f>([&](auto &pix, const int *pos)->void {
		int row = pos[0], col = pos[1];
		auto &pix2 = result.at<float>(row, col);
		pix2 = GetPixelValue(pix, fieldType);
	});
	cv::normalize(result, result, 0, 1, cv::NORM_MINMAX);
	result.convertTo(result, CV_16UC1, 65535);
	cv::imwrite(filename + ".bmp", result);
}

void CpuGator::ShowSave(const std::string & filename, FieldType fieldType)
{
	Show(fieldType);
	Save(filename + ".bmp", fieldType);
}

void CpuGator::ShowSaveAll(const std::string & filePrefix)
{
	Show(CpuGator::Re);
	Save(filePrefix + "Re.bmp", CpuGator::Re);
	Show(CpuGator::Im);
	Save(filePrefix + "Im.bmp", CpuGator::Im);
	Show(CpuGator::Phase);
	Save(filePrefix + "Phase.bmp", CpuGator::Phase);
	Show(CpuGator::Intensity);
	Save(filePrefix + "Intensity.bmp", CpuGator::Intensity);
	Show(CpuGator::LogIntensity);
	Save(filePrefix + "LogIntensity.bmp", CpuGator::LogIntensity);
}

void CpuGator::SetPixelValue(cv::Vec2f & pix, float value, FieldType fieldType)
{
	switch(fieldType)
	{
	case CpuGator::Re:
	case CpuGator::Amplitude:
		pix[0] = value;
		break;
	case CpuGator::Im:
	case CpuGator::Phase:
		pix[1] = value;
		break;
	case CpuGator::Intensity:
		pix[0] = std::sqrtf(value);
		pix[1] = 0;
		break;
	case CpuGator::LogIntensity:
		pix[0] = std::expf(value);
		pix[1] = 0;
		break;
	default:
		break;
	}
}

float CpuGator::GetPixelValue(const cv::Vec2f & pix, FieldType fieldType)
{
	switch(fieldType)
	{
	case CpuGator::Re:
	case CpuGator::Amplitude:
		return pix[0];
	case CpuGator::Im:
	case CpuGator::Phase:
		return pix[1];
		// Assuming that intensity will be calculated as Amplitude^2 from AmpPhase
	case CpuGator::Intensity:
		return pix[0] * pix[0];
	case CpuGator::LogIntensity:
		return std::logf(1.0f + pix[0] * pix[0]);
	}
	return 0;
}

void CpuGator::ToggleFieldType()
{
	if(m_fieldType == InternalFieldType::ReIm)
	{
		CplxToExp();
		m_fieldType = InternalFieldType::AmpPhase;
	}
	else if(m_fieldType == InternalFieldType::AmpPhase)
	{
		ExpToCplx();
		m_fieldType = InternalFieldType::ReIm;
	}
}

void CpuGator::ToggleFieldType(field_type desiredFieldType)
{
	if(!(m_fieldType & desiredFieldType))
	{
		ToggleFieldType();
	}
}

