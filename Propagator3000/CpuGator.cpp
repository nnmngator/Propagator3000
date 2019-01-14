#include "CpuGator.h"

// To use all complex math on Pix w/o namespace qualifier
using namespace math;

CpuGator::CpuGator(int rows, int cols) :
	PitchXSrc(10e-6f), PitchYSrc(10e-6f),
	PitchXDst(10e-6f), PitchYDst(10e-6f),
	Wavelength(632.8e-9f),
	Data(rows, cols, CV_32FC2, cv::Scalar(1))
{}

CpuGator::CpuGator(const cv::Mat & image, FieldType fieldType) :
	PitchXSrc(10e-6f), PitchYSrc(10e-6f),
	PitchXDst(10e-6f), PitchYDst(10e-6f),
	Wavelength(632.8e-9f)
{
	SetImage(image, fieldType);
}

CpuGator::CpuGator(const CpuGator & other) :
	PitchXSrc(other.PitchXSrc), PitchYSrc(other.PitchYSrc),
	PitchXDst(other.PitchXDst), PitchYDst(other.PitchYDst),
	Wavelength(other.Wavelength),
	// Data need to be cloned so that it is separate for each CpuGator instance
	Data(other.Data.clone())
{}

void CpuGator::SetImage(const cv::Mat & image, FieldType fieldType)
{
	assert(image.type() == CV_32FC1 &&
		"[CpuGator::SetImage] One channel floaWting point images allowed only");

	// Dispose old data
	Data.release();
	// Allocate memory for new data
	Data.create(image.rows, image.cols, CV_32FC2);
	// Set both Re and Im to 1 for all pixels
	Data = cv::Scalar(1, 1);
	// Assign given field value to each pixel
	Data.forEach<Pix>([&](Pix& pix, const int *pos)-> void {
		int row = pos[0], col = pos[1];
		float value = image.at<float>(row, col);
		SetPixelValue(pix, value, fieldType);
	});

}

void CpuGator::FFT()
{/*
	int padTopBottom = Data.rows / 2;
	int padLeftRight = Data.cols / 2;
	cv::copyMakeBorder(Data, Data, padTopBottom, padTopBottom, padLeftRight, padLeftRight, cv::BORDER_CONSTANT, cv::Scalar(0));*/
	cv::dft(Data, Data, cv::DFT_COMPLEX_OUTPUT);
	//Data = Data(cv::Rect(padLeftRight, padTopBottom, Data.rows / 2, Data.cols / 2));
}

void CpuGator::FFT(Direction direction)
{
	if (direction == Direction::Y)
	{
		cv::dft(Data, Data, cv::DFT_ROWS | cv::DFT_COMPLEX_OUTPUT);
	}
	else if (direction == Direction::X)
	{
		// [TESTED] It has to be conterclockwise FIRST
		cv::rotate(Data, Data, cv::ROTATE_90_COUNTERCLOCKWISE);
		cv::dft(Data, Data, cv::DFT_ROWS | cv::DFT_COMPLEX_OUTPUT);
		cv::rotate(Data, Data, cv::ROTATE_90_CLOCKWISE);
	}
}

void CpuGator::IFFT()
{
	cv::dft(Data, Data, cv::DFT_INVERSE | cv::DFT_SCALE);
}

void CpuGator::IFFT(Direction direction)
{
	if (direction == Direction::Y)
	{
		cv::dft(Data, Data, cv::DFT_ROWS | cv::DFT_INVERSE | cv::DFT_SCALE);
	}
	else if (direction == Direction::X)
	{
		// [TESTED] It has to be conterclockwise FIRST
		cv::rotate(Data, Data, cv::ROTATE_90_COUNTERCLOCKWISE);
		cv::dft(Data, Data, cv::DFT_ROWS | cv::DFT_INVERSE | cv::DFT_SCALE);
		cv::rotate(Data, Data, cv::ROTATE_90_CLOCKWISE);
	}
}

void CpuGator::FFTShift()
{
	Data.forEach<Pix>([&](auto &pix, const int *pos)->void {
		int row = pos[0], col = pos[1];

		if (row >= Data.rows / 2) return;

		// 2nd (top left) quarter
		if (col < Data.cols / 2)
		{
			auto& other_pix = Data.at<Pix>(row + Data.rows / 2, col + Data.cols / 2);
			std::swap(other_pix, pix);
		}
		// 1st quareter (top right)
		else
		{
			auto& other_pix = Data.at<Pix>(row + Data.rows / 2, col - Data.cols / 2);
			std::swap(other_pix, pix);
		}
	});
}

void CpuGator::NormalizeIntensity()
{
	Data.forEach<Pix>([&](Pix &pix, const int *pos) -> void {
		float intensity = GetPixelValue(pix, FieldType::Intensity);
		pix /= intensity;
	});
}

void CpuGator::BinarizePhase()
{
	Data.forEach<Pix>([&](Pix &pix, const int *pos) -> void {
		float phase = GetPixelValue(pix, FieldType::Phase);
		if (phase < 0)
		{
			pix[1] = 0;
		}
		else
		{
			pix[0] = 0;
		}
	});
}

void CpuGator::MulTransferFunction(float distance)
{
	Data.forEach<Pix>([&](Pix &pix, const int *pos)->void {
		// here explicit declaration to optics::TF need to be made
		// since there's also a static method on class with the same name
		auto tf = optics::TransferFunction(distance, Wavelength, PitchXSrc, PitchYSrc, pos[0], pos[1], Data.rows, Data.cols);
		pix = Mul(tf, pix);
	});
}

void CpuGator::ASDPropagate(float distance)
{
	FFTShift();
	FFT();
	FFTShift();
	MulTransferFunction(distance);
	FFTShift();
	IFFT();
	FFTShift();
}

void CpuGator::ASDPropagate(float distance, Direction axis)
{
	FFTShift();
	FFT(axis);
	FFTShift();
	MulTransferFunction(distance);
	FFTShift();
	IFFT(axis);
	FFTShift();
}
void CpuGator::ChirpC(float distance, float offsetX, float offsetY) {
	//X

	float s = PitchXSrc / PitchXDst;
	float o = offsetX;
	float p = PitchXDst;

	Data.forEach<Pix>([&](Pix& pix, const int * pos) -> void {
		int row = pos[0], col = pos[1];
		float x2 = (Data.cols / 2 - col) * p;

		pix = Exp({ 0, (float)(2 * CV_PI / Wavelength *distance + CV_PI / (Wavelength*distance)*((1 - s)*x2*x2 + 2 * o*x2 + o*o)) }) / (-Wavelength * distance);
	});


	//Y
	s = PitchYSrc / PitchYDst;
	o = offsetY;
	p = PitchYDst;

	Data.forEach<Pix>([&](Pix& pix, const int * pos) -> void {
		int row = pos[0], col = pos[1];
		float y2 = (Data.rows / 2 - row) * p;

		pix = Mul(pix, Exp({ 0, (float)(2 * CV_PI / Wavelength *distance + CV_PI / (Wavelength*distance)*((1 - s)*y2*y2 + 2 * o*y2 + o*o)) }));
	});


}
//void CpuGator::MulChirpC(float distance, float offsetX, float offsetY, int xsize, int ysize, float wavelength, float pitchxscr, float pitchysrc, float pitchxdst, float pitchydst) {
//	CpuGator chirpC(xsize, ysize);
//	chirpC.Wavelength = wavelength;
//	chirpC.PitchXSrc = pitchxscr;
//	chirpC.PitchYSrc = pitchysrc;
//	chirpC.PitchXDst = pitchxdst;
//	chirpC.PitchYDst = pitchydst;
//	chirpC.ChirpC(distance, 0, 0);
//	(*this) *= chirpC;
//}
void CpuGator::ChirpH(float distance, float offsetX, float offsetY)
{
	float s = PitchXSrc / PitchXDst;
	float o = offsetX;
	float p = PitchXDst;
	float mhmax = Wavelength*distance / (2 * s*p*p);
	float xmax = std::min((float)Data.cols, mhmax);

	/*Data.forEach<Pix>([&](Pix& pix, const int * pos) -> void {
		int row = pos[0], col = pos[1];
		float y2 = (Data.rows / 2 - row) * p;

		pix = Exp({ 0, (float)((CV_PI * s*y2 *y2) / (Wavelength * distance)) });

	});*/
	Data.forEach<Pix>([&](Pix& pix, const int * pos) -> void {
		int row = pos[0], col = pos[1];
		float xh = (Data.cols / 2 - col) * p;
		if (xh <= xmax) {
			pix = Exp({ 0, (float)((CV_PI * s*xh *xh) / (Wavelength * distance)) });
		}
		else {
			pix = { 0,0 };
		}
	});
	s = PitchYSrc / PitchYDst;
	o = offsetY;
	p = PitchYDst;
	mhmax = Wavelength*distance / (2 * s*p*p);
	xmax = std::min((float)Data.cols, mhmax);

	Data.forEach<Pix>([&](Pix& pix, const int * pos) -> void {
		int row = pos[0], col = pos[1];
		float yh = (Data.rows / 2 - row) * p;
		if (yh <= xmax) {
			pix = Mul(pix, Exp({ 0, (float)(CV_PI * s*yh *yh / (Wavelength * distance)) }));
		}
		else {
			pix = { 0,0 };
		}
	});

}
//void CpuGator::MulChirpH(float distance, float offsetX, float offsetY, int xsize, int ysize, float wavelength, float pitchxscr, float pitchysrc, float pitchxdst, float pitchydst) {
//	CpuGator chirpH(xsize, ysize);
//	chirpH.Wavelength = wavelength;
//	chirpH.PitchXSrc = pitchxscr;
//	chirpH.PitchYSrc = pitchysrc;
//	chirpH.PitchXDst = pitchxdst;
//	chirpH.PitchYDst = pitchydst;
//	chirpH.ChirpH(distance, 0, 0);
//	(*this) *= chirpH;
//}
void CpuGator::ChirpU(float distance, float offsetX, float offsetY)
{
	float s = PitchXSrc / PitchXDst;
	float o = offsetX;
	float p = PitchXDst;

	Data.forEach<Pix>([&](Pix& pix, const int * pos) -> void {
		int row = pos[0], col = pos[1];
		float x1 = (Data.cols / 2 - col) * p;

		pix = Exp({ 0, (float)(CV_PI * ((s*s - s)*x1*x1 - 2 * s*o*x1) / (Wavelength * distance)) });
	});

	s = PitchYSrc / PitchYDst;
	o = offsetY;
	p = PitchYDst;

	Data.forEach<Pix>([&](Pix& pix, const int * pos) -> void {
		int row = pos[0], col = pos[1];
		float y1 = (Data.rows / 2 - row) * p;

		pix = Mul(pix, Exp({ 0, (float)(CV_PI * ((s*s - s)*y1*y1 - 2 * s*o*y1) / (Wavelength * distance)) }));
	});
}

//void CpuGator::MulChirpU(float distance, float offsetX, float offsetY) {
//	CpuGator chirpU(Data.rows, Data.cols);
//	chirpU.Wavelength = Wavelength;
//	chirpU.PitchXSrc = PitchXDst;
//	chirpU.PitchYSrc = PitchYSrc;
//	chirpU.PitchXDst = PitchXDst;
//	chirpU.PitchYDst = PitchYDst;
//	chirpU.ChirpU(distance, 0, 0);
//	(*this) *= chirpU;
//}

void CpuGator::ARSSPropagate(float distance, float offsetX, float offsetY)
{
	// Placeholder for chirp functions
	CpuGator chirp(Data.rows, Data.cols);
	chirp.Wavelength = Wavelength;
	chirp.PitchXSrc = PitchXSrc;
	chirp.PitchYSrc = PitchYSrc;
	chirp.PitchXDst = PitchXDst;
	chirp.PitchYDst = PitchYDst;

	// First mul data by Chirp U
	chirp.ChirpU(distance, offsetX, offsetY);

	(*this) *= chirp;

	// 1st FFT
	FFTShift();
	FFT();
	FFTShift();

	// Compute Chirp H (rect already included)
	chirp.ChirpH(distance, offsetX, offsetY);

	// 2nd FFT
	chirp.FFTShift();
	chirp.FFT();
	chirp.FFTShift();

	// Mul 1st FFT by 2nd
	(*this) *= chirp;

	// IFFT
	FFTShift();
	IFFT();
	FFTShift();

	// Times chirpC
	chirp.ChirpC(distance, offsetX, offsetY);
	(*this) *= chirp;
}

void CpuGator::MulLens(float focal)
{
	Data.forEach<Pix>([&](Pix &pix, const int *pos)->void {
		// here explicit declaration to optics::Lens need to be made
		// since there's also a static method on class with the same name
		auto lens = optics::Lens(focal, Wavelength, PitchXSrc, PitchYSrc, pos[0], pos[1], Data.rows, Data.cols);
		pix = Mul(lens, pix);
	});
}

void CpuGator::Propagate(float distance, PropagationMethod method)
{
	switch (method)
	{
	case PropagationMethod::ASD:
		FFTShift();
		FFT();
		FFTShift();
		MulTransferFunction(distance);
		FFTShift();
		IFFT();
		FFTShift();
		break;
	case PropagationMethod::ARSSFresnel:
		// TUTAJ PISZ
		break;
	}
}

void CpuGator::Propagate(float distance, Direction direction, PropagationMethod method)
{
	// TODO check how to perform 1D propagations other than ASD
	FFTShift();
	FFT(direction);
	FFTShift();
	MulTransferFunction(distance);
	FFTShift();
	IFFT(direction);
	FFTShift();
}

void CpuGator::Show(FieldType fieldType) const
{
	cv::Mat result(Data.rows, Data.cols, CV_32FC1);
	Data.forEach<Pix>([&](auto &pix, const int *pos)->void {
		int row = pos[0], col = pos[1];
		result.at<float>(row, col) = GetPixelValue(pix, fieldType);
	});
	std::string winname = FieldTypeToString(fieldType);
	cv::normalize(result, result, 0, 1, cv::NORM_MINMAX);
	cv::namedWindow(winname, cv::WINDOW_NORMAL);
	cv::resizeWindow(winname, 768, 768);
	cv::moveWindow(winname, 768, 0);

	// ROI
	int w = 512, h = 512;
	int x = result.cols / 2 - w / 2;
	int y = result.cols / 2 - h / 2;
	cv::Rect r(x, y, w, h);
	cv::imshow(winname, result);
	cv::waitKey();
	cv::destroyWindow(winname);
}

void CpuGator::ShowAll() const
{
	for (int i = FieldType::Re; i != FieldType::LogIntensity + 1; i++)
	{
		auto fieldType = static_cast<FieldType>(i);
		Show(fieldType);
	}
}

void CpuGator::Save(const std::string & filename, FieldType fieldType) const
{
	cv::Mat result(Data.rows, Data.cols, CV_32FC1);
	Data.forEach<Pix>([&](auto &pix, const int *pos)->void {
		int row = pos[0], col = pos[1];
		Data.forEach<Pix>([&](auto &pix, const int *pos)->void {
			int row = pos[0], col = pos[1];
			result.at<float>(row, col) = GetPixelValue(pix, fieldType);
		});
	});
	cv::normalize(result, result, 0, 1, cv::NORM_MINMAX);
	result.convertTo(result, CV_16UC1, 65535);
	cv::imwrite(filename + ".bmp", result);
}

void CpuGator::ShowSave(const std::string & filename, FieldType fieldType) const
{
	Show(fieldType);
	Save(filename + ".bmp", fieldType);
}

void CpuGator::ShowSaveAll(const std::string & filePrefix) const
{
	for (int i = FieldType::Re; i != FieldType::LogIntensity; i++)
	{
		auto fieldType = static_cast<FieldType>(i);
		auto fieldTypeName = FieldTypeToString(fieldType);
		Show(fieldType);
		Save(filePrefix + fieldTypeName + ".bmp", fieldType);
	}
}

cv::Vec2f CpuGator::Min() const
{
	std::vector<cv::Mat> chnls;
	cv::split(Data, chnls);
	double minRe, minIm;
	cv::minMaxLoc(chnls[0], &minRe, nullptr);
	cv::minMaxLoc(chnls[1], &minIm, nullptr);
	Pix minVal{ static_cast<float>(minRe),static_cast<float>(minIm) };
	return minVal;
}

cv::Vec2f CpuGator::Max() const
{
	std::vector<cv::Mat> chnls;
	cv::split(Data, chnls);
	double maxRe, maxIm;
	cv::minMaxLoc(chnls[0], nullptr, &maxRe);
	cv::minMaxLoc(chnls[1], nullptr, &maxIm);
	Pix maxVal{ static_cast<float>(maxRe), static_cast<float>(maxIm) };
	return maxVal;
}

void CpuGator::Re()
{
	Data.forEach<Pix>([&](auto& pix, const int * pos) -> void {
		pix[1] = 0;
	});
}

void CpuGator::Im()
{
	Data.forEach<Pix>([&](auto& pix, const int * pos) -> void {
		pix[0] = 0;
	});
}

void CpuGator::Phase()
{
	Data.forEach<Pix>([&](auto& pix, const int * pos) -> void {
		auto phase = GetPixelValue(pix, FieldType::Phase);
		pix[0] = std::cosf(phase);
		pix[1] = std::sinf(phase);
	});
}

void CpuGator::Amplitude()
{
	Data.forEach<Pix>([&](auto& pix, const int * pos) -> void {
		pix[0] = GetPixelValue(pix, FieldType::Amplitude);
		pix[1] = 0;
	});
}

void CpuGator::Intensity()
{
	Data.forEach<Pix>([&](auto& pix, const int * pos) -> void {
		pix[0] = GetPixelValue(pix, FieldType::Intensity);
		pix[1] = 0;
	});
}

CpuGator CpuGator::TransferFunction(int rows, int cols, float distance, float pitchX, float pitchY, float wavelength)
{
	CpuGator tf(rows, cols);
	tf.PitchXSrc = pitchX;
	tf.PitchYSrc = pitchY;
	tf.Wavelength = wavelength;
	tf.MulTransferFunction(distance);
	return tf;
}

CpuGator CpuGator::Lens(int rows, int cols, float focal, float pitchX, float pitchY, float wavelength)
{
	CpuGator lens(rows, cols);
	lens.PitchXSrc = pitchX;
	lens.PitchYSrc = pitchY;
	lens.Wavelength = wavelength;
	lens.MulLens(focal);
	return lens;
}

CpuGator & CpuGator::operator=(const CpuGator & other)
{
	Data.release();
	Data = other.Data.clone();
	PitchXSrc = other.PitchXSrc;
	PitchYSrc = other.PitchYSrc;
	Wavelength = other.Wavelength;
	return *this;
}

CpuGator & CpuGator::operator=(const Pix & value)
{
	Data = value;
	return *this;
}

void CpuGator::operator+=(const CpuGator & other)
{
	Data.forEach<Pix>([&](auto& pix, const int* pos) -> void {
		pix = Add(pix, other.Data.at<Pix>(pos[0], pos[1]));
	});
}

void CpuGator::operator-=(const CpuGator & other)
{
	Data.forEach<Pix>([&](auto& pix, const int* pos) -> void {
		pix = Sub(pix, other.Data.at<Pix>(pos[0], pos[1]));
	});
}

void CpuGator::operator*=(const CpuGator & other)
{
	Data.forEach<Pix>([&](auto& pix, const int* pos) -> void {
		pix = Mul(pix, other.Data.at<Pix>(pos[0], pos[1]));
	});
}

void CpuGator::operator/=(const CpuGator & other)
{
	Data.forEach<Pix>([&](auto& pix, const int* pos) -> void {
		pix = Div(pix, other.Data.at<Pix>(pos[0], pos[1]));
	});
}

void CpuGator::operator+=(const Pix & value)
{
	Data.forEach<Pix>([&](auto& pix, const int* pos) -> void {
		pix = Add(pix, value);
	});
}

void CpuGator::operator-=(const Pix & value)
{
	Data.forEach<Pix>([&](auto& pix, const int* pos) -> void {
		pix = Sub(pix, value);
	});
}

void CpuGator::operator*=(const Pix & value)
{
	Data.forEach<Pix>([&](auto& pix, const int* pos) -> void {
		pix = Mul(pix, value);
	});
}

void CpuGator::operator/=(const Pix & value)
{
	Data.forEach<Pix>([&](auto& pix, const int* pos) -> void {
		pix = Div(pix, value);
	});
}

CpuGator & CpuGator::operator+(const CpuGator & other)
{
	Data.forEach<Pix>([&](auto& pix, const int* pos) -> void {
		pix = Add(pix, other.Data.at<Pix>(pos[0], pos[1]));
	});
	return *this;
}

CpuGator & CpuGator::operator-(const CpuGator & other)
{
	Data.forEach<Pix>([&](auto& pix, const int* pos) -> void {
		pix = Sub(pix, other.Data.at<Pix>(pos[0], pos[1]));
	});
	return *this;
}

CpuGator & CpuGator::operator*(const CpuGator & other)
{
	Data.forEach<Pix>([&](auto& pix, const int* pos) -> void {
		pix = Mul(pix, other.Data.at<Pix>(pos[0], pos[1]));
	});
	return *this;
}

CpuGator & CpuGator::operator/(const CpuGator & other)
{
	Data.forEach<Pix>([&](auto& pix, const int* pos) -> void {
		pix = Div(pix, other.Data.at<Pix>(pos[0], pos[1]));
	});
	return *this;
}

CpuGator & CpuGator::operator+(const Pix & value)
{
	Data.forEach<Pix>([&](auto& pix, const int* pos) -> void {
		pix = Add(pix, value);
	});
	return *this;
}

CpuGator & CpuGator::operator-(const Pix & value)
{
	Data.forEach<Pix>([&](auto& pix, const int* pos) -> void {
		pix = Sub(pix, value);
	});
	return *this;
}

CpuGator & CpuGator::operator*(const Pix & value)
{
	Data.forEach<Pix>([&](auto& pix, const int* pos) -> void {
		pix = Mul(pix, value);
	});
	return *this;
}

CpuGator & CpuGator::operator/(const Pix & value)
{
	Data.forEach<Pix>([&](auto& pix, const int* pos) -> void {
		pix = Div(pix, value);
	});
	return *this;
}

void CpuGator::SetPixelValue(Pix & pix, float value, FieldType fieldType)
{
	float currentAmp;

	switch (fieldType)
	{
	case FieldType::Re:
		pix[0] = value;
		break;
	case FieldType::Im:
		pix[1] = value;
		break;
	case FieldType::Amplitude:
		currentAmp = GetPixelValue(pix, FieldType::Amplitude);
		pix *= value / currentAmp;
	case FieldType::Phase:
		pix[0] = std::cosf(value);
		pix[1] = std::sinf(value);
		break;
	case FieldType::Intensity:
		currentAmp = GetPixelValue(pix, FieldType::Amplitude);
		pix *= std::sqrtf(value) / currentAmp;
		break;
	case FieldType::LogIntensity:
		currentAmp = GetPixelValue(pix, FieldType::Amplitude);
		pix *= std::sqrtf(std::expf(value)) / currentAmp;
		break;
	}
}

float CpuGator::GetPixelValue(const Pix & pix, FieldType fieldType) const
{
	switch (fieldType)
	{
	case FieldType::Re:
		return pix[0];
	case FieldType::Im:
		return pix[1];
	case FieldType::Amplitude:
		return std::sqrtf(pix[0] * pix[0] + pix[1] * pix[1]);
	case FieldType::Phase:
		return std::atan2f(pix[1], pix[0]);
	case FieldType::Intensity:
		return pix[0] * pix[0] + pix[1] * pix[1];
	case FieldType::LogIntensity:
		// 1.0f + -> because log of 0 is -inf, to avoid this, also
		// to always return positive values since log(x) = 0 for x = 1
		return std::logf(1.0f + pix[0] * pix[0] + pix[1] * pix[1]);
	default:
		// This should not happen :) but just in case trigger error 
		// so we can notice that
		assert(false && "[CpuGator::GetPixelValue] Unsupported field type");
		return 0;
	}
}

