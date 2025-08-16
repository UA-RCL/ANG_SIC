#include "main.h"

float CubicHermite_Bicubic(int A, int B, int C, int D, float t)
{
	float	fA = (float)A;
	float	fB = (float)B;
	float	fC = (float)C;
	float	fD = (float)D;

	float a = -fA / 2.0f + (3.0f*fB) / 2.0f - (3.0f*fC) / 2.0f + fD / 2.0f;
	float b = fA - (5.0f*fB) / 2.0f + 2.0f*fC - fD / 2.0f;
	float c = -fA / 2.0f + fC / 2.0f;
	float d = fB;

	return a * t*t*t + b * t*t + c * t + d;
}

void GetPixelClamped_Bicubic(int *source_image, int nSourceX, int nSourceY, int x, int y, int temp[]) {

	CLAMP(x, 0, nSourceX - 1);
	CLAMP(y, 0, nSourceY - 1);

	temp[0] = source_image[x + (nSourceX*y)];
	temp[1] = 0;
	temp[2] = 0;
}

void Sample_Bicubic(int *source_image, int nSourceX, int nSourceY, float u, float v, int sample[]) {

	float x = (u * (float)nSourceX) - 0.5f;
	int xint = (int)x;
	float xfract = x - (float)floor(x);

	float y = (v * (float)nSourceY) - 0.5f;
	int yint = (int)y;
	float yfract = y - (float)floor(y);

	int i;

	int p00[3];
	int p10[3];
	int p20[3];
	int p30[3];

	int p01[3];
	int p11[3];
	int p21[3];
	int p31[3];

	int p02[3];
	int p12[3];
	int p22[3];
	int p32[3];

	int p03[3];
	int p13[3];
	int p23[3];
	int p33[3];

	// 1st row
	GetPixelClamped_Bicubic(source_image, nSourceX, nSourceY, xint - 1, yint - 1, p00);
	GetPixelClamped_Bicubic(source_image, nSourceX, nSourceY, xint + 0, yint - 1, p10);
	GetPixelClamped_Bicubic(source_image, nSourceX, nSourceY, xint + 1, yint - 1, p20);
	GetPixelClamped_Bicubic(source_image, nSourceX, nSourceY, xint + 2, yint - 1, p30);

	// 2nd row
	GetPixelClamped_Bicubic(source_image, nSourceX, nSourceY, xint - 1, yint + 0, p01);
	GetPixelClamped_Bicubic(source_image, nSourceX, nSourceY, xint + 0, yint + 0, p11);
	GetPixelClamped_Bicubic(source_image, nSourceX, nSourceY, xint + 1, yint + 0, p21);
	GetPixelClamped_Bicubic(source_image, nSourceX, nSourceY, xint + 2, yint + 0, p31);

	// 3rd row
	GetPixelClamped_Bicubic(source_image, nSourceX, nSourceY, xint - 1, yint + 1, p02);
	GetPixelClamped_Bicubic(source_image, nSourceX, nSourceY, xint + 0, yint + 1, p12);
	GetPixelClamped_Bicubic(source_image, nSourceX, nSourceY, xint + 1, yint + 1, p22);
	GetPixelClamped_Bicubic(source_image, nSourceX, nSourceY, xint + 2, yint + 1, p32);

	// 4th row
	GetPixelClamped_Bicubic(source_image, nSourceX, nSourceY, xint - 1, yint + 2, p03);
	GetPixelClamped_Bicubic(source_image, nSourceX, nSourceY, xint + 0, yint + 2, p13);
	GetPixelClamped_Bicubic(source_image, nSourceX, nSourceY, xint + 1, yint + 2, p23);
	GetPixelClamped_Bicubic(source_image, nSourceX, nSourceY, xint + 2, yint + 2, p33);

	// interpolate bi-cubically!
	for (i = 0; i < 3; i++) {

		float col0 = CubicHermite_Bicubic(p00[i], p10[i], p20[i], p30[i], xfract);
		float col1 = CubicHermite_Bicubic(p01[i], p11[i], p21[i], p31[i], xfract);
		float col2 = CubicHermite_Bicubic(p02[i], p12[i], p22[i], p32[i], xfract);
		float col3 = CubicHermite_Bicubic(p03[i], p13[i], p23[i], p33[i], xfract);

		float value = CubicHermite_Bicubic((int)col0, (int)col1, (int)col2, (int)col3, yfract);

		CLAMP(value, 0.0f, 255.0f);

		sample[i] = (int)value;

		//printf("sample[%d]=%d\n", i, sample[i]);

	}

}

void ResizeImage_Bicubic(int *source_image, int nSourceX, int nSourceY, int *destination_image, int nDestX, int nDestY)
{

	int sample[3];
	int y, x;


	//printf("x-width=%d | y-width=%d\n", nDestX, nDestY);

	for (y = 0; y < nDestY; y++) {

		float v = (float)y / (float)(nDestY - 1);

		for (x = 0; x < nDestX; ++x) {

			float u = (float)x / (float)(nDestX - 1);
			//printf("v=%f\n", v);
			//printf("u=%f\n", u);
			Sample_Bicubic(source_image, nSourceX, nSourceY, u, v, sample);

			destination_image[x + ((nDestX)*y)] = sample[0];
		}
	}
}

// // NA_1.00.72_0.8312



// na.exe -mode 4 -config D:\Data\MNIST\config\mnist_ver_2_NG_1.cfg
// na.exe -mode 4 -config D:\Data\MNIST\config\mnist_ver_2_NG_2.cfg
// na.exe -mode 4 -config D:\Data\MNIST\config\mnist_ver_2_NG_3.cfg
// na.exe -mode 4 -config D:\Data\MNIST\config\mnist_ver_2_NG_4.cfg
// na.exe -mode 4 -config D:\Data\MNIST\config\mnist_ver_2_NG_5.cfg
// na.exe -mode 4 -config D:\Data\MNIST\config\mnist_ver_2_NG_6.cfg
// na.exe -mode 4 -config D:\Data\MNIST\config\mnist_ver_2_NG_7.cfg
// na.exe -mode 4 -config D:\Data\MNIST\config\mnist_ver_2_NG_8.cfg
// na.exe -mode 4 -config D:\Data\MNIST\config\mnist_ver_2_NG_9.cfg
// na.exe -mode 4 -config D:\Data\MNIST\config\mnist_ver_2_NG_10.cfg
// na.exe -mode 4 -config D:\Data\MNIST\config\mnist_ver_2_NG_11.cfg

// na.exe -mode 4 -config D:\Data\MNIST\config\mnist_ver_2_NG_11_0505_seq.cfg -prune 0.0095

// -mode 1 -config D:\Data\MNIST\config\mnist_ver_2.cfg -network D:\Data\mnist\networks\NA_1.00.70_0.9871.dna -prune 0.001

// na.exe -mode 1 -config D:\Data\isar\config\pruning\isar_07_baseline_reduced.cfg -network D:\Data\isar\networks\NA_1.00.72_0.8312.dna -prune 0.001

// -mode 0 -config D:\Data\isar\config\pruning\isar_07_baseline_reduced.cfg

// na_73.exe -mode 4 -config D:\Data\ISAR\config\pruning\isar_07_neuro_9_60.cfg