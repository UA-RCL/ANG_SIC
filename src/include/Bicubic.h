
#define W 21
#define H 20
#define CLAMP(v, min, max) if(v < min) { v = min; } else if(v > max) { v = max; }

float	CubicHermite_Bicubic(int A, int B, int C, int D, float t);
void	GetPixelClamped_Bicubic(int *source_image, int nSourceX, int nSourceY, int x, int y, int temp[]);
void	Sample_Bicubic(int *source_image, int nSourceX, int nSourceY, float u, float v, int sample[]);
void	ResizeImage_Bicubic(int *source_image, int nSourceX, int nSourceY, int *destination_image, int nDestX, int nDestY);


