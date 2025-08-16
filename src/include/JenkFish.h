
typedef struct structData
{
	int		nID;
	float	fInput;
	float	fOutput;
	int		nCount;
	float	fSlope;
	float	fIntercept;
	float	fStart;
	float	fStartInput;
	float	fEnd;
	float	fEndInput;
	float	fSD;
	float	fAverage;
} structData;

int		CalculateAll_JenkFish(structData *dataCumulate, float *fPrevSquaredMean, float *fCurSquaredMean, int *nCurBuffer, int nClusters, int nBufferSize);
void	CalculateRange_JenkFish(structData *dataCumulate, int nBIndex, int nEIndex, int nClusterPoint, int nEPoint, int nCompleteRows, float *fPrevSquaredMean, float *fCurSquaredMean);
int		FindMaxBreakIndex_JenkFish(structData *dataCumulate, float *fPrevSquaredMean, float *fCurSquaredMean, int nIndex, int nClusterPoint, int nEPoint, int nCompleteRows);
float	GetWeightValue_JenkFish(structData *dataCumulate, int b, int e);
int		GetWeight_JenkFish(structData *dataCumulate, int b, int e);
float	GetSquaredMean_JenkFish(structData *dataCumulate, int b, int e);
void	JenkFish(structData *data, structData **dataBreak, int nClusters, int nCount);
