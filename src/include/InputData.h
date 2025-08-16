#include "Classes.h"


typedef struct structInputData
{
	int		nID;
	char	sDescription[128];
	char	sLabel[32];
	int		nLabelID;
	int		nSequence;
	int		nDataSource;
	int		nGroupA;
	int		nGroupB;
	int		bTrained;
	fx32	*fx32Intensity;
	float	*fIntensity;
	int		*nIntensity;
	float	fError;
	float	fRank;
	float	fAspect;
	float	fRadialSpeed;
	float	fSpread;

	float	fThreshold;
	float	fDifference;
	
	int		nHRRCount;
	int		nRight;
	int		nCluster;
	float	*fHRR;
	int		bCorrectClassification;
	int		bCorrect;
	int		nMissCount;


} structInputData;

typedef struct structInput
{
	structInputData		*data;
	char				sPath[256];

	int		nDataSource;
	int		nInputCount;
	char	nChannels;
	int		nRowCount;
	int		nColumnCount;
	int		nSize;

	int		nClassCount;
	int		*nClassMemberCount;
	int		*nAverageIDArray;
	int		*nMaxIDArray;
	float	*fRatioArray;

} structInput;



void	Get_InputData(structInput **inputData, structInput **inputTrainingData, structInput **inputVerifyData, structInput **inputTestingData, int nTrainingVerifySplit, char *sDrive, char *sTrainingFilePath, char *sTestingFilePath, char *sDataSource, structClass **classHead, int *nClassCount, int *nDataSource, int *nRowCount, int *nColumnCount);

void	ReadFile_InputData(structInput *input, char *sDataSource);
void	DeleteData_InputData(structInput **input);
void	GetAverages_InputData(structInput *input, structClass *classHead);
void	SplitData_InputData(structInput *input, structInput **inputTraining, structInput **inputVerify, int nPercent, int nTotalDataPercent);

int		ExpandData_InputData(structInput *inputDataSource, structInput **inputDataDestination, int nCount, int nRow, int nColumn, int nMultiplier, float fAngleIncrement, float fStartScale, float fEndScale);
int		ExpandClassData_InputData(structClass *classData, structInput *inputDataSource, structInput **inputDataDestination, int nMultiplier, float fAngleIncrement, float fStartScale, float fEndScale, int nLabelID, int nInputCount, int *nIndexArray);
void	RotateDataWithClip_InputData(float *pSrcBase, float *pDstBase, int nRow, int nColumn, float fAngle, float fScale);

float	ComputeSSIM_InputData(float *average, float *currPic, int num_pixels);

void	ReduceDataSet_InputData(structInput *inputSource, structInput **inputDestination, structClass *classHead, int nClassMemberCount);

void	SiftClasses_InputData(structInput *input, structClass **classHead, int nMode);
void	Sort_InputData(structInputData arr[], int n, int nSize, int nMode);
int		ReadAR2File_InputData(structInputData **data, char *sPath, int *nCount, int nLabelID);

void	SetStatistics_InputData(structInput *inputTrainingData, structInput *inputVerifyData, structInput *inputTestingData, structClass *classHead, int *nClassCount, int **nClassMemberCount);

void	SortByMissCount(structInputData* data, int nCount);
void	SortByDifference(structInputData* data, int nCount);
void	SortFloatAscend(float* data, int nCount);


void	GroupByDifference(structInputData* data, int nInputCount, int nClusterCount);
