
typedef struct structSort
{
	int		nClassID;
	fx32	fx32Value;
	float	fValue;

} structSort;

void	CreateMatrix(int ***nMatrix, int nClassCount);
void	ClearMatrix(int **nMatrix, int nClassCount);
void	DeleteMatrix(int **nMatrix, int nClassCount);
void	Swap(structSort *xp, structSort *yp);
void	BubbleSort(structSort arr[], int n);
void	RandomizeArray(int	*nIndexArray, int nCount, int nMode);

float	ScoreMatrixV2(structNetwork *networkMain, int nDisplayMode, int nMode);


int		CalculateWindowSize(int nWindowRow, int nKernel, int nStride);


void	GetClassLevelNetworkWeightCount(structCLN *cln);
void	DescribeClassLevelNetwork(structCLN *cln, FILE *fpFileOut);
void	CopyClassLevelNetworkWeights(structCLN *cln, float *fWeights, float *fThresholds, int nMode);

float	ScoreClassLevelNetworkMatrix(int nNetworkType, int nLabelID, structClass *classHead, int **nMatrix, int nDisplayMode, FILE *fpFileOut);
void	ClearClassLevelNetworkMatrix(structClass *classHead, int **nMatrix);

float	Slope(float fY2, float fY1, float fX2, float fX1);
float	Intercept(float fY2, float fY1, float fX2, float fX1);
float	CalculateWeight(float fY2, float fY1, float fX2, float fX1, float fTarget);
void	DisplayHelpFile();
void	HoldDisplay(const char *sMessage);
int		DisplayResults(FILE *fpFile, char **sArray, int *nArray, int nCount);

void	MakeDirectory(char *path);
FILE	*FOpenMakeDirectory(char *path, const char *mode);



int	GetClassLevelNetworkWeightZeroCount(structCLN *cln);
void DisplayMessage(const char *sMessage, int nMode);

void CalculateStandardDeviationArray(float *fArray, float *fSD, float *fAverage, int nCount);
char *ReverseString(char *str);


#ifdef _WINDOWS
void		StartTimer(LARGE_INTEGER *lnFrequency, LARGE_INTEGER *lnStart);
float		EndTimer(LARGE_INTEGER *lnFrequency, LARGE_INTEGER *lnStart);
#endif

#ifdef _VXWORKS
void		StartTimer(uint64_t *lnFrequency, uint64_t *lnStart);
float		EndTimer(uint64_t *lnFrequency, uint64_t *lnStart);
#endif

#ifdef _LINUX
void		StartTimer(uint64_t *lnFrequency, struct timespec *timeStart);
float		EndTimer(uint64_t *lnFrequency, struct timespec *timeStart);
#endif
void FormatTime(float fSeconds, char *sBuffer);


int ClusterPerceptronWeights(structPerceptron *perceptronCur);

void CalculateStandardDeviationMACArray(float **fArray, float *fSD, float *fAverage, int nCount);

void ShuffleSwap(int* xp, int* yp);
void ShuffleArray(int* nIndexArray, int nCount, int nMode);

int CompareMissCounts(const void* a, const void* b);
int CompareDifference(const void* a, const void* b);
int CompareFloatAscend(const void* a, const void* b);

structLayer* CalculateOutputSize(structCLN* cln);
void CalculateLayerOutputSize(structLayer* layerCur);
