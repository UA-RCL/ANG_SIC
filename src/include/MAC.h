
typedef struct structMAC
{
	int		nID;
	int		nCount;
	float	**fWeight;
	float	**fInput;
	float	*fOutput;
	float	*fDifferential;
	float	*fLearningRate;
	float	**fConnectToDifferential;
	float	*fAverage;
	float	*fSumSquares;
	float	***fInputArray;
	float	**fInputSum;
	float	**fInputAverage;
	float	fAdjustLearningRate;
	float	fAngle;
	float	fFeedBackWeight;

	int		*nInputCount;
	int		nLayerType;
	int		nLayerCount;
	int		nKernelID;
	int		*nAverageCount;
	int		nSDCount;
	int		*nConnectFromID;

} structMAC;


void	PrintPerceptronInputData_MAC(structMAC *macData, int nPerceptronCount);
void	ClearAverages_MAC(structMAC *macData, int nPerceptronCount);
void	CalculateAverages_MAC(structMAC *macData, int nPerceptronCount);
float	CalculateStandardDeviations_MAC(structMAC *macData, int nPerceptronCount, int nDisplayMode);
void	FreeArray_MAC(structMAC **macData, int *nMACCount);
