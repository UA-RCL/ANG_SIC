

typedef struct structCLN
{
	structLayer			*layer;
	structLayer			*layerHead;
	structLayer			*layerClassifier;
	structPerceptron	*perceptronClassifier;
	structMAC			*macData;
	
	float				fLearningRate;
	float				fInitialError;
	float				fThreshold;
	float				fThresholdPercent;
	float				fAccuracy;
	float				fTrainAccuracy;
	float				fValidateAccuracy;
	float				fTestAccuracy;
	int					nID;
	int					nNetworkType;
	int					nLabelID;
	int					nSize;
	int					nPerceptronLayerCount;
	int					nWeightCount;
	int					nMACCount;
	int					nClassCount;
	int					bKeep;

	int					bAdjustGlobalLearningRate;
	float				fLearningRateMinimum;
	float				fLearningRateMaximum;
	int					bAdjustPerceptronLearningRate;
	int					bAdjustThreshold;
	float				fRatioAverage;
	float				fPercentBackProp;
	int					*nStartArray;
	int					*nEndArray;
	int					nLayerCount;
	float				**fOutputArray;

	int					nTargetClass;
	int					nStage;
	int						bRebuilt;

	int			nClassifierMode;
	int			nNumberFormat;

	int						nPaddingMode;
	int						bResponded;
	float					fValidationAccuracy;

	struct structCLN		*next;
	struct structCLN		*prev;

} structCLN;

structCLN		*AllocateCLN(int *nID);
void			AddNew_ClassLevelNetworks(structCLN **head, structCLN *newCLN);
structCLN		*DeleteCLN_ClassLevelNetworks(structCLN **head, int nID);
void			DeleteCLN_V2_ClassLevelNetworks(structCLN **head, int nID);
void			DeleteAll_ClassLevelNetworks(structCLN **head);

int		CreateCLN_ClassLevelNetworks(structCLN	**cln, int nLayerCount, structArchitecture *architecture, int nNetworkType, int nLabelID, int *nClassLevelNetworkCount, float fLearningRate, float fInitialError, float fThreshold, int nRowCount, int nColumnCount, int *nPerceptronID, int *nSynapseID, float *fInputArray, int nMode);
void	CreateMaxPoolLayer_ClassLevelNetworks(structCLN *cln, structArchitecture *architecture, float fLearningRate, int nRandomMode, int nRowCount, int nColumnCount, int *nPerceptronID, int *nSynapseID, float *fInputArray);
void	Create2DConvolveLayer_ClassLevelNetworks(structCLN *cln, structArchitecture *architecture, float fLearningRate, int nRandomMode, int nRowCount, int nColumnCount, int *nPerceptronID, int *nSynapseID, float *fInputArray);
void	Create3DConvolveLayer_ClassLevelNetworks(structCLN *cln, structArchitecture *architecture, float fLearningRate, int nRandomMode, int nRowCount, int nColumnCount, int *nPerceptronID, int *nSynapseID, float *fInputArray);
void	CreateConnectionLayer_ClassLevelNetworks(structCLN *cln, int nNeuralCount, int nMode, float fLearningRate, int nRandomMode, int nRowCount, int nColumnCount, int *nPerceptronID, int *nSynapseID, float *fInputArray);
void	CreateMACArray_ClassLevelNetworks(structCLN *cln);
float	GetMedian_ClassLevelNetworks(structCLN *clnHead);

void	InitializeWeights_ClassLevelNetworks(structCLN *cln, int nMode, float fSameWeight);
void	InitializeWeightsOld_ClassLevelNetworks(structCLN *cln, float *fRandomWeightarray, int nMode);
void	SetLearningRates_ClassLevelNetworks(structCLN *cln, float fLearningRateMin, float fLearningRateMax);

void	WriteSynapseWeight_ClassLevelNetworks(structCLN *cln, int nID, int nImageID, FILE *pFile);
void	SetSynapseWeight_ClassLevelNetworks(structCLN *cln, int nID, float fWeight);
void	AnalyzeSynapseInputs_ClassLevelNetworks(structCLN *cln, structInput *input, float *fInputArray);

void	InitializeWeights_ClassLevelNetworksMAC(structMAC *macData, int nMACCount);

void		WriteClassLevelNetwork(structCLN *clnCur, char *sFilePath, FILE *pFile);
structCLN	*ReadClassLevelNetwork(fxpt *fxptInputArray, int *nID, char *sFilePath, FILE *pFile);
void		AnalyzeCLN(structCLN *clnHead, int nNumberMode);


int CreateCLN_ClassLevelNetworks_v2(structCLN **cln, structArchitecture *archHead, int nNetworkType, int nLabelID, int *nClassLevelNetworkCount, float fLearningRate, float fInitialError, float fThreshold, int nRowCount, int nColumnCount, int *nPerceptronID, int *nSynapseID, float *fInputArray, int nMode);

void CreateMACArrayFromDNX_ClassLevelNetworks(structCLN *cln);

void AlterWeights(structCLN *cln, float fMultiplier);