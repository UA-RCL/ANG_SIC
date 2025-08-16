typedef struct structNetwork
{
	structConfigParameters	*parameterData;
	structCLN				*clnHead;
	structCLN				*clnCur;
	structClass				*classHead;
	structLayer				*layerNew;
	structLayer				*layerInput;
	structArchitecture *	architecture;
	char					sTitle[256];
	char					sDrive[256];
	char					sDataSource[256];
	char					sTrainingFilePath[256];
	char					sTestingFilePath[256];
	char					sNetworkFilePath[256];
	char					sConfigFilePath[256];
	char					sFilePath[256];
	char					sOutputStructurePath[256];
	char					sDNAOutputPath[256];
	char					sDNXOutputPath[256];


	int						**nMatrix;
	int						*nClassMemberCount;
	int						nClassCount;
	int						nDataSource;
	int						nRowCount;
	int						nColumnCount;
	int						nWeightCount;
	int						nID;
	int						nPerceptronID;
	int						nSynapseID;
	int						nTrainVerifySplit;
	int						nPrimingCycles;
	int						nTrainingCycles;
	int						nClassLevelNetworkCount;

	
	fxpt					*fxptInputArray;
	float					*fInputArray;

	float					fMaximumThreshold;
	float					fMinimumThreshold;
	float					fLearningRate;
	float					fInitialError;
	float					fThreshold;

	int						nLayerCount;
	int						*nLayerTypeArray;
	int						*nLayerMapCountArray;
	int						*nLayerRowCountArray;
	int						*nLayerColumnCountArray;
	int						*nLayerRowStrideArray;
	int						*nLayerColumnStrideArray;
	int						*nLayerNeuronsArray;

	int						nKernelCountStart;
	int						nKernelCountEnd;
	int						nSubNetCount;
	int						nBuildSort;
	int						nBuildThreshold;
	int						nTrainSort;
	int						nTrainResplit;
	int						nTrainThreshold;
	int						nPostTrain;

	int						bAdjustGlobalLearningRate;
	float					fLearningRateMinimum;
	float					fLearningRateMaximum;
	int						bAdjustPerceptronLearningRate;
	int						bAdjustThreshold;
	float					fTargetWeight;
	int						nOpMode;
	int						bPruneNetwork;
	float					fPruneConvThreshold;
	float					fPruneFCThreshold;
	float					fThresholdPercent;
	int						bLearnRateInitialization;
	int						nParameterCount;

	int						nPruneNetwork;
	int						nPruneInterval;
	int						nNoProgressCount;
	int						nNoProgressResplitCount;
	int						nNoProgressResortCount;
	int						nTrainInferenceExecute;
	int						nValidateInferenceExecute;
	int						nTestInferenceExecute;

	int			nClassifierMode;
	int			nNumberFormat;
	int			nInputRowCount;
	int			nInputColumnCount;
	int			nOrientation;
	int			nCLNCount;
	int			nSIC;
	
	
	char		sConfigFile[256];


	char		sInputNetworkPath[256];
	char		sInputStructurePath[256];

} structNetwork;

void	Initialize_Network(structNetwork *networkMain);
void	Write_DNA_Network(structNetwork *networkMain, char *sFilePath);
void	Write_DNA_Network_V2(structNetwork *networkMain, char *sFilePath);
void	Read_Network(structNetwork *networkMain, char *sFilePath);
void	WriteV2_Network(structNetwork *networkMain, char *sFilePath);
void	ReadV2_Network(structNetwork *networkMain, char *sFilePath);
void	CreateCombiningClassifier_Network(structNetwork *networkMain, float *fRandomWeightarray, int nRandomizeMode);
void	PrintHeader_Network(structNetwork *networkMain, FILE *fpFileOut);
void	DisplayInputData(structNetwork *networkMain, FILE *fpFileOut);

void	WriteNetwork(structCLN *clnHead, char *sFilePath);
void	WriteDNANetwork(structNetwork* networkMain, char* sFilePath);
int		ReadNetwork(structNetwork *network, char *sFilePath);
void	SetFilePath(structNetwork *network, char *sFilePath);
int		DumpWeights(structNetwork *networkMain);

void	Write_DNX_Network(structNetwork *networkMain, char *sFilePath);
void	Read_DNX_Network(structNetwork *networkMain, char *sFilePath);

void	CopyWeights(structCLN* clnHead, int nMode);
