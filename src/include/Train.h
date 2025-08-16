
void	Network_Train(structNetwork *networkMain, structCLN *clnCur, structInput **inputTrainingData, structInput **inputVerifyData, structInput *inputTestingData, structInput *inputData, int nCycles, FILE *fpFileOut, int nExecuteTestInference);
int		ForwardPropagate_Train(structMAC *macData, int nPerceptronCount, FILE *pFile);
void	BackPropagate_Train(structCLN *cln, structLayer *layerClassifier, int nTargetClass, float fInitialError, float *fErrorArray, structMAC *macData, int nPerceptronCount, int bAdjustLearningRate);
float	ClassLevelNetwork_Train(structCLN *cln, structClass *classHead, structInput *inputTrain, structInput *inputVerify, int nDisplayMode, int nCycle, int **nMatrix, float *fInputArray, FILE *fpFileOut);
float	CombiningClassifier_Train(structNetwork *networkMain, structInput *inputTestingData, structInput *inputTrain, structInput *inputVerify, FILE *fpFileOut);
int 	ForwardPropagateCombiningClassifier_Train(structCLN *clnHead, int *nZeroCount, int *nSingleCount, int *nMultipleCount);
float	ClassLevelNetworkGroup_Train(structNetwork *networkMain, structCLN *cln, structInput *inputTrainingData, structInput *inputVerifyData, structInput *inputTestingData, float *fRandomWeightarray, int nEpochMax, int nRandomizeMode, int nThresholdMode, FILE *fpFileOut);


void	GetForwardPropagateAverage_Train(structMAC *macData, int nPerceptronCount, int nMode);
void	GetForwardPropagateSD_Train(structMAC *macData, int nPerceptronCount, int nMode);

int ForwardPropagateAnalyze_Train(structMAC *macData, int nPerceptronCount, int **nAdditionArray, int **nMultiplicationArray, int *nLayerCount, int nSIR);
