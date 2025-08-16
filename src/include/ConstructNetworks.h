

float		BuildCompleteNetwork_ConstructNetworks(structNetwork *networkMain, structInput *inputTrainingData, structInput *inputVerifyData, structInput *inputTestingData, structInput *inputData, FILE *fpFileOut);
void		RebuildCompleteNetwork_ConstructNetworks(structNetwork *networkMain, FILE *fpFileOut, int nDisplayNetwork, structInput *inputTrainingData);
float		BuildClassLevelNetwork_ConstructNetworks(structNetwork *networkMain, structInput *inputTrainingData, structInput *inputVerifyData, structInput *inputTestingData, structInput *inputData, FILE *fpFileOut);
float		RebuildClassLevelNetwork_ConstructNetworks(structNetwork *networkMain, structInput *inputTrainingData, structInput *inputVerifyData, structInput *inputTestingData, structInput *inputData, FILE *fpFileOut);

structCLN	*CreateSeedNetwork_ConstructNetworks(structNetwork *networkMain);
float		PrimeSeedNetwork_ConstructNetworks(structNetwork *networkMain, structCLN *cln, structClass *classHead, structInput *inputData, structInput **inputTrainingData, structInput **inputVerifyData, structInput *inputTestingData, float *fInputArray, int nCycles, float fTargetWeight, int bRandomizeWeights, int nLearningRateMode, float fLearningRate, int bMatrixDisplay, FILE *fpFileOut, int bLearnRateInitializationOverride);
