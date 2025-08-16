
structLayer *NeuroGenesis(structNetwork *networkMain, structInput *inputData, structInput *inputTestingData, structInput **inputTrainingData, structInput **inputVerifyData, float fLearningRate, float fThreshold, float fMinimumThreshold, float fMaximumThreshold, structMAC **macData, int nMACCount, int nBuildMode, float *fTestAccuracy, FILE *fpFileOut);
void		GetLastLayerOutputAverages_Neurogenesis(structCLN *cln, structInput *inputData, structClass *classHead, structLayer *layerInput, float *fInputArray);
void		GetExtremeInputClassMembers_Neurogenesis(structNetwork *networkMain, structLayer *layerCur, structInput *inputData, float **fOutputArray, int nClassID, structMAC **macData, int nMACCount);

void		AddExtremePerceptron_Neurogenesis(structNetwork *networkMain, structLayer *layerCur, structLayer **layerNew, structInput *inputData, int nInputIndex, int *nPerceptronCount, float fLearningRate, float fThreshold, float fMinimumThreshold, float fMaximumThreshold, structMAC **macData, int nMACCount);

void		ConnectExtremePerceptrons_Neurogenesis(structNetwork *networkMain, structLayer *layerCur);
void		GrowNewLayer_Neurogenesis(structNetwork *networkMain, structCLN *cln, structLayer *layer, structInput *inputData, int *nPerceptronCount, float fLearningRate, float fThreshold, float fMinimumThreshold, float fMaximumThreshold);
