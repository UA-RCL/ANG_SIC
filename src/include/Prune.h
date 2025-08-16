

int		PruneWeights(structLayer **layerHead, float fConvThreshold, float fFCThreshold);
int		SetPruneWeightsZero(structLayer **layerHead, float fConvThreshold, float fFCThreshold);
int		SetMACPruneWeightsZero(structMAC *macData, int nMACCount, float fConvThreshold, float fFCThreshold);

int		PruneWeights_V2(structNetwork *network, structLayer **layerHead, structInput *inputValidateData, structInput *inputTrainData);
void	RealignWeights(structCLN *cln);
void	PrunePerceptronWeights(structNetwork* network, structInput* inputTrain, structInput* inputTest, float* fAccuracy, int nMaxCluster);
void	PrunePerceptronWeightsSinglePass(structNetwork* network, structInput* inputTrain, structInput* inputTest, int nMaxCluster);