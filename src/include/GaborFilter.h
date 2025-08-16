

void		CreateGaborArray_GaborFilter(structLayer *layerData);
int			FindLegal_GaborFilter(structArchitecture **archData, int nPrimingCycles, int nMaxWeight, int nRowCount);
void		FindBest_GaborFilter(structCLN **clnReturn, structNetwork *networkMain, structArchitecture *archData, int nArchCount, int nLabelID, int nMaxWeight, int nPrimingCycles, structInput *inputData, structInput *inputTestingData, structInput *inputTrainingData, structInput *inputVerifyData);
