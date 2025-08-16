

float Infer_Inference(structNetwork *networkMain, structInput *input, int nDisplayMode, structMAC **macData, int nMACCount, int bWriteOutput, int nMode);
float InferCLN_Inference(structCLN *cln, structClass *classHead, structInput *input, int nDisplayMode, int **nMatrix, float *fInputArray, int bWriteOutput, char *sDrive, char *sTitle, int bMark, int bThreshold, FILE *fpFileOut, char *sTimeBuffer, int nMode, int nClusterMax);
void Analyze_Inference(structCLN *cln, structInput *input, float *fInputArray);
void MarkInputData(structNetwork* networkMain, structInput* inputTrainingData, structInput* inputVerifyData, structInput* inputTestingData);
