#pragma once
void	ReadIntermediateFile_InputFileSAR(char *sTrainingFilePath, char *sTestingFilePath);
void	ReadRawFile_InputFileSAR(structInputData *data, char *sImage, char *sAspect, char *sSpeed, char *sSpread, char *sHRR, int *nCount, int nLabelID, int nRow);
void	ReadFile_InputFileSAR(structInput *input, char *sDataSource);
