#pragma once
void	ReadRawImagenetteFile(structInputData *data, char *sImage, char *sAspect, char *sSpeed, char *sSpread, char *sHRR, int *nCount, int nLabelID, int nRow);
void	ReadImagenetteFile(structInput *input, char *sDataSource);
