
#pragma once
void	ReadRawCIFARFile(structInputData *data, char *sImage, char *sAspect, char *sSpeed, char *sSpread, char *sHRR, int *nCount, int nLabelID, int nRow);
void	ReadCIFARFile(structInput *input, char *sDataSource);
