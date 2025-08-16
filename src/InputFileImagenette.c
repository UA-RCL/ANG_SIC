#include "main.h"


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void ReadRawImagenetteFile(structInputData *data, char *sImage, char *sAspect, char *sSpeed, char *sSpread, char *sHRR, int *nCount, int nLabelID, int nRow)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	FILE		*pFile;
	int			nIndex;
	int			nStart;
	int			i, j;
	float		fValue;

	if ((pFile = FOpenMakeDirectory(sImage, "rb")) == NULL)
	{
		printf("Error ReadRawFile_InputFileSAR(): Could not open the input file -- %s\n\n", sImage);
		while (1);
	}

	nStart = *nCount;

	while (!feof(pFile))
	{
		data[*nCount].nID = *nCount;
		data[*nCount].nLabelID = nLabelID;
		data[*nCount].nIntensity = (int *)calloc((nRow * nRow), sizeof(int));
		data[*nCount].fIntensity = (float *)calloc((nRow * nRow), sizeof(float));

		nIndex = 0;
		for (i = 0; i<nRow; ++i)
		{
			for (j = 0; j<nRow; ++j)
			{
				fread(&fValue, sizeof(float), 1, pFile);
				data[*nCount].fIntensity[nIndex++] = fValue;
			}
		}

		++(*nCount);
	}

	fclose(pFile);


	if ((pFile = FOpenMakeDirectory(sAspect, "rb")) == NULL)
	{
		printf("Error ReadRawFile_InputFileSAR(): Could not open the input file -- %s\n\n", sImage);
		while (1);
	}

	for (i = nStart; i<*nCount; ++i)
		fread(&data[i].fAspect, sizeof(float), 1, pFile);

	fclose(pFile);



	if ((pFile = FOpenMakeDirectory(sSpeed, "rb")) == NULL)
	{
		printf("Error ReadRawFile_InputFileSAR(): Could not open the input file -- %s\n\n", sImage);
		while (1);
	}

	for (i = nStart; i<*nCount; ++i)
		fread(&data[i].fRadialSpeed, sizeof(float), 1, pFile);

	fclose(pFile);


	if ((pFile = FOpenMakeDirectory(sSpread, "rb")) == NULL)
	{
		printf("Error ReadRawFile_InputFileSAR(): Could not open the input file -- %s\n\n", sImage);
		while (1);
	}

	for (i = nStart; i<*nCount; ++i)
		fread(&data[i].fSpread, sizeof(float), 1, pFile);

	fclose(pFile);

	if ((pFile = FOpenMakeDirectory(sHRR, "rb")) == NULL)
	{
		printf("Could not find the input file -- %s\n\n", sImage);
	}
	else
	{
		for (i = nStart; i<*nCount; ++i)
		{
			data[i].fHRR = (float *)calloc(1024, sizeof(float));

			for (data[i].nHRRCount = 0; data[i].nHRRCount<1024; ++data[i].nHRRCount)
			{
				fread(&fValue, sizeof(float), 1, pFile);
				data[i].fHRR[data[i].nHRRCount] = fValue;
			}
		}

		fclose(pFile);
	}

	return;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void ReadImagenetteFile(structInput *input, char *sDataSource)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	FILE		*pFile;
	int			nImageIndex;
	int			nClassID;
	int			j;
	int			*nDataArray;


	while ((pFile = FOpenMakeDirectory(input->sPath, "rb")) == NULL)
	{
		printf("Error ReadFile_InputFileSAR(): Could not open the input file -- %s\n\n", input->sPath);
		while (1);
	}



	fread(&input->nInputCount, sizeof(int), 1, pFile);
	fread(&input->nRowCount, sizeof(int), 1, pFile);
	fread(&input->nColumnCount, sizeof(int), 1, pFile);

	input->nSize = input->nRowCount * input->nColumnCount * 3;
	
	nDataArray = (int*)calloc(input->nSize, sizeof(int));

	if ((input->data = (structInputData *)calloc(input->nInputCount, sizeof(structInputData))) == NULL)
		exit(0);

	for (nImageIndex = 0; nImageIndex < input->nInputCount; ++nImageIndex)
	{
		input->data[nImageIndex].fIntensity = (float *)calloc(input->nSize, sizeof(float));

		input->data[nImageIndex].nID = nImageIndex;
		fread(&nClassID, sizeof(int), 1, pFile);
		fread(input->data[nImageIndex].sLabel, sizeof(char), 32, pFile);

		fread(nDataArray, sizeof(int), input->nSize, pFile);

		for (j=0; j< input->nSize; ++j)
		{
			input->data[nImageIndex].fIntensity[j] = (float)nDataArray[j];
		}
	}

	fclose(pFile);


	return;
}
