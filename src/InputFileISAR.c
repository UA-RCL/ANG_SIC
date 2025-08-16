#include "main.h"

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void ReadIntermediateFile_InputFileSAR(char *sTrainingFilePath, char *sTestingFilePath)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	FILE				*pFile;
	structInputData		*data;
	structInput			input;
	float				fValue;
	float				fMax;
	float				fMin;
	float				fMultiplier;
	float				fTrainingPercent = 60.0f;
	int					*nClassMemberCount;
	int					nClassCount;
	int					nTrainingCount;
	int					*nIndexArray = NULL;
	int					nMinimumCount = 99999999;
	int					nCount = 0;
	int					nIndex = 0;
	int					nTemp = 0;
	int					nValue = 0;
	int					nSize = 0;
	int					nRow = 128;
	int					i, j, k;
	char				sDirectory[256];
	char				sFilePath[256];
	char				sImage[256];
	char				sAspect[256];
	char				sSpeed[256];
	char				sSpread[256];
	char				sHRR[256];

	//// ISAR

	strcpy(sDirectory, sTrainingFilePath);

	sprintf(sFilePath, "%s\\training.bin", sDirectory);
	if ((pFile = FOpenMakeDirectory(sFilePath, "rb")) == NULL)
	{
		nCount = 0;
		data = (structInputData *)calloc(100000, sizeof(structInputData));

		printf(".");
		sprintf(sImage, "%s\\saveimageDDG.bin", sDirectory);
		sprintf(sImage, "%s\\saveimagefloatDDG.bin", sDirectory);
		sprintf(sAspect, "%s\\saveaspectDDG.bin", sDirectory);
		sprintf(sSpeed, "%s\\savevofstDDG.bin", sDirectory);
		sprintf(sSpread, "%s\\savevspanDDG.bin", sDirectory);
		sprintf(sHRR, "%s\\saveHRRfloatDDG.bin", sDirectory);
		ReadRawFile_InputFileSAR(data, sImage, sAspect, sSpeed, sSpread, sHRR, &nCount, 0, nRow);

		printf(".");
		sprintf(sImage, "%s\\saveimageAUX.bin", sDirectory);
		sprintf(sImage, "%s\\saveimagefloatAUX.bin", sDirectory);
		sprintf(sAspect, "%s\\saveaspectAUX.bin", sDirectory);
		sprintf(sSpeed, "%s\\savevofstAUX.bin", sDirectory);
		sprintf(sSpread, "%s\\savevspanAUX.bin", sDirectory);
		sprintf(sHRR, "%s\\saveHRRfloatAUX.bin", sDirectory);
		ReadRawFile_InputFileSAR(data, sImage, sAspect, sSpeed, sSpread, sHRR, &nCount, 1, nRow);

		printf(".");
		sprintf(sImage, "%s\\saveimageFFG.bin", sDirectory);
		sprintf(sImage, "%s\\saveimagefloatFFG.bin", sDirectory);
		sprintf(sAspect, "%s\\saveaspectFFG.bin", sDirectory);
		sprintf(sSpeed, "%s\\savevofstFFG.bin", sDirectory);
		sprintf(sSpread, "%s\\savevspanFFG.bin", sDirectory);
		sprintf(sHRR, "%s\\saveHRRfloatFFG.bin", sDirectory);
		ReadRawFile_InputFileSAR(data, sImage, sAspect, sSpeed, sSpread, sHRR, &nCount, 2, nRow);

		nClassCount = 3;
		nClassMemberCount = (int *)calloc(nClassCount, sizeof(int));

		for (i = 0; i < nCount; ++i)
			++nClassMemberCount[data[i].nLabelID];

		/////////////////////////////////////////////////////////////////////////////////////////////
		nCount = 0;
		for (i = 0; i < nClassCount; ++i)
		{
			if (nClassMemberCount[i] < nMinimumCount)
				nMinimumCount = nClassMemberCount[i];

			nCount += nClassMemberCount[i];
			nClassMemberCount[i] = 0;
		}

		printf("\n%d\n", nCount);

		printf(".");
		nIndexArray = (int *)calloc(nCount, sizeof(int));
		RandomizeArray(nIndexArray, nCount, RANDOMIZE);

		nTrainingCount = (int)((float)nMinimumCount * (fTrainingPercent / 100.0f));

////////////////////////////////////////////////////////////////////////////////////////////

		nTemp = 0;
		for (i = 0; i < nCount; ++i)
		{
			if (nClassMemberCount[0] < nTrainingCount && data[nIndexArray[i]].nLabelID == 0)
			{
				++nClassMemberCount[0];
				nTemp++;
				data[nIndexArray[i]].bTrained = 1;
			}
			else if (nClassMemberCount[1] < nTrainingCount && data[nIndexArray[i]].nLabelID == 1)
			{
				++nClassMemberCount[1];
				nTemp++;
				data[nIndexArray[i]].bTrained = 1;
			}
			else if (nClassMemberCount[1] < nTrainingCount && data[nIndexArray[i]].nLabelID == 2)
			{
				++nClassMemberCount[2];
				nTemp++;
				data[nIndexArray[i]].bTrained = 1;
			}
		}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		input.nDataSource = 300;
		input.nInputCount = nTemp;
		input.nChannels = 1;
		input.nRowCount = 128;
		input.nColumnCount = 128;
		input.nSize = input.nRowCount * input.nColumnCount;

// Normalize Input 2D Data //////////////////////////////////////////////////////////////////////////////////////////////////
		for (i = 0, fMax = 0.0f, fMin = 0.0f; i < nCount; ++i)
		{
			//data[i].fIntensity = (float *)calloc(input.nSize, sizeof(float));

			for (j = 0; j < input.nSize; ++j)
			{
				//data[i].fIntensity[j] = (float)((int)(unsigned char)data[i].nIntensity[j]) - 128.0F;

				if (data[i].fIntensity[j] > fMax)
					fMax = data[i].fIntensity[j];
				if (data[i].fIntensity[j] < fMin)
					fMin = data[i].fIntensity[j];
			}
		}

		fMultiplier = 1.0f / fMax;

		for (i = 0; i < nCount; ++i)
			for (j = 0; j < input.nSize; ++j)
				data[i].fIntensity[j] *= fMultiplier;

// Normalize Input 1D Data //////////////////////////////////////////////////////////////////////////////////////////////////
		for (i = 0, fMax = 0.0f, fMin = 0.0f; i < nCount; ++i)
		{
			for (j = 0; j < data[i].nHRRCount; ++j)
			{
				if (data[i].fHRR[j] > fMax)
					fMax = data[i].fHRR[j];
				if (data[i].fHRR[j] < fMin)
					fMin = data[i].fHRR[j];
			}
		}

		fMultiplier = 1.0f / fMax;

		for (i = 0; i < nCount; ++i)
			for (j = 0; j < data[i].nHRRCount; ++j)
				data[i].fHRR[j] *= fMultiplier;



		printf(".");
		sprintf(sFilePath, "%s", sTrainingFilePath);
		sprintf(sTrainingFilePath, "%s\\training.bin", sDirectory);

		if ((pFile = FOpenMakeDirectory(sTrainingFilePath, "wb")) == NULL)
		{
			printf("Error ReadFile_InputFileIR(): Could not find the input file -- %s\n\n", sTrainingFilePath);
			while (1);
		}

		fwrite(&input.nDataSource, sizeof(int), 1, pFile);
		fwrite(&input.nInputCount, sizeof(int), 1, pFile);
		fwrite(&input.nChannels, sizeof(unsigned char), 1, pFile);
		fwrite(&input.nRowCount, sizeof(int), 1, pFile);
		fwrite(&input.nColumnCount, sizeof(int), 1, pFile);

		

		for (i = 0; i < nCount; ++i)
		{
			if (data[nIndexArray[i]].bTrained == 1)
			{
				fwrite(&data[nIndexArray[i]].nLabelID, sizeof(int), 1, pFile);
				fwrite(&data[nIndexArray[i]].nGroupA, sizeof(int), 1, pFile);
				fwrite(&data[nIndexArray[i]].nGroupB, sizeof(int), 1, pFile);

				fwrite(&data[nIndexArray[i]].fAspect, sizeof(float), 1, pFile);
				fwrite(&data[nIndexArray[i]].fRadialSpeed, sizeof(float), 1, pFile);
				fwrite(&data[nIndexArray[i]].fSpread, sizeof(float), 1, pFile);


				sprintf(data[nIndexArray[i]].sLabel, "%d", data[nIndexArray[i]].nLabelID);

				for (j = 0; j < input.nRowCount; ++j)
				{
					for (k = 0; k < input.nColumnCount; ++k)
					{
						nIndex = k * input.nRowCount + j;
						fValue = data[nIndexArray[i]].fIntensity[nIndex];
						fwrite(&fValue, sizeof(float), 1, pFile);
					}
				}

				fwrite(&data[nIndexArray[i]].nHRRCount, sizeof(int), 1, pFile);

				for (j = 0; j < data[nIndexArray[i]].nHRRCount; ++j)
				{
					fValue = data[nIndexArray[i]].fHRR[j];
					fwrite(&fValue, sizeof(float), 1, pFile);
				}
			}
		}

		fclose(pFile);

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////

		for (i = 0, nTemp = 0; i < nCount; ++i)
		{
			if (!data[nIndexArray[i]].bTrained)
			{
				++nTemp;
			}
		}

		input.nInputCount = nTemp;

		printf(".");
		sprintf(sTestingFilePath, "%s\\testing.bin", sDirectory);
		if ((pFile = FOpenMakeDirectory(sTestingFilePath, "wb")) == NULL)
		{
			printf("Error ReadFile_InputFileIR(): Could not find the input file -- %s\n\n", sTestingFilePath);
			while (1);
		}

		fwrite(&input.nDataSource, sizeof(int), 1, pFile);
		fwrite(&input.nInputCount, sizeof(int), 1, pFile);
		fwrite(&input.nChannels, sizeof(unsigned char), 1, pFile);
		fwrite(&input.nRowCount, sizeof(int), 1, pFile);
		fwrite(&input.nColumnCount, sizeof(int), 1, pFile);

		input.nSize = input.nRowCount * input.nColumnCount;

		for (i = 0; i < nCount; ++i)
		{
			if (!data[nIndexArray[i]].bTrained)
			{
				fwrite(&data[nIndexArray[i]].nLabelID, sizeof(int), 1, pFile);
				fwrite(&data[nIndexArray[i]].nGroupA, sizeof(int), 1, pFile);
				fwrite(&data[nIndexArray[i]].nGroupB, sizeof(int), 1, pFile);

				fwrite(&data[nIndexArray[i]].fAspect, sizeof(float), 1, pFile);
				fwrite(&data[nIndexArray[i]].fRadialSpeed, sizeof(float), 1, pFile);
				fwrite(&data[nIndexArray[i]].fSpread, sizeof(float), 1, pFile);

				sprintf(data[nIndexArray[i]].sLabel, "%d", data[nIndexArray[i]].nLabelID);

				for (j = 0; j < input.nRowCount; ++j)
				{
					for (k = 0; k < input.nColumnCount; ++k)
					{
						nIndex = k * input.nRowCount + j;
						fValue = data[nIndexArray[i]].fIntensity[nIndex];
						fwrite(&fValue, sizeof(float), 1, pFile);
					}
				}

				fwrite(&data[nIndexArray[i]].nHRRCount, sizeof(int), 1, pFile);

				for (j = 0; j < data[nIndexArray[i]].nHRRCount; ++j)
				{
					fValue = data[nIndexArray[i]].fHRR[j];
					fwrite(&fValue, sizeof(float), 1, pFile);
				}
			}
		}

		free(data);
		free(nIndexArray);

		fclose(pFile);
	}
	else
	{
		fclose(pFile);

		printf(".");

		sprintf(sFilePath, "%s", sTrainingFilePath);
		sprintf(sTrainingFilePath, "%s\\training.bin", sDirectory);
		sprintf(sTestingFilePath, "%s\\testing.bin", sDirectory);
	}
	return;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void ReadRawFile_InputFileSAR(structInputData *data, char *sImage, char *sAspect, char *sSpeed, char *sSpread, char *sHRR, int *nCount, int nLabelID, int nRow)
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
void ReadFile_InputFileSAR(structInput *input, char *sDataSource)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	FILE		*pFile;
	float		fValue;
	int			nImageIndex;
	int			nClassID;
	int			nIndex;
	int			j, k;


	while ((pFile = FOpenMakeDirectory(input->sPath, "rb")) == NULL)
	{
		printf("Error ReadFile_InputFileSAR(): Could not open the input file -- %s\n\n", input->sPath);
		while (1);
	}

	fread(&input->nDataSource, sizeof(int), 1, pFile);
	fread(&input->nInputCount, sizeof(int), 1, pFile);
	fread(&input->nChannels, sizeof(unsigned char), 1, pFile);
	fread(&input->nRowCount, sizeof(int), 1, pFile);
	fread(&input->nColumnCount, sizeof(int), 1, pFile);

	input->nSize = input->nRowCount * input->nColumnCount;

	if ((input->data = (structInputData *)calloc(input->nInputCount, sizeof(structInputData))) == NULL)
		exit(0);

	for (nImageIndex = 0; nImageIndex < input->nInputCount; ++nImageIndex)
	{
		input->data[nImageIndex].fIntensity = (float *)calloc(input->nSize, sizeof(float));

		input->data[nImageIndex].nID = nImageIndex;
		fread(&nClassID, sizeof(int), 1, pFile);
		fread(&input->data[nImageIndex].nGroupA, sizeof(int), 1, pFile);
		fread(&input->data[nImageIndex].nGroupB, sizeof(int), 1, pFile);
		fread(&input->data[nImageIndex].fAspect, sizeof(float), 1, pFile);
		fread(&input->data[nImageIndex].fRadialSpeed, sizeof(float), 1, pFile);
		fread(&input->data[nImageIndex].fSpread, sizeof(float), 1, pFile);
		sprintf(input->data[nImageIndex].sLabel, "%d", nClassID);

		for (j = 0; j<input->nRowCount; ++j)
		{
			for (k = 0; k<input->nColumnCount; ++k)
			{
				nIndex = j * input->nColumnCount + k;
				fread(&fValue, sizeof(float), 1, pFile);

				input->data[nImageIndex].fIntensity[nIndex] = fValue;
			}
		}

		fread(&input->data[nImageIndex].nHRRCount, sizeof(int), 1, pFile);
		if (input->data[nImageIndex].nHRRCount > 0)
		{
			input->data[nImageIndex].fHRR = (float *)calloc(input->data[nImageIndex].nHRRCount, sizeof(float));

			for (j = 0; j < input->data[nImageIndex].nHRRCount; ++j)
			{
				fread(&fValue, sizeof(float), 1, pFile);
				input->data[nImageIndex].fHRR[j] = fValue;
			}
		}
	}

	fclose(pFile);

	return;
}
