#include "main.h"

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void ReadIntermediateFile_InputFileIR(char *sDrive, char *sTrainingFilePath, char *sTestingFilePath)
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
	int					*nIndexArray = NULL;
	int					nMinimumCount = 99999999;
	int					nCount = 0;
	int					nIndex = 0;
	int					nTemp = 0;
	int					nValue = 0;
	int					nSize = 0;
	int					i, j, k;
	char				sDirectory[256];
	char				sFilePath[256];




	//// IR
	sprintf(sDirectory, "%s\\bin", sDrive);

	sprintf(sFilePath, "%s\\training.bin", sDirectory);
	if ((pFile = FOpenMakeDirectory(sFilePath, "rb")) == NULL)
	{
		nCount = ReadRawFile_InputFileIR(&data, sTrainingFilePath);
		nIndexArray = (int *)calloc(nCount, sizeof(int));
		RandomizeArray(nIndexArray, nCount, RANDOMIZE);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		input.nDataSource = 900;
		input.nInputCount = nCount;
		input.nChannels = 1;
		input.nRowCount = 64;
		input.nColumnCount = 64;
		input.nSize = input.nRowCount * input.nColumnCount;

// Normalize Input 2D Data //////////////////////////////////////////////////////////////////////////////////////////////////
		for (i = 0, fMax = 0.0f, fMin = 0.0f; i < nCount; ++i)
		{
			data[i].fIntensity = (float *)calloc(input.nSize, sizeof(float));

			for (j = 0; j < input.nSize; ++j)
			{
				data[i].fIntensity[j] = (float)((int)(unsigned char)data[i].nIntensity[j]) - 127.0F;

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

		if (fMax != 0.0f)
		{
			fMultiplier = 1.0f / fMax;

			for (i = 0; i < nCount; ++i)
				for (j = 0; j < data[i].nHRRCount; ++j)
					data[i].fHRR[j] *= fMultiplier;
		}



		printf(".");
		sprintf(sFilePath, "%s", sTrainingFilePath);
		sprintf(sTrainingFilePath, "%s\\training.bin", sDirectory);

		if ((pFile = FOpenMakeDirectory(sTrainingFilePath, "wb")) == NULL)
		{
			printf("Error ReadIntermediateFile_InputFileIR(): Could not find the input file -- %s\n\n", sTrainingFilePath);
			while (1);
		}

		fwrite(&input.nDataSource, sizeof(int), 1, pFile);
		fwrite(&input.nInputCount, sizeof(int), 1, pFile);
		fwrite(&input.nChannels, sizeof(unsigned char), 1, pFile);
		fwrite(&input.nRowCount, sizeof(int), 1, pFile);
		fwrite(&input.nColumnCount, sizeof(int), 1, pFile);

		for (i = 0; i < nCount; ++i)
		{
			fwrite(&data[nIndexArray[i]].nLabelID, sizeof(int), 1, pFile);
			fwrite(&data[nIndexArray[i]].nGroupA, sizeof(int), 1, pFile);
			fwrite(&data[nIndexArray[i]].nGroupB, sizeof(int), 1, pFile);

			fwrite(&data[nIndexArray[i]].fAspect, sizeof(float), 1, pFile);
			fwrite(&data[nIndexArray[i]].fRadialSpeed, sizeof(float), 1, pFile);
			fwrite(&data[nIndexArray[i]].fSpread, sizeof(float), 1, pFile);
			fwrite(data[nIndexArray[i]].sLabel, sizeof(char), 32, pFile);

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

		fclose(pFile);
		free(data);
		free(nIndexArray);

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		nCount = ReadRawFile_InputFileIR(&data, sTestingFilePath);
		nIndexArray = (int *)calloc(nCount, sizeof(int));
		RandomizeArray(nIndexArray, nCount, RANDOMIZE);

		input.nDataSource = 900;
		input.nInputCount = nCount;
		input.nChannels = 1;
		input.nRowCount = 64;
		input.nColumnCount = 64;
		input.nSize = input.nRowCount * input.nColumnCount;

		// Normalize Input 2D Data //////////////////////////////////////////////////////////////////////////////////////////////////
		for (i = 0, fMax = 0.0f, fMin = 0.0f; i < nCount; ++i)
		{
			data[i].fIntensity = (float *)calloc(input.nSize, sizeof(float));

			for (j = 0; j < input.nSize; ++j)
			{
				data[i].fIntensity[j] = (float)((int)(unsigned char)data[i].nIntensity[j]) - 127.0F;

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

		if (fMax != 0.0f)
		{
			fMultiplier = 1.0f / fMax;

			for (i = 0; i < nCount; ++i)
				for (j = 0; j < data[i].nHRRCount; ++j)
					data[i].fHRR[j] *= fMultiplier;
		}


		input.nInputCount = 0;
		for (i = 0; i < nCount; ++i)
		{
			if (data[i].nLabelID < 5)
			{
				++input.nInputCount;
			}
		}



		printf(".");
		sprintf(sFilePath, "%s", sTrainingFilePath);
		sprintf(sTrainingFilePath, "%s\\testing.bin", sDirectory);

		if ((pFile = FOpenMakeDirectory(sTrainingFilePath, "wb")) == NULL)
		{
			printf("Error ReadIntermediateFile_InputFileIR(): Could not find the input file -- %s\n\n", sTrainingFilePath);
			while (1);
		}

		fwrite(&input.nDataSource, sizeof(int), 1, pFile);
		fwrite(&input.nInputCount, sizeof(int), 1, pFile);
		fwrite(&input.nChannels, sizeof(unsigned char), 1, pFile);
		fwrite(&input.nRowCount, sizeof(int), 1, pFile);
		fwrite(&input.nColumnCount, sizeof(int), 1, pFile);

		for (i = 0; i < nCount; ++i)
		{
			if (data[nIndexArray[i]].nLabelID > 4)
				continue;
			
			
			fwrite(&data[nIndexArray[i]].nLabelID, sizeof(int), 1, pFile);
			fwrite(&data[nIndexArray[i]].nGroupA, sizeof(int), 1, pFile);
			fwrite(&data[nIndexArray[i]].nGroupB, sizeof(int), 1, pFile);

			fwrite(&data[nIndexArray[i]].fAspect, sizeof(float), 1, pFile);
			fwrite(&data[nIndexArray[i]].fRadialSpeed, sizeof(float), 1, pFile);
			fwrite(&data[nIndexArray[i]].fSpread, sizeof(float), 1, pFile);
			fwrite(data[nIndexArray[i]].sLabel, sizeof(char), 32, pFile);

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

		fclose(pFile);

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////

		free(data);
		free(nIndexArray);
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
int ReadRawFile_InputFileIR(structInputData **data, char *sDataPath)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	FILE		*pFile;
	int			*nArray;
	int			nInputCount;
	int			nSize;
	
#ifdef _WINDOWS
	unsigned __int16  	nValue;
	unsigned __int16  	nLabelID;
	unsigned __int16  	nRows;
	unsigned __int16  	nColumns;
#endif

#ifdef _LINUX
	int  	nValue;
	int  	nLabelID;
	int  	nRows;
	int  	nColumns;
#endif	
	
	int			nCount;
	int			i, j;
	int			p, q, r, s;
	float		fScale;
	
	int  	nMin=99999;
	int 	nMax = 0;
	int 	nDelta = 0;

	typedef struct structLabel
	{
		int		nID;
		int		nLabelID;
		char	sLabel[33];
	} structLabel;

	structLabel		ir_train[6];
	structLabel		ir_mmstd_03[17];

	nCount = 0;
	ir_train[1].nID = nCount;
	ir_train[1].nLabelID = nCount++;
	strcpy(ir_train[1].sLabel, "BMP");

	ir_train[2].nID = nCount;
	ir_train[2].nLabelID = nCount++;
	strcpy(ir_train[2].sLabel, "BRDM");

	ir_train[3].nID = nCount;
	ir_train[3].nLabelID = nCount++;
	strcpy(ir_train[3].sLabel, "HMMWV");

	ir_train[4].nID = nCount;
	ir_train[4].nLabelID = nCount++;
	strcpy(ir_train[4].sLabel, "T72");

	ir_train[5].nID = nCount;
	ir_train[5].nLabelID = nCount++;
	strcpy(ir_train[5].sLabel, "TECH");

	nCount = 0;
	ir_mmstd_03[1].nID = nCount;
	ir_mmstd_03[1].nLabelID = nCount++;
	strcpy(ir_mmstd_03[1].sLabel, "BMP");

	ir_mmstd_03[3].nID = nCount;
	ir_mmstd_03[3].nLabelID = nCount++;
	strcpy(ir_mmstd_03[3].sLabel, "BRDM");

	ir_mmstd_03[8].nID = nCount;
	ir_mmstd_03[8].nLabelID = nCount++;
	strcpy(ir_mmstd_03[8].sLabel, "HMMWV");

	ir_mmstd_03[9].nID = nCount;
	ir_mmstd_03[9].nLabelID = nCount++;
	strcpy(ir_mmstd_03[9].sLabel, "T72");

	ir_mmstd_03[11].nID = nCount;
	ir_mmstd_03[11].nLabelID = nCount++;
	strcpy(ir_mmstd_03[11].sLabel, "TECH");

	ir_mmstd_03[2].nID = nCount;
	ir_mmstd_03[2].nLabelID = nCount++;
	strcpy(ir_mmstd_03[2].sLabel, "BMP_1");

	ir_mmstd_03[4].nID = nCount;
	ir_mmstd_03[4].nLabelID = nCount++;
	strcpy(ir_mmstd_03[4].sLabel, "DECOY_SA13");

	ir_mmstd_03[5].nID = nCount;
	ir_mmstd_03[5].nLabelID = nCount++;
	strcpy(ir_mmstd_03[5].sLabel, "DECOY_T72_1");

	ir_mmstd_03[6].nID = nCount;
	ir_mmstd_03[6].nLabelID = nCount++;
	strcpy(ir_mmstd_03[6].sLabel, "DECOY_T72_2");

	ir_mmstd_03[7].nID = nCount;
	ir_mmstd_03[7].nLabelID = nCount++;
	strcpy(ir_mmstd_03[7].sLabel, "DECOY_T72_3");

	ir_mmstd_03[10].nID = nCount;
	ir_mmstd_03[10].nLabelID = nCount++;
	strcpy(ir_mmstd_03[10].sLabel, "T72_1");

	ir_mmstd_03[12].nID = nCount;
	ir_mmstd_03[12].nLabelID = nCount++;
	strcpy(ir_mmstd_03[12].sLabel, "UNKNOWN_1");

	ir_mmstd_03[13].nID = nCount;
	ir_mmstd_03[13].nLabelID = nCount++;
	strcpy(ir_mmstd_03[13].sLabel, "UNKNOWN_WHEELED");

	ir_mmstd_03[14].nID = nCount;
	ir_mmstd_03[14].nLabelID = nCount++;
	strcpy(ir_mmstd_03[14].sLabel, "UNKNOWN_WHEELED_1");

	ir_mmstd_03[15].nID = nCount;
	ir_mmstd_03[15].nLabelID = nCount++;
	strcpy(ir_mmstd_03[15].sLabel, "UNKNOWN_WHEELED_2");

	ir_mmstd_03[16].nID = nCount;
	ir_mmstd_03[16].nLabelID = nCount++;
	strcpy(ir_mmstd_03[16].sLabel, "UNKNOWN_WHEELED_3");






	if ((pFile = FOpenMakeDirectory(sDataPath, "rb")) == NULL)
	{
		printf("Error ReadRawFile_InputFileIR(): Could not find the input file -- %s\n\n", sDataPath);
		while (1);
	}

	fread(&nInputCount, sizeof(int), 1, pFile);
	(*data) = (structInputData *)calloc(nInputCount, sizeof(structInputData));

	for (i=0; i<nInputCount; ++i)
	{

#ifdef _WINDOWS
		fread(&nLabelID, sizeof(unsigned __int16), 1, pFile);
		fread(&nRows, sizeof(unsigned __int16), 1, pFile);
		fread(&nColumns, sizeof(unsigned __int16), 1, pFile);
#endif

#ifdef _LINUX
		fread(&nLabelID, sizeof(int), 1, pFile);
		fread(&nRows, sizeof(int), 1, pFile);
		fread(&nColumns, sizeof(int), 1, pFile);
#endif	



		nSize = nRows * nColumns;
		
		(*data)[i].nID = i;
		
		if (!strcmp(sDataPath, "D:\\Data\\IR\\bin\\ir_train.bin"))
		{
			(*data)[i].nLabelID = ir_train[nLabelID].nLabelID;
			strcpy((*data)[i].sLabel, ir_train[nLabelID].sLabel);
		}
		else if (!strcmp(sDataPath, "D:\\Data\\IR\\bin\\ir_mmstd_03.bin"))
		{
			(*data)[i].nLabelID = ir_mmstd_03[nLabelID].nLabelID;
			strcpy((*data)[i].sLabel, ir_mmstd_03[nLabelID].sLabel);
		}
		else
		{
			HoldDisplay("Error: Missing structLabel\n");
		}

		nArray = (int *)calloc(nSize, sizeof(int));

		for (j=0; j<nSize; ++j)
		{
			
			
#ifdef _WINDOWS
			fread(&nValue, sizeof(unsigned __int16), 1, pFile);
#endif

#ifdef _LINUX
			fread(&nValue, sizeof(int), 1, pFile);
#endif	

			
			
			nArray[j] = (int)nValue;

			if (nArray[j] < nMin)
				nMin = nArray[j];
			if (nArray[j] > nMax)
				nMax = nArray[j];
		}

		nDelta = nMin;

		fScale = 255.0f / (nMax - nMin);

		nMin = 99999;
		nMax = 0;

		for (j = 0; j<nSize; ++j)
		{
			nArray[j]-= nDelta;
			nArray[j] = (int)((float)nArray[j] * fScale);
		}

		if (nColumns > nRows)
		{
			int nOffset = (nColumns - nRows) / 2;

			(*data)[i].nIntensity = (int *)calloc((nColumns * nColumns), sizeof(int));

			for (p = 0, r = 0; p<nColumns; ++p)
			{
				for (q = 0; q<nRows; ++q, ++r)
				{
					s = ((p + nOffset) * nRows) + q;
					
					(*data)[i].nIntensity[s] = nArray[r];
				}
			}

			free(nArray);
			nArray = (int *)calloc((64 * 64), sizeof(int));

			ResizeImage_Bicubic((*data)[i].nIntensity, nRows, nColumns, nArray, 64, 64);

			free((*data)[i].nIntensity);
			(*data)[i].nIntensity = (int *)calloc((64 * 64), sizeof(int));

			nRows = nColumns;
		}
		else
		{
			int nOffset = (nRows - nColumns) / 2;

			(*data)[i].nIntensity = (int *)calloc((nRows * nRows), sizeof(int));

			for (p = 0, r = 0; p<nRows; ++p)
			{
				for (q = 0; q<nColumns; ++q, ++r)
				{
					(*data)[i].nIntensity[r + nOffset] = nArray[r];
				}
			}	
			
			free(nArray);
			nArray = (int *)calloc((64 * 64), sizeof(int));

			ResizeImage_Bicubic((*data)[i].nIntensity, nRows, nColumns, nArray, 64, 64);

			free((*data)[i].nIntensity);
			(*data)[i].nIntensity = (int *)calloc((64 * 64), sizeof(int));

			nColumns = nRows;
		}

		for (p = 0, r = 0; p<64; ++p)
		{
			for (q = 0; q<64; ++q, ++r)
			{
				(*data)[i].nIntensity[r] = nArray[r];
			}
		}

		free(nArray);
	}

	fclose(pFile);


	return(nInputCount);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void ReadFile_InputFileIR(structInput *input, char *sDataSource)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	FILE		*pFile;
	float		fValue;
	float				fMax;
	float				fMin;
	int			nImageIndex;
	int			nClassID;
	int			nIndex;
	int			j, k;

	fMax = -10.0f;
	fMin = 10.0f;

	while ((pFile = FOpenMakeDirectory(input->sPath, "rb")) == NULL)
	{
		printf("Error ReadFile_InputFileIR(): Could not find the input file -- %s\n\n", input->sPath);
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
		fread(input->data[nImageIndex].sLabel, sizeof(char), 32, pFile);

		for (j = 0; j<input->nRowCount; ++j)
		{
			for (k = 0; k<input->nColumnCount; ++k)
			{
				nIndex = j * input->nColumnCount + k;
				fread(&fValue, sizeof(float), 1, pFile);

				input->data[nImageIndex].fIntensity[nIndex] = fValue;

				if (input->data[nImageIndex].fIntensity[nIndex] > fMax)
					fMax = input->data[nImageIndex].fIntensity[nIndex];
				if (input->data[nImageIndex].fIntensity[nIndex] < fMin)
					fMin = input->data[nImageIndex].fIntensity[nIndex];
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

