#include "main.h"

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void Get_InputData(structInput **inputData, structInput **inputTrainingData, structInput **inputVerifyData, structInput **inputTestingData, int nTrainingVerifySplit, char *sDrive, char *sTrainingFilePath, char *sTestingFilePath, char *sDataSource, structClass **classHead, int *nClassCount, int *nDataSource, int *nRowCount, int *nColumnCount)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structClass		*classCur;
	int				nCount = 0;

	printf("\nReading Input Data ");

	if (((*inputData) = (structInput *)calloc(1, sizeof(structInput))) == NULL)
		exit(0);
	if (((*inputTrainingData) = (structInput *)calloc(1, sizeof(structInput))) == NULL)
		exit(0);
	if (((*inputVerifyData) = (structInput *)calloc(1, sizeof(structInput))) == NULL)
		exit(0);
	if (((*inputTestingData) = (structInput *)calloc(1, sizeof(structInput))) == NULL)
		exit(0);

	// Read Training Data

	printf(".");
	if (!strcmp(sDataSource, "ISAR"))
	{
		ReadIntermediateFile_InputFileSAR(sTrainingFilePath, sTestingFilePath);
	}
	else if (!strcmp(sDataSource, "IR"))
	{
		ReadIntermediateFile_InputFileIR(sDrive, sTrainingFilePath, sTestingFilePath);
	}

	printf(".");
	sprintf((*inputData)->sPath, "%s", sTrainingFilePath);
	if (!strcmp(sDataSource, "AR2"))
	{
		ReadAR2File_InputData(&(*inputData)->data, (*inputData)->sPath, &(*inputData)->nInputCount, 0);
		(*inputData)->nDataSource = 700;
		(*inputData)->nChannels = 1;
		(*inputData)->nRowCount = 63;
		(*inputData)->nColumnCount = 1;
		(*inputData)->nSize = (*inputData)->nRowCount * (*inputData)->nColumnCount;
	}
	else if (!strcmp(sDataSource, "IR"))
	{
		ReadFile_InputFileIR((*inputData), sDataSource);
	}
	else if (!strcmp(sDataSource, "ISAR"))
	{
		ReadFile_InputFileSAR((*inputData), sDataSource);
	}
	else if (!strcmp(sDataSource, "CIFAR"))
	{
		ReadCIFARFile((*inputData), sDataSource);
	}
	else if (!strcmp(sDataSource, "IMAGENETTE"))
	{
		ReadImagenetteFile((*inputData), sDataSource);
	}
	else
	{
		ReadFile_InputData((*inputData), sDataSource);
	}

	printf(".");
	SiftClasses_InputData((*inputData), classHead, TRAINING);

	if (((*inputData)->nAverageIDArray = (int *)calloc((*inputData)->nClassCount, sizeof(int))) == NULL)
		exit(0);
	if (((*inputData)->nMaxIDArray = (int *)calloc((*inputData)->nClassCount, sizeof(int))) == NULL)
		exit(0);

	if (((*inputData)->fRatioArray = (float *)calloc((*inputData)->nClassCount, sizeof(float))) == NULL)
		exit(0);


	SplitData_InputData((*inputData), inputTrainingData, inputVerifyData, nTrainingVerifySplit, 100);

	printf(".");
	SiftClasses_InputData((*inputTrainingData), classHead, TRAINING);
	SiftClasses_InputData((*inputVerifyData), classHead, VERIFY);


	// Read Testing Data
	sprintf((*inputTestingData)->sPath, "%s", sTestingFilePath);
	if (!strcmp(sDataSource, "AR2"))
	{
		ReadAR2File_InputData(&(*inputTestingData)->data, (*inputTestingData)->sPath, &(*inputTestingData)->nInputCount, 0);
		(*inputTestingData)->nDataSource = 700;
		(*inputTestingData)->nChannels = 1;
		(*inputTestingData)->nRowCount = 63;
		(*inputTestingData)->nColumnCount = 1;
		(*inputTestingData)->nSize = (*inputTestingData)->nRowCount * (*inputTestingData)->nColumnCount;
	}
	else if (!strcmp(sDataSource, "IR"))
	{
		ReadFile_InputFileIR((*inputTestingData), sDataSource);
	}
	else if (!strcmp(sDataSource, "ISAR"))
	{
		ReadFile_InputFileSAR((*inputTestingData), sDataSource);
	}
	else if (!strcmp(sDataSource, "CIFAR"))
	{
		ReadCIFARFile((*inputTestingData), sDataSource);
	}
	else if (!strcmp(sDataSource, "IMAGENETTE"))
	{
		ReadImagenetteFile((*inputTestingData), sDataSource);
	}
	else
	{
		ReadFile_InputData((*inputTestingData), sDataSource);
	}

	if (!strcmp(sDataSource, "IMAGENETTE"))
	{
		float	fMax = 0.0f;
		float	fMin = 0.0f;
		float	fMultiplier = 0.0f;
		int		i, j;
		
		for (i = 0; i < (*inputData)->nInputCount; ++i)
		{
			for (j = 0; j < (*inputData)->nSize; ++j)
			{
				if ((*inputData)->data[i].fIntensity[j] > fMax)
					fMax = (*inputData)->data[i].fIntensity[j];
				if ((*inputData)->data[i].fIntensity[j] < fMin)
					fMin = (*inputData)->data[i].fIntensity[j];
			}
		}

		for (i = 0; i < (*inputTestingData)->nInputCount; ++i)
		{
			for (j = 0; j < (*inputTestingData)->nSize; ++j)
			{
				if ((*inputTestingData)->data[i].fIntensity[j] > fMax)
					fMax = (*inputTestingData)->data[i].fIntensity[j];
				if ((*inputTestingData)->data[i].fIntensity[j] < fMin)
					fMin = (*inputTestingData)->data[i].fIntensity[j];
			}
		}


		for (i = 0; i < (*inputTrainingData)->nInputCount; ++i)
		{
			for (j = 0; j < (*inputTrainingData)->nSize; ++j)
				(*inputTrainingData)->data[i].fIntensity[j]=2.0f * (((*inputTrainingData)->data[i].fIntensity[j] - fMin)/(fMax- fMin)) - 1.0f;
		}

		for (i = 0; i < (*inputVerifyData)->nInputCount; ++i)
		{
			for (j = 0; j < (*inputVerifyData)->nSize; ++j)
				(*inputVerifyData)->data[i].fIntensity[j] = 2.0f * (((*inputVerifyData)->data[i].fIntensity[j] - fMin) / (fMax - fMin)) - 1.0f;
		}

		for (i = 0; i < (*inputTestingData)->nInputCount; ++i)
		{
			for (j = 0; j < (*inputTestingData)->nSize; ++j)
				(*inputTestingData)->data[i].fIntensity[j] = 2.0f * (((*inputTestingData)->data[i].fIntensity[j] - fMin) / (fMax - fMin)) - 1.0f;
		}
	}







	printf(".");
	SiftClasses_InputData((*inputTestingData), classHead, TESTING);


	*nClassCount = 0;
	for (classCur = *classHead; classCur != NULL; classCur = classCur->next)
		++(*nClassCount);

	*nDataSource = (*inputData)->nDataSource;
	*nRowCount = (*inputData)->nRowCount;
	*nColumnCount = (*inputData)->nColumnCount;

	printf(" Done\r");

	return;

}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int ReadAR2File_InputData(structInputData **data, char *sPath, int *nCount, int nLabelID)
{
	FILE		*pFile;
	int			nIndex;
	int			i;
	int			nImageID = 0;

	if ((pFile = FOpenMakeDirectory(sPath, "rb")) == NULL)
	{
		printf("Error ReadAR2File_InputData(): Could not find the input file -- %s\n\n", sPath);
		while (1);
	}

	if (((*data) = (structInputData *)calloc(8800, sizeof(structInputData))) == NULL)
		exit(0);

	while (!feof(pFile))
	{
		nIndex = 0;

		 nImageID++;
		fscanf(pFile, "%d %s %s %d %d ", &(*data)[*nCount].nID, (*data)[*nCount].sDescription, (*data)[*nCount].sLabel, &(*data)[*nCount].nLabelID, &(*data)[*nCount].nSequence);

		(*data)[*nCount].fIntensity = (float *)calloc(63, sizeof(float));
		(*data)[*nCount].fx32Intensity = (fx32 *)calloc(63, sizeof(fx32));

		for (i = 0; i < 62; ++i)
			fscanf(pFile, "%f,", &(*data)[*nCount].fIntensity[i]);

		fscanf(pFile, "%f\n", &(*data)[*nCount].fIntensity[i]);

		++(*nCount);
	}

	fclose(pFile);

	return(nImageID);
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void SplitData_InputData(structInput *input, structInput **inputTrainingData, structInput **inputVerifyData, int nPercent, int nTotalDataPercent)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	float	fPercent = (float)nPercent / 100.0f;
	float	fTotalDataPercent = (float)nTotalDataPercent / 100.0f;
	float	fMax;
	float	fMin;
	int		nImageIndex;
	int		*nClassMemberCountArray;
	int		i, j;
	
	if ((nClassMemberCountArray = (int *)calloc(input->nClassCount, sizeof(int))) == NULL)
		exit(0);
	
///////////////////////////////////////////////
	
	if ((*inputTrainingData) != NULL)
		DeleteData_InputData(inputTrainingData);

	if (((*inputTrainingData) = (structInput *)calloc(1, sizeof(structInput))) == NULL)
		exit(0);

	(*inputTrainingData)->nClassCount = input->nClassCount;

	if (((*inputTrainingData)->nAverageIDArray = (int *)calloc((*inputTrainingData)->nClassCount, sizeof(int))) == NULL)
		exit(0);
	if (((*inputTrainingData)->nMaxIDArray = (int *)calloc((*inputTrainingData)->nClassCount, sizeof(int))) == NULL)
		exit(0);

	if (((*inputTrainingData)->fRatioArray = (float *)calloc((*inputTrainingData)->nClassCount, sizeof(float))) == NULL)
		exit(0);

	if (((*inputTrainingData)->nClassMemberCount = (int *)calloc((*inputTrainingData)->nClassCount, sizeof(int))) == NULL)
		exit(0);

///////////////////////////////////////////////

	if ((*inputVerifyData) != NULL)
		DeleteData_InputData(inputVerifyData);
	
	if (((*inputVerifyData) = (structInput *)calloc(1, sizeof(structInput))) == NULL)
			exit(0);

	(*inputVerifyData)->nClassCount=input->nClassCount;


	if (((*inputVerifyData)->nClassMemberCount = (int *)calloc((*inputVerifyData)->nClassCount, sizeof(int))) == NULL)
		exit(0);

	for (j = 0; j < input->nInputCount; ++j)
	{
		if(input->data[j].bTrained == 0)
			++nClassMemberCountArray[input->data[j].nLabelID];
	}

	(*inputTrainingData)->nInputCount = 0;
	for (i = 0; i < input->nClassCount; ++i)
	{
		(*inputTrainingData)->nClassMemberCount[i] = (int)((float)((float)input->nClassMemberCount[i] * fTotalDataPercent) * fPercent); 
		(*inputVerifyData)->nClassMemberCount[i] = (int)((float)input->nClassMemberCount[i] * fTotalDataPercent) - (*inputTrainingData)->nClassMemberCount[i];

		if (((*inputTrainingData)->nClassMemberCount[i] + (*inputVerifyData)->nClassMemberCount[i]) > nClassMemberCountArray[i])
		{
			(*inputTrainingData)->nClassMemberCount[i] = (int)((float)((float)nClassMemberCountArray[i]) * fPercent);
		}
		
		(*inputTrainingData)->nInputCount += (*inputTrainingData)->nClassMemberCount[i];
	}

	for (i = 0; i < input->nClassCount; ++i)
	{
		(*inputVerifyData)->nClassMemberCount[i] = (int)((float)input->nClassMemberCount[i] * fTotalDataPercent) - (*inputTrainingData)->nClassMemberCount[i];


		if (((*inputTrainingData)->nClassMemberCount[i] + (*inputVerifyData)->nClassMemberCount[i]) > nClassMemberCountArray[i])
		{
			(*inputVerifyData)->nClassMemberCount[i] = nClassMemberCountArray[i] - (*inputTrainingData)->nClassMemberCount[i];
		}

		(*inputVerifyData)->nInputCount += (*inputVerifyData)->nClassMemberCount[i];
	}


	(*inputTrainingData)->nDataSource = input->nDataSource;
	(*inputTrainingData)->nChannels = input->nChannels;
	(*inputTrainingData)->nRowCount = input->nRowCount;
	(*inputTrainingData)->nColumnCount = input->nColumnCount;

	(*inputTrainingData)->nSize = (*inputTrainingData)->nRowCount * (*inputTrainingData)->nColumnCount;

	if (((*inputTrainingData)->data = (structInputData *)calloc((*inputTrainingData)->nInputCount, sizeof(structInputData))) == NULL)
		exit(0);

	for (j = 0; j < (*inputTrainingData)->nInputCount; ++j)
		(*inputTrainingData)->data[j].nLabelID = -1;


	fMax=-1.0f;
	fMin = 1.0f;

	for (i = 0, nImageIndex=0; i<input->nInputCount; ++i)
	{
		if ((*inputTrainingData)->nClassMemberCount[input->data[i].nLabelID] > 0 && input->data[i].bTrained == 0)
		{
			(*inputTrainingData)->data[nImageIndex].fIntensity = (float *)calloc((*inputTrainingData)->nSize, sizeof(float));

			(*inputTrainingData)->data[nImageIndex].nID = nImageIndex;
			(*inputTrainingData)->data[nImageIndex].nLabelID = input->data[i].nLabelID;
			strcpy((*inputTrainingData)->data[nImageIndex].sLabel, input->data[i].sLabel);
			(*inputTrainingData)->data[nImageIndex].nGroupA = input->data[i].nGroupA;
			(*inputTrainingData)->data[nImageIndex].nGroupB = input->data[i].nGroupB;

			for (j = 0; j < (*inputTrainingData)->nSize; ++j)
			{
				(*inputTrainingData)->data[nImageIndex].fIntensity[j] = input->data[i].fIntensity[j];
				
				if ((*inputTrainingData)->data[nImageIndex].fIntensity[j] > fMax)
					fMax= (*inputTrainingData)->data[nImageIndex].fIntensity[j];
				if ((*inputTrainingData)->data[nImageIndex].fIntensity[j] < fMin)
					fMin = (*inputTrainingData)->data[nImageIndex].fIntensity[j];
			}

			--(*inputTrainingData)->nClassMemberCount[input->data[i].nLabelID];
			input->data[i].bTrained = 2;
			(*inputTrainingData)->data[nImageIndex].bTrained = 0;

			++nImageIndex;
		}
	}

	for (i = 0; i < input->nClassCount; ++i)
	{
		(*inputTrainingData)->nClassMemberCount[i] = 0;

		for (j=0; j<(*inputTrainingData)->nInputCount; ++j)
		{
			if ((*inputTrainingData)->data[j].nLabelID == i)
				++(*inputTrainingData)->nClassMemberCount[i];
		}
	}

	(*inputTrainingData)->nInputCount = 0;
	for (i = 0; i < input->nClassCount; ++i)
		(*inputTrainingData)->nInputCount += (*inputTrainingData)->nClassMemberCount[i];



	(*inputVerifyData)->nDataSource = input->nDataSource;
	(*inputVerifyData)->nChannels = input->nChannels;
	(*inputVerifyData)->nRowCount = input->nRowCount;
	(*inputVerifyData)->nColumnCount = input->nColumnCount;

	(*inputVerifyData)->nSize = (*inputVerifyData)->nRowCount * (*inputVerifyData)->nColumnCount;

	if (((*inputVerifyData)->data = (structInputData *)calloc((*inputVerifyData)->nInputCount, sizeof(structInputData))) == NULL)
		exit(0);

	for (j = 0; j < (*inputVerifyData)->nInputCount; ++j)
		(*inputVerifyData)->data[j].nLabelID = -1;


	for (i = 0, nImageIndex = 0; i<input->nInputCount; ++i)
	{
		if ((*inputVerifyData)->nClassMemberCount[input->data[i].nLabelID] > 0 && input->data[i].bTrained == 0)
		{
			(*inputVerifyData)->data[nImageIndex].fIntensity = (float *)calloc((*inputVerifyData)->nSize, sizeof(float));

			(*inputVerifyData)->data[nImageIndex].nID = nImageIndex;
			(*inputVerifyData)->data[nImageIndex].nLabelID = input->data[i].nLabelID;
			strcpy((*inputVerifyData)->data[nImageIndex].sLabel, input->data[i].sLabel);
			(*inputVerifyData)->data[nImageIndex].nGroupA = input->data[i].nGroupA;
			(*inputVerifyData)->data[nImageIndex].nGroupB = input->data[i].nGroupB;

			for (j = 0; j < (*inputVerifyData)->nSize; ++j)
			{
				(*inputVerifyData)->data[nImageIndex].fIntensity[j] = input->data[i].fIntensity[j];
			}

			--(*inputVerifyData)->nClassMemberCount[input->data[i].nLabelID];
			input->data[i].bTrained = 2;
			(*inputVerifyData)->data[nImageIndex].bTrained = 0;

			++nImageIndex;
		}
	}

	for (i = 0; i < input->nClassCount; ++i)
	{
		(*inputVerifyData)->nClassMemberCount[i] = 0;

		for (j = 0; j<(*inputVerifyData)->nInputCount; ++j)
		{
			if ((*inputVerifyData)->data[j].nLabelID == i)
				++(*inputVerifyData)->nClassMemberCount[i];
		}
	}

	(*inputVerifyData)->nInputCount = 0;
	for (i = 0; i < input->nClassCount; ++i)
		(*inputVerifyData)->nInputCount += (*inputVerifyData)->nClassMemberCount[i];

	for (i = 0, nImageIndex = 0; i < input->nInputCount; ++i)
		if(input->data[i].bTrained == 2)
			input->data[i].bTrained = 0;

	free(nClassMemberCountArray);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void ReadFile_InputData(structInput *input, char *sDataSource)
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
		printf("\n\nError ReadFile_InputData(): Could not find the input file -- %s\n\n", input->sPath);
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
		input->data[nImageIndex].fx32Intensity = (fx32 *)calloc(input->nSize, sizeof(fx32));
		input->data[nImageIndex].fIntensity = (float *)calloc(input->nSize, sizeof(float));

		input->data[nImageIndex].nID = nImageIndex;
		fread(&nClassID, sizeof(int), 1, pFile);
		fread(&input->data[nImageIndex].nGroupA, sizeof(int), 1, pFile);
		fread(&input->data[nImageIndex].nGroupB, sizeof(int), 1, pFile);
		
		if (!strcmp(sDataSource, "ISAR"))
		{
			fread(&input->data[nImageIndex].fAspect, sizeof(float), 1, pFile);
			fread(&input->data[nImageIndex].fRadialSpeed, sizeof(float), 1, pFile);
			fread(&input->data[nImageIndex].fSpread, sizeof(float), 1, pFile);
		}
		
		sprintf(input->data[nImageIndex].sLabel, "%d", nClassID);

		for (j = 0; j<input->nRowCount; ++j)
		{
			for (k = 0; k<input->nColumnCount; ++k)
			{
				nIndex = j * input->nColumnCount + k;
				fread(&fValue, sizeof(float), 1, pFile);

				input->data[nImageIndex].fIntensity[nIndex] = fValue;
				input->data[nImageIndex].fx32Intensity[nIndex] = FloatToFx32(input->data[nImageIndex].fIntensity[nIndex]);
			}
		}
	}

	fclose(pFile);

	return;
}




/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void DeleteData_InputData(structInput **input)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	int		i;

	for (i=0; i<(*input)->nInputCount; ++i)
	{
		free((*input)->data[i].fIntensity);
		free((*input)->data[i].fx32Intensity);
	}

	if((*input)->nMaxIDArray !=NULL)
		free((*input)->nMaxIDArray);
	
	if((*input)->nClassMemberCount != NULL)
		free((*input)->nClassMemberCount);
	
	if ((*input)->nAverageIDArray != NULL)
		free((*input)->nAverageIDArray);
	
	if ((*input)->fRatioArray != NULL)
		free((*input)->fRatioArray);

	free(*input);
	*input = NULL;

	return;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void GetAverages_InputData(structInput *input, structClass *classHead)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structClass	*classCur;
	float		**fPixelTotal;
	float		fMaxError;
	float		fMinError;
	float		fTemp;
	float		fA, fB;
	int			i, j, k;

	if ((fPixelTotal = (float **)calloc(input->nClassCount, sizeof(float *))) == NULL)
		exit(0);

	for (i = 0; i < input->nClassCount; ++i)
	{
		if ((fPixelTotal[i] = (float *)calloc(input->nSize, sizeof(float))) == NULL)
			exit(0);
	}

	if ((input->nAverageIDArray = (int *)calloc(input->nClassCount, sizeof(int))) == NULL)
		exit(0);
	if ((input->nMaxIDArray = (int *)calloc(input->nClassCount, sizeof(int))) == NULL)
		exit(0);

	if ((input->fRatioArray = (float *)calloc(input->nClassCount, sizeof(float))) == NULL)
		exit(0);

	for (i = 0; i < input->nInputCount; ++i)
	{
		for (j = 0; j<input->nSize; ++j)
		{
			fPixelTotal[input->data[i].nLabelID][j] += input->data[i].fIntensity[j];
		}
	}

	for (classCur = classHead; classCur != NULL; classCur = classCur->next)
	{
		if (input->nClassMemberCount[classCur->nID] > 0)
		{
			for (i = 0; i < input->nSize; ++i)
			{
				fPixelTotal[classCur->nID][i] /= (float)input->nClassMemberCount[classCur->nID];
			}
		}
	}



	for (classCur = classHead; classCur != NULL; classCur = classCur->next)
	{
		fMaxError = -999999.0f;
		fMinError = 999999.0f;

		for (j = 0; j < input->nInputCount; ++j)
		{
			if (input->data[j].nLabelID == classCur->nID)
			{
				fTemp = 0.0;

				for (k = 0; k < input->nSize; ++k)
				{
					fA = fPixelTotal[classCur->nID][k];
					fB = input->data[j].fIntensity[k];

					fTemp += ((fA - fB) * (fA - fB));
				}

				input->data[j].fError = (1.0f / (input->nSize*input->nSize))*fTemp;
				//input->data[j].fError = ComputeSSIM_InputData(fPixelTotal[classCur->nID], input->data[j].fIntensity, input->nSize);

				if (input->data[j].fError < fMinError)
				{
					fMinError = input->data[j].fError;
					input->nAverageIDArray[input->data[j].nLabelID] = j;
				}

				if (input->data[j].fError > fMaxError)
				{
					fMaxError = input->data[j].fError;
					input->nMaxIDArray[input->data[j].nLabelID] = j;
				}
			}
		}
	}

	for (i = 0; i < input->nClassCount; ++i)
		free(fPixelTotal[i]);
	
	free(fPixelTotal);

	return;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
float ComputeSSIM_InputData(float *average, float *currPic, int num_pixels)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	float	SSIM = 0.0;
	float	C1 = 0.0;
	float	C2 = 0.0;
	float	mean_x = 0.0;
	float	mean_y = 0.0;
	float	std_x = 0.0;
	float	std_y = 0.0;
	float	cov = 0.0;
	float	temp_sum_average = 0.0;
	float	temp_sum_pic = 0.0;
	float	temp_sum_cov = 0.0;
	int		i = 0;


	/*
	Calculating the mean of both images
	*/
	for (i = 0; i < num_pixels; i++)
	{
		temp_sum_average += average[i];
		temp_sum_pic += currPic[i];
	}

	mean_x = temp_sum_average / num_pixels;
	mean_y = temp_sum_pic / num_pixels;

	/*
	Calculating the standard deviation and covariance of both images
	*/
	temp_sum_average = 0.0;
	temp_sum_pic = 0.0;
	for (i = 0; i < num_pixels; i++)
	{
		temp_sum_average += (average[i] - mean_x) * (average[i] - mean_x);
		temp_sum_pic += (currPic[i] - mean_y) * (currPic[i] - mean_y);
		temp_sum_cov += (average[i] - mean_x) * (currPic[i] - mean_y);
	}

	std_x = (float)sqrt(temp_sum_average / (num_pixels - 1));
	std_y = (float)sqrt(temp_sum_pic / (num_pixels - 1));
	cov = temp_sum_cov / (num_pixels - 1);



	SSIM = ((2 * mean_x * mean_y + C1) * (2 * cov + C2)) / ((mean_x * mean_x + mean_y * mean_y + C1) * (std_x * std_x + std_y * std_y + C2));

	return SSIM;
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int ExpandData_InputData(structInput *inputDataSource, structInput **inputDataDestination, int nCount, int nRow, int nColumn, int nMultiplier, float fAngleIncrement, float fStartScale, float fEndScale)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	float	fAngle;
	float	fScale;
	float	fStartAngle = 0.0f;
	float	fEndAngle = 0.0f;
	float	fScaleIncrement = 0.0f;
	int		nSize = nCount * nMultiplier * nMultiplier;
	int		nImageSize = nRow * nColumn;
	int		i, j, k, u, n;

	// Adjust Scale
	if ((fEndScale - fStartScale) > 0.0f && nMultiplier > 2)
		fScaleIncrement = (fEndScale - fStartScale) / (float)(nMultiplier - 1);

	if (nMultiplier > 1)
	{
		if (fAngleIncrement > 0.0f)
		{
			fStartAngle = -(((float)nMultiplier * fAngleIncrement) / 2.0f);
			fEndAngle = (((float)nMultiplier * fAngleIncrement) / 2.0f);
		}

	}


	if ((*inputDataDestination = (structInput *)calloc(1, sizeof(structInput))) == NULL)
		exit(0);

	if (((*inputDataDestination)->data = (structInputData *)calloc(nSize, sizeof(structInputData))) == NULL)
		exit(0);


	for (i = 0, n = 0, fScale = fStartScale; i<nMultiplier && fScale <= fEndScale; ++i, fScale += fScaleIncrement)
	{
		for (j = 0, fAngle = fStartAngle; j<nMultiplier && fAngle <= fEndAngle; ++j, fAngle += fAngleIncrement)
		{
			for (k = 0; k<nCount; ++k, ++n)
			{
				(*inputDataDestination)->data[n].nDataSource = inputDataSource->data[inputDataSource->nAverageIDArray[k]].nDataSource;
				(*inputDataDestination)->data[n].nLabelID = inputDataSource->data[inputDataSource->nAverageIDArray[k]].nLabelID;
				(*inputDataDestination)->data[n].nGroupA = inputDataSource->data[inputDataSource->nAverageIDArray[k]].nGroupA;
				(*inputDataDestination)->data[n].nGroupB = inputDataSource->data[inputDataSource->nAverageIDArray[k]].nGroupB;
				(*inputDataDestination)->data[n].bTrained = inputDataSource->data[inputDataSource->nAverageIDArray[k]].bTrained;

				(*inputDataDestination)->data[n].fIntensity = (float *)calloc(nImageSize, sizeof(float));
				(*inputDataDestination)->data[n].fx32Intensity = (fx32 *)calloc(nImageSize, sizeof(fx32));

				RotateDataWithClip_InputData(inputDataSource->data[inputDataSource->nAverageIDArray[k]].fIntensity, (*inputDataDestination)->data[n].fIntensity, nRow, nColumn, fAngle, fScale);

				for (u = 0; u<nImageSize; ++u)
				{
					(*inputDataDestination)->data[n].fx32Intensity[u] = FloatToFx32((*inputDataDestination)->data[n].fIntensity[u]);
				}
			}
		}
	}

	return(n);
}





/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int ExpandClassData_InputData(structClass *classData, structInput *inputDataSource, structInput **inputDataDestination, int nMultiplier, float fAngleIncrement, float fStartScale, float fEndScale, int nLabelID, int nSize, int *nIndexArray)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	float		fAngle;
	float		fScale;
	float		fStartAngle = 0.0f;
	float		fEndAngle = 0.0f;
	float		fScaleIncrement = 0.0f;
	int			nNotClassCount=0;
	int			i, j, k, u, n;
	

	// Adjust Scale
	if ((fEndScale - fStartScale) > 0.0f && nMultiplier > 2)
		fScaleIncrement = (fEndScale - fStartScale) / (float)(nMultiplier - 1);

	if (nMultiplier > 1)
	{
		if (fAngleIncrement > 0.0f)
		{
			fStartAngle = -(((float)nMultiplier * fAngleIncrement) / 2.0f);
			fEndAngle = (((float)nMultiplier * fAngleIncrement) / 2.0f);
		}

	}

	for (i = 0, n = 0, fScale = fStartScale; i < nMultiplier && fScale <= fEndScale; ++i, fScale += fScaleIncrement)
		for (j = 0, fAngle = fStartAngle; j < nMultiplier && fAngle <= fEndAngle; ++j, fAngle += fAngleIncrement, ++n);

	for (i=0; i<inputDataSource->nClassCount; ++i)
		if(i != nLabelID)
			nNotClassCount += inputDataSource->nClassMemberCount[i];
	
	if ((*inputDataDestination = (structInput *)calloc(1, sizeof(structInput))) == NULL)
		exit(0);

	if (((*inputDataDestination)->data = (structInputData *)calloc(nSize, sizeof(structInputData))) == NULL)
		exit(0);

	(*inputDataDestination)->nDataSource = inputDataSource->nDataSource;
	(*inputDataDestination)->nInputCount = nSize;
	(*inputDataDestination)->nChannels = inputDataSource->nChannels;
	(*inputDataDestination)->nRowCount = inputDataSource->nRowCount;
	(*inputDataDestination)->nColumnCount = inputDataSource->nColumnCount;
	(*inputDataDestination)->nSize = inputDataSource->nSize;


	for (k = 0, n=0; k<inputDataSource->nInputCount; ++k, ++n)
	{
		(*inputDataDestination)->data[nIndexArray[k]].nDataSource = inputDataSource->data[k].nDataSource;
		strcpy((*inputDataDestination)->data[nIndexArray[k]].sLabel, inputDataSource->data[k].sLabel);
		(*inputDataDestination)->data[nIndexArray[k]].nLabelID = inputDataSource->data[k].nLabelID;
		(*inputDataDestination)->data[nIndexArray[k]].nGroupA = inputDataSource->data[k].nGroupA;
		(*inputDataDestination)->data[nIndexArray[k]].nGroupB = inputDataSource->data[k].nGroupB;
		(*inputDataDestination)->data[nIndexArray[k]].bTrained = inputDataSource->data[k].bTrained;

		(*inputDataDestination)->data[nIndexArray[k]].fIntensity = (float *)calloc(inputDataSource->nSize, sizeof(float));
		(*inputDataDestination)->data[nIndexArray[k]].fx32Intensity = (fx32 *)calloc(inputDataSource->nSize, sizeof(fx32));


		for (u = 0; u<inputDataSource->nSize; ++u)
		{
			(*inputDataDestination)->data[nIndexArray[k]].fIntensity[u] = inputDataSource->data[k].fIntensity[u];
			(*inputDataDestination)->data[nIndexArray[k]].fx32Intensity[u] = inputDataSource->data[k].fx32Intensity[u];
		}
	}

	for (i = 0, fScale = fStartScale; i<nMultiplier && fScale <= fEndScale && nNotClassCount > 0; ++i, fScale += fScaleIncrement)
	{
		for (j = 0, fAngle = fStartAngle; j<nMultiplier && fAngle <= fEndAngle && nNotClassCount > 0; ++j, fAngle += fAngleIncrement)
		{
			for (k = 0; k<inputDataSource->nInputCount && nNotClassCount > 0; ++k)
			{
				if (inputDataSource->data[k].nLabelID == nLabelID)
				{
					(*inputDataDestination)->data[nIndexArray[n]].nDataSource = inputDataSource->data[k].nDataSource;
					strcpy((*inputDataDestination)->data[nIndexArray[n]].sLabel, inputDataSource->data[k].sLabel);
					(*inputDataDestination)->data[nIndexArray[n]].nLabelID = inputDataSource->data[k].nLabelID;
					(*inputDataDestination)->data[nIndexArray[n]].nGroupA = inputDataSource->data[k].nGroupA;
					(*inputDataDestination)->data[nIndexArray[n]].nGroupB = inputDataSource->data[k].nGroupB;
					(*inputDataDestination)->data[nIndexArray[n]].bTrained = inputDataSource->data[k].bTrained;

					(*inputDataDestination)->data[nIndexArray[n]].fIntensity = (float *)calloc(inputDataSource->nSize, sizeof(float));
					(*inputDataDestination)->data[nIndexArray[n]].fx32Intensity = (fx32 *)calloc(inputDataSource->nSize, sizeof(fx32));

					RotateDataWithClip_InputData(inputDataSource->data[k].fIntensity, (*inputDataDestination)->data[nIndexArray[n]].fIntensity, inputDataSource->nRowCount, inputDataSource->nColumnCount, fAngle, fScale);

					for (u = 0; u < inputDataSource->nSize; ++u)
					{
						(*inputDataDestination)->data[nIndexArray[n]].fx32Intensity[u] = FloatToFx32((*inputDataDestination)->data[nIndexArray[n]].fIntensity[u]);
					}

					++n;
					--nNotClassCount;
				}
			}
		}
	}

	SiftClasses_InputData((*inputDataDestination), &classData, TRAINING);

	return(n);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void RotateDataWithClip_InputData(float *pSrcBase, float *pDstBase, int nRow, int nColumn, float fAngle, float fScale)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	float	duCol = (float)sin(-fAngle) * (1.0f / fScale);
	float	dvCol = (float)cos(-fAngle) * (1.0f / fScale);
	float	duRow = dvCol;
	float	dvRow = -duCol;
	float	fRowCenter = (float)nRow / 2.0f;
	float	fColumnCenter = (float)nColumn / 2.0f;
	float	rowu;
	float	rowv;
	float	*pDst;
	float	*pSrc;
	int		x, y;
	float	u;
	float	v;


	rowu = fRowCenter - (fRowCenter * dvCol + fColumnCenter * duCol);
	rowv = fColumnCenter - (fRowCenter * dvRow + fColumnCenter * duRow);

	for (y = 0; y < nColumn; y++)
	{
		u = rowu;
		v = rowv;

		pDst = pDstBase + (nRow * y);

		for (x = 0; x < nRow; x++)
		{
			if (u>0 && v>0 && u<nRow && v<nColumn)
			{
				pSrc = pSrcBase + (int)u + ((int)v * nColumn);

				*pDst++ = *pSrc++;
			}
			else
			{
				*pDst++ = *pSrcBase;
			}

			u += duRow;
			v += dvRow;
		}

		rowu += duCol;
		rowv += dvCol;
	}
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void ReduceDataSet_InputData(structInput *inputSource, structInput **inputDestination, structClass *classHead, int nClassMemberCount)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structClass *classCur;
	structInput	*inputTemp;
	int			*nSkipArray;
	int			**nMemberArray;
	int			nImageIndex;
	int			nIncrement;
	int			i, j, k;


	if ((nSkipArray = (int *)calloc(inputSource->nClassCount, sizeof(int))) == NULL)
		exit(0);

	if ((nMemberArray = (int **)calloc(inputSource->nClassCount, sizeof(int *))) == NULL)
		exit(0);

	if ((inputTemp = (structInput *)calloc(1, sizeof(structInput))) == NULL)
		exit(0);



	for (i = 0; i < inputSource->nInputCount; ++i)
	{
		if (!inputSource->data[i].bTrained)
		{
			++inputTemp->nInputCount;
		}
	}


	inputTemp->nDataSource = inputSource->nDataSource;
	inputTemp->nChannels = inputSource->nChannels;
	inputTemp->nRowCount = inputSource->nRowCount;
	inputTemp->nColumnCount = inputSource->nColumnCount;

	inputTemp->nSize = inputTemp->nRowCount * inputTemp->nColumnCount;

	if ((inputTemp->data = (structInputData *)calloc(inputTemp->nInputCount, sizeof(structInputData))) == NULL)
		exit(0);

	if ((inputTemp->nClassMemberCount = (int *)calloc(inputSource->nClassCount, sizeof(int))) == NULL)
		return;

	for (i = 0, nImageIndex = 0; i<inputSource->nInputCount; ++i)
	{
		if (!inputSource->data[i].bTrained)
		{
			inputTemp->data[nImageIndex].fx32Intensity = (fx32 *)calloc(inputTemp->nSize, sizeof(fx32));
			inputTemp->data[nImageIndex].fIntensity = (float *)calloc(inputTemp->nSize, sizeof(float));

			inputTemp->data[nImageIndex].nID = inputSource->data[i].nID;
			inputTemp->data[nImageIndex].nLabelID = inputSource->data[i].nLabelID;
			strcpy(inputTemp->data[nImageIndex].sLabel, inputSource->data[i].sLabel);
			inputTemp->data[nImageIndex].nGroupA = inputSource->data[i].nGroupA;
			inputTemp->data[nImageIndex].nGroupB = inputSource->data[i].nGroupB;
			inputTemp->data[nImageIndex].fError = inputSource->data[i].fError;
			inputTemp->data[nImageIndex].fRank = inputSource->data[i].fRank;
			inputTemp->data[nImageIndex].bTrained = inputSource->data[i].bTrained;

			for (j = 0; j < inputTemp->nSize; ++j)
			{
				inputTemp->data[nImageIndex].fIntensity[j] = inputSource->data[i].fIntensity[j];
				inputTemp->data[nImageIndex].fx32Intensity[j] = inputSource->data[i].fx32Intensity[j];
			}

			++nImageIndex;
		}
		else
		{
			//printf("Trained\n");
		}
	}

	Sort_InputData(inputTemp->data, inputTemp->nInputCount, inputTemp->nSize, SORT);
	SiftClasses_InputData(inputTemp, &classHead, TRAINING);


	for (classCur = classHead; classCur != NULL; classCur = classCur->next)
	{
		if ((nMemberArray[classCur->nID] = (int *)calloc(inputTemp->nClassMemberCount[classCur->nID], sizeof(int))) == NULL)
			exit(0);

		j = 0;
		for (i = 0; i<nImageIndex; ++i)
		{
			if (inputTemp->data[i].nLabelID == classCur->nID)
			{
				nMemberArray[classCur->nID][j++] = i;
			}
		}
	}

	(*inputDestination)->nDataSource = inputSource->nDataSource;
	(*inputDestination)->nChannels = inputSource->nChannels;
	(*inputDestination)->nRowCount = inputSource->nRowCount;
	(*inputDestination)->nColumnCount = inputSource->nColumnCount;

	(*inputDestination)->nInputCount = (nClassMemberCount * inputSource->nClassCount);
	(*inputDestination)->nSize = (*inputDestination)->nRowCount * (*inputDestination)->nColumnCount;

	if (((*inputDestination)->data = (structInputData *)calloc((*inputDestination)->nInputCount, sizeof(structInputData))) == NULL)
		exit(0);

	nImageIndex = 0;
	for (classCur = classHead; classCur != NULL; classCur = classCur->next)
	{
		// First
		i = nMemberArray[classCur->nID][0];

		(*inputDestination)->data[nImageIndex].fx32Intensity = (fx32 *)calloc(inputTemp->nSize, sizeof(fx32));
		(*inputDestination)->data[nImageIndex].fIntensity = (float *)calloc(inputTemp->nSize, sizeof(float));

		(*inputDestination)->data[nImageIndex].nID = i;
		(*inputDestination)->data[nImageIndex].nLabelID = inputTemp->data[i].nLabelID;
		strcpy((*inputDestination)->data[nImageIndex].sLabel, inputTemp->data[i].sLabel);
		(*inputDestination)->data[nImageIndex].nGroupA = inputTemp->data[i].nGroupA;
		(*inputDestination)->data[nImageIndex].nGroupB = inputTemp->data[i].nGroupB;
		(*inputDestination)->data[nImageIndex].fError = inputTemp->data[i].fError;
		(*inputDestination)->data[nImageIndex].fRank = inputTemp->data[i].fRank;

		for (j = 0; j < (*inputDestination)->nSize; ++j)
		{
			(*inputDestination)->data[nImageIndex].fIntensity[j] = inputTemp->data[i].fIntensity[j];
			(*inputDestination)->data[nImageIndex].fx32Intensity[j] = inputTemp->data[i].fx32Intensity[j];
		}
		++nImageIndex;

		// Last
		i = nMemberArray[classCur->nID][inputTemp->nClassMemberCount[classCur->nID] - 1];

		(*inputDestination)->data[nImageIndex].fx32Intensity = (fx32 *)calloc(inputTemp->nSize, sizeof(fx32));
		(*inputDestination)->data[nImageIndex].fIntensity = (float *)calloc(inputTemp->nSize, sizeof(float));

		(*inputDestination)->data[nImageIndex].nID = i;
		(*inputDestination)->data[nImageIndex].nLabelID = inputTemp->data[i].nLabelID;
		strcpy((*inputDestination)->data[nImageIndex].sLabel, inputTemp->data[i].sLabel);
		(*inputDestination)->data[nImageIndex].nGroupA = inputTemp->data[i].nGroupA;
		(*inputDestination)->data[nImageIndex].nGroupB = inputTemp->data[i].nGroupB;
		(*inputDestination)->data[nImageIndex].fError = inputTemp->data[i].fError;
		(*inputDestination)->data[nImageIndex].fRank = inputTemp->data[i].fRank;

		for (j = 0; j < (*inputDestination)->nSize; ++j)
		{
			(*inputDestination)->data[nImageIndex].fIntensity[j] = inputTemp->data[i].fIntensity[j];
			(*inputDestination)->data[nImageIndex].fx32Intensity[j] = inputTemp->data[i].fx32Intensity[j];
		}
		++nImageIndex;

		if (nClassMemberCount > inputTemp->nClassMemberCount[classCur->nID])
			nClassMemberCount = inputTemp->nClassMemberCount[classCur->nID];

		nIncrement = (inputTemp->nClassMemberCount[classCur->nID] - 1) / (nClassMemberCount - 1);


		for (k = 1; k<nClassMemberCount - 1; ++k, ++nImageIndex)
		{
			i = nMemberArray[classCur->nID][nIncrement*k];

			(*inputDestination)->data[nImageIndex].fx32Intensity = (fx32 *)calloc(inputTemp->nSize, sizeof(fx32));
			(*inputDestination)->data[nImageIndex].fIntensity = (float *)calloc(inputTemp->nSize, sizeof(float));

			(*inputDestination)->data[nImageIndex].nID = i;
			(*inputDestination)->data[nImageIndex].nLabelID = inputTemp->data[i].nLabelID;
			strcpy((*inputDestination)->data[nImageIndex].sLabel, inputTemp->data[i].sLabel);
			(*inputDestination)->data[nImageIndex].nGroupA = inputTemp->data[i].nGroupA;
			(*inputDestination)->data[nImageIndex].nGroupB = inputTemp->data[i].nGroupB;
			(*inputDestination)->data[nImageIndex].fError = inputTemp->data[i].fError;
			(*inputDestination)->data[nImageIndex].fRank = inputTemp->data[i].fRank;

			for (j = 0; j < (*inputDestination)->nSize; ++j)
			{
				(*inputDestination)->data[nImageIndex].fIntensity[j] = inputTemp->data[i].fIntensity[j];
				(*inputDestination)->data[nImageIndex].fx32Intensity[j] = inputTemp->data[i].fx32Intensity[j];
			}
		}
	}

	(*inputDestination)->nInputCount = nImageIndex;
	Sort_InputData((*inputDestination)->data, (*inputDestination)->nInputCount, (*inputDestination)->nSize, RANDOMIZE);
	SiftClasses_InputData((*inputDestination), &classHead, TRAINING);

	for (i = 0; i < inputTemp->nInputCount; ++i)
	{
		free(inputTemp->data[i].fIntensity);
		free(inputTemp->data[i].fx32Intensity);
	}

	free(inputTemp->data);
	free(inputTemp->nClassMemberCount);
	free(inputTemp);

	//printf("--- %d ---\n", nImageIndex);
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void SiftClasses_InputData(structInput *input, structClass **classHead, int nMode)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structClass	*classCur;
	int			i;

	for (i = 0; i<input->nInputCount; ++i)
	{
		for (classCur = *classHead; classCur != NULL; classCur = classCur->next)
		{
			if (!strcmp(classCur->sLabel, input->data[i].sLabel))
				break;
		}

		if (classCur == NULL)
			input->data[i].nLabelID = AddNew_Classes(classHead, input->data[i].sLabel);
		else
			input->data[i].nLabelID = classCur->nID;

	}

	for (classCur = *classHead, input->nClassCount = 0; classCur != NULL; classCur = classCur->next, ++input->nClassCount);

	if (input->nClassMemberCount == NULL)
	{
		if ((input->nClassMemberCount = (int *)calloc(input->nClassCount, sizeof(int))) == NULL)
			exit(0);
	}
	else
	{
		for (i = 0; i < input->nClassCount; ++i)
			input->nClassMemberCount[i] = 0;
	}


	for (i = 0; i < input->nInputCount; ++i)
	{
		if (!input->data[i].bTrained)
			++input->nClassMemberCount[input->data[i].nLabelID];
	}


	for (classCur = *classHead; classCur != NULL; classCur = classCur->next)
	{
		if (nMode == TRAINING)
			classCur->nTrainingCount = input->nClassMemberCount[classCur->nID];
		else if (nMode == VERIFY)
			classCur->nVerifyCount = input->nClassMemberCount[classCur->nID];
		else if (nMode == TESTING)
			classCur->nTestingCount = input->nClassMemberCount[classCur->nID];
	}


	return;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void Sort_InputData(structInputData arr[], int n, int nSize, int nMode)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structInputData	*inputDataTemp;
	structInputData temp;
	int				*nIndexArray;
	int				i, j;

	if (nMode == RANDOMIZE)
	{
		inputDataTemp = (structInputData *)calloc(n, sizeof(structInputData));
		nIndexArray = (int *)calloc(n, sizeof(int));
		
		//RandomizeArray(nIndexArray, n, RANDOMIZE);
		ShuffleArray(nIndexArray, n, 0);


		for (i = 0; i < n; ++i)
		{
			strcpy(inputDataTemp[i].sDescription, arr[nIndexArray[i]].sDescription);
			strcpy(inputDataTemp[i].sLabel, arr[nIndexArray[i]].sLabel);

			inputDataTemp[i].nID = arr[nIndexArray[i]].nID;
			inputDataTemp[i].nLabelID = arr[nIndexArray[i]].nLabelID;
			inputDataTemp[i].nSequence = arr[nIndexArray[i]].nSequence;
			inputDataTemp[i].nDataSource = arr[nIndexArray[i]].nDataSource;
			inputDataTemp[i].nGroupA = arr[nIndexArray[i]].nGroupA;
			inputDataTemp[i].nGroupB = arr[nIndexArray[i]].nGroupB;
			inputDataTemp[i].bTrained = arr[nIndexArray[i]].bTrained;
			inputDataTemp[i].fError = arr[nIndexArray[i]].fError;
			inputDataTemp[i].fRank = arr[nIndexArray[i]].fRank;
			inputDataTemp[i].fAspect = arr[nIndexArray[i]].fAspect;
			inputDataTemp[i].fRadialSpeed = arr[nIndexArray[i]].fRadialSpeed;
			inputDataTemp[i].fSpread = arr[nIndexArray[i]].fSpread;

			inputDataTemp[i].fIntensity = (float *)calloc(nSize, sizeof(float));
			
			for (j = 0; j<nSize; ++j)
			{
				inputDataTemp[i].fIntensity[j] = arr[nIndexArray[i]].fIntensity[j];
			}
			

			//*fx32Intensity;	  *fx32Intensity;
			//*nIntensity;	  *nIntensity;

			inputDataTemp[i].nHRRCount = arr[nIndexArray[i]].nHRRCount;
			inputDataTemp[i].fHRR = (float *)calloc(inputDataTemp[i].nHRRCount, sizeof(float));

			for (j = 0; j<inputDataTemp[i].nHRRCount; ++j)
			{
				inputDataTemp[i].fHRR[j] = arr[nIndexArray[i]].fHRR[j];
			}

		}

		for (i = 0; i<n; ++i)
			arr[i] = inputDataTemp[i];

		free(nIndexArray);
		free(inputDataTemp);
	}
	else if (nMode == SHUFFLE)
	{
		for (i = 0; i < n - 1; i++)
		{
			// Last i elements are already in place   
			for (j = 0; j < n - i - 1; j++)
			{
				if (arr[j].fError > arr[j + 1].fError)
				{
					temp = arr[j];
					arr[j] = arr[j + 1];
					arr[j + 1] = temp;
				}
			}
		}

		inputDataTemp = (structInputData *)calloc(n, sizeof(structInputData));

		for (i = 0, j = 0; j < n; ++i, j += 2)
			inputDataTemp[j] = arr[i];

		for (i = n - 1, j = 1; j < n; --i, j += 2)
			inputDataTemp[j] = arr[i];

		for (i = 0; i<n; ++i)
			arr[i] = inputDataTemp[i];

		free(inputDataTemp);

	}
	else
	{
		for (i = 0; i < n - 1; i++)
		{
			for (j = 0; j < n - i - 1; j++)
			{
				if (arr[j].fError > arr[j + 1].fError)
				{
					temp = arr[j];
					arr[j] = arr[j + 1];
					arr[j + 1] = temp;
				}
			}
		}
	}
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void SetStatistics_InputData(structInput *inputTrainingData, structInput *inputVerifyData, structInput *inputTestingData, structClass *classHead, int *nClassCount, int **nClassMemberCount)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structClass	*classCur;
	int			i;

	for (classCur = classHead, *nClassCount = 0; classCur != NULL; classCur = classCur->next, ++(*nClassCount));

	if ((*nClassMemberCount = (int *)calloc(*nClassCount, sizeof(int))) == NULL)
		exit(0);

	if (inputTrainingData->nClassMemberCount)
		free(inputTrainingData->nClassMemberCount);

	inputTrainingData->nClassCount = *nClassCount;
	if ((inputTrainingData->nClassMemberCount = (int *)calloc(inputTrainingData->nClassCount, sizeof(int))) == NULL)
		exit(0);

	if ((inputTrainingData->nAverageIDArray = (int *)calloc(inputTrainingData->nClassCount, sizeof(int))) == NULL)
		exit(0);

	if ((inputTrainingData->nMaxIDArray = (int *)calloc(inputTrainingData->nClassCount, sizeof(int))) == NULL)
		exit(0);

	///////////////////////////////////////////////////////////////////////
	if (inputVerifyData->nClassMemberCount)
		free(inputVerifyData->nClassMemberCount);

	inputVerifyData->nClassCount = *nClassCount;
	if ((inputVerifyData->nClassMemberCount = (int *)calloc(inputVerifyData->nClassCount, sizeof(int))) == NULL)
		exit(0);

	////////////////////////////////////////////////////////////////////////
	if (inputTestingData->nClassMemberCount)
		free(inputTestingData->nClassMemberCount);

	inputTestingData->nClassCount = *nClassCount;
	if ((inputTestingData->nClassMemberCount = (int *)calloc(inputTestingData->nClassCount, sizeof(int))) == NULL)
		exit(0);

	for (i = 0; i < inputTrainingData->nInputCount; ++i)
	{
		++inputTrainingData->nClassMemberCount[inputTrainingData->data[i].nLabelID];
		++(*nClassMemberCount)[inputTrainingData->data[i].nLabelID];
	}

	for (i = 0; i < inputVerifyData->nInputCount; ++i)
	{
		++inputVerifyData->nClassMemberCount[inputVerifyData->data[i].nLabelID];
		++(*nClassMemberCount)[inputVerifyData->data[i].nLabelID];
	}

	for (i = 0; i < inputTestingData->nInputCount; ++i)
	{
		++inputTestingData->nClassMemberCount[inputTestingData->data[i].nLabelID];
		++(*nClassMemberCount)[inputTestingData->data[i].nLabelID];
	}

}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void SortByMissCount(structInputData *data, int nCount)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	qsort(data, nCount, sizeof(structInputData), CompareMissCounts);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void SortByDifference(structInputData* data, int nCount)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	qsort(data, nCount, sizeof(structInputData), CompareDifference);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void SortFloatAscend(float* data, int nCount)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	qsort(data, nCount, sizeof(float), CompareFloatAscend);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void GroupByDifference(structInputData* data, int nInputCount, int nClusterCount)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structData	*dataJenkFish = NULL;
	structData	*dataCluster = NULL;
	int			nImageIndex;
	int			nCluster;

	
	dataJenkFish = (structData*)calloc(nInputCount, sizeof(structData));

	for (nImageIndex = 0; nImageIndex < nInputCount; ++nImageIndex)
	{
		dataJenkFish[nImageIndex].nID = nImageIndex;
		dataJenkFish[nImageIndex].fOutput = data[nImageIndex].fDifference;
	}

	JenkFish(dataJenkFish, &dataCluster, nClusterCount, nInputCount);
	free(dataJenkFish);

	for (nImageIndex = 0; nImageIndex < nInputCount; ++nImageIndex)
	{
		for (nCluster = 0; nCluster< nClusterCount; ++nCluster)
		{
			if (data[nImageIndex].fDifference >= dataCluster[nCluster].fStart && data[nImageIndex].fDifference <= dataCluster[nCluster].fEnd)
			{
				data[nImageIndex].nGroupA = nCluster;
				break;
			}
		}
	}

	free(dataCluster);
	return;
}
