#include "main.h"
#include <sys/stat.h>
#include <unistd.h>
#ifdef _WINDOWS
	#include <direct.h>
#endif

structLayer *CalculateOutputSize(structCLN* cln)
{
	structLayer* layerCur = NULL;
	structLayer* layerPrev = NULL;

	for (layerCur = cln->layerHead; layerCur != NULL; layerCur = layerCur->next)
	{
		CalculateLayerOutputSize(layerCur);
		layerPrev = layerCur;
	}

	return(layerPrev);
}

void CalculateLayerOutputSize(structLayer* layer)
{
	int nPH1 = 0;
	int nPH2 = 0;
	int nPW1 = 0;
	int nPW2 = 0;

	if (layer->nLayerType == CONV_2D_LAYER || layer->nLayerType == CONV_3D_LAYER || layer->nLayerType == MAX_POOLING_LAYER || layer->nLayerType == AVERAGE_POOLING_LAYER)
	{
		if (layer->prev == NULL)
		{
			layer->nInputMapCount = 1;
			layer->nOutputMapCount = layer->nKernelCount;
		}
		else
		{

			if (layer->nLayerType == MAX_POOLING_LAYER || layer->nLayerType == AVERAGE_POOLING_LAYER)
			{
				layer->nInputMapCount = layer->prev->nInputMapCount;
				layer->nKernelCount = layer->prev->nOutputMapCount;
				layer->nOutputMapCount = layer->nKernelCount;
			}
			else
			{
				layer->nInputMapCount = layer->prev->nOutputMapCount;
				layer->nOutputMapCount = layer->nKernelCount;
			}
		}

		layer->nOutputRowCount = (layer->nInputRowCount + (nPH1 + nPH2) - layer->nKernelRowCount) / layer->nStrideRow + 1;
		layer->nOutputColumnCount = (layer->nInputColumnCount + (nPW1 + nPW2) - layer->nKernelColumnCount) / layer->nStrideColumn + 1;
		layer->nPerceptronCount = layer->nOutputRowCount * layer->nOutputColumnCount * layer->nOutputMapCount;
		
		if (layer->nLayerType == MAX_POOLING_LAYER || layer->nLayerType == AVERAGE_POOLING_LAYER)
		{
			layer->nConnectionCount = layer->nKernelRowCount * layer->nKernelColumnCount * layer->nPerceptronCount;
			layer->nOutputArraySize = layer->nPerceptronCount;
		}
		else
		{
			layer->nConnectionCount = layer->nKernelRowCount * layer->nKernelColumnCount * layer->nPerceptronCount * layer->nInputMapCount;
			layer->nOutputArraySize = layer->nPerceptronCount * layer->nInputMapCount;
		}


		layer->nWeightCount = layer->nKernelCount * layer->nInputMapCount * (layer->nKernelRowCount * layer->nKernelColumnCount + 1);
	}
	else
	{
		layer->nOutputMapCount = 1;
		layer->nOutputRowCount = 1;
		layer->nOutputColumnCount = layer->nPerceptronCount;

		if (layer->prev == NULL)
			layer->nInputMapCount = 1;
		else
			layer->nInputMapCount = layer->prev->nOutputMapCount;

	
		layer->nOutputArraySize = layer->nPerceptronCount;
	}

	if (layer->next != NULL)
	{
		layer->next->nInputRowCount = layer->nOutputRowCount;
		layer->next->nInputColumnCount = layer->nOutputColumnCount;
	}

}







void MakeDirectory(char *sPath)
{
	char *sSeperate = strrchr(sPath, '\\');

	if (sSeperate != NULL)
	{
		*sSeperate = 0;

		MakeDirectory(sPath);

		*sSeperate = '\\';
	}
}

#ifdef _WINDOWS
	if (_mkdir(sPath) && errno != EEXIST && strlen(sPath) != 2)
		printf("error while trying to create %s\n", sPath);
#endif



FILE *FOpenMakeDirectory(char *sPath, const char *sMode)
{
	char *sSeperate = strrchr(sPath, '\\');

	if (sSeperate)
	{
		char *sNewPath = STRIDUP(sPath);

		sNewPath[sSeperate - sPath] = 0;
		
		MakeDirectory(sNewPath);

		free(sNewPath);
	}

	return fopen(sPath, sMode);
}




/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int DisplayResults(FILE *fpFile, char **sArray, int *nArray, int nCount)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	int		nLength;
	int		nMaxLength = 0;
	int		i, j;

	if (!nArray[0])
	{
		nArray[0] = (int)strlen(sArray[0]) + 3;

		if (fpFile != NULL)
			fprintf(fpFile, "%s", sArray[0]);

		for (i = 1; i < nCount; ++i)
		{
			nArray[i] = (int)strlen(sArray[i]) + 3;

			if (fpFile != NULL)
				fprintf(fpFile, "\t%s", sArray[i]);
		}

		if (fpFile != NULL)
			fprintf(fpFile, "\n");
	}



	if (fpFile != NULL)
		fprintf(fpFile, "%s", sArray[0]);

	nLength = (int)strlen(sArray[0]);
	printf("%s", sArray[0]);

	for (j = 0; j<(nArray[0] - nLength); ++j)
		printf(" ");

	for (i = 1; i < nCount; ++i)
	{
		if (fpFile != NULL)
			fprintf(fpFile, "\t%s", sArray[i]);

		nLength = (int)strlen(sArray[i]);
		printf("%s", sArray[i]);

		for (j = 0; j<(nArray[i] - nLength); ++j)
			printf(" ");
	}

	printf("\n");

	if (fpFile != NULL)
		fprintf(fpFile, "\n");

	return(nMaxLength);
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void DisplayHelpFile()
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	FILE		*pFile;
	char		sBuffer[1024];
	char		sFilePath[256];

	sprintf(sFilePath, "D:\\Data\\help.txt");
	
	if ((pFile = FOpenMakeDirectory(sFilePath, "rb")) == NULL)
	{
		HoldDisplay("Could not find the help file  (help.txt)");
	}

	while (!feof(pFile))
	{
		fgets(sBuffer, 1024, pFile);
		printf("%s", sBuffer);
	}

	fclose(pFile);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void HoldDisplay(const char *sMessage)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	printf("\n %s\n", sMessage);
	printf("\n press enter to quit\n");
	
	while (1);

	exit(0);
}



int CalculateWindowSize(int nWindowRow, int nKernel, int nStride)
{
	float	fWindowSize;
	int		nWindowSizeSC = 0;

	if (nStride > 0)
		fWindowSize = ((float)(nWindowRow - (nKernel - nStride))) / (float)nStride;
	else
		fWindowSize = (float)nWindowRow / (float)nKernel;

	nWindowSizeSC = (int)fWindowSize;

	if (fWindowSize > (float)nWindowSizeSC)
		++nWindowSizeSC;


	return(nWindowSizeSC);
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void CreateMatrix(int ***nMatrix, int nClassCount)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	int		i;

	(*nMatrix) = (int **)calloc(nClassCount, sizeof(int *));
	for (i = 0; i<nClassCount; ++i)
		(*nMatrix)[i] = (int *)calloc(nClassCount, sizeof(int));
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
float ScoreMatrixV2(structNetwork *networkMain, int nDisplayMode, int nMode)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structClass	*classCurRow;
	structClass	*classCurColumn;
	float		**fMatrix;
	float		fTotal = 0.0f;
	float		fAveragePercent;
	int			nCount = 0;
	int			nSum;
	int			nClassCount;
	int			i, j;

	if (nMode == ALL_CLN)
		nClassCount = networkMain->nClassCount;
	else
		nClassCount = 2;

	fMatrix = (float **)calloc(nClassCount, sizeof(float *));
	for (i = 0; i<nClassCount; ++i)
		fMatrix[i] = (float *)calloc(nClassCount, sizeof(float));


	for (i = 0; i<nClassCount; ++i)
	{
		for (j = 0, nSum=0; j<nClassCount; ++j)
			nSum += networkMain->nMatrix[i][j];

		for (j = 0; j<nClassCount; ++j)
			fMatrix[i][j]= (float)networkMain->nMatrix[i][j] / (float)nSum;
	}
	
	for (i = 0; i<nClassCount; ++i)
		fTotal += fMatrix[i][i];

	fAveragePercent = fTotal / (float)nClassCount;

	
	if (nDisplayMode == SHOW_MATRIX)
	{
		if (nMode == ALL_CLN)
		{
			for (classCurRow = networkMain->classHead; classCurRow != NULL; classCurRow = classCurRow->next)
				printf(" \t%s", classCurRow->sLabel);

			printf("\n");

			for (classCurRow = networkMain->classHead; classCurRow != NULL; classCurRow = classCurRow->next)
			{
				printf("%s\t", classCurRow->sLabel);
				for (classCurColumn = networkMain->classHead, nSum = 0; classCurColumn != NULL; classCurColumn = classCurColumn->next)
					printf("%0.4f\t", fMatrix[classCurRow->nID][classCurColumn->nID]);

				printf("\n");
			}
		}
		else
		{
			for (classCurRow = networkMain->classHead; classCurRow != NULL; classCurRow = classCurRow->next)
			{
				if (classCurRow->nID == networkMain->clnCur->nLabelID)
				{
					printf("\n \t%s \tNOT\n", classCurRow->sLabel);
					printf("%s\t%0.4f\t%0.4f\n", classCurRow->sLabel, fMatrix[0][0], fMatrix[0][1]);
					printf("NOT\t%0.4f\t%0.4f\n\n", fMatrix[1][0], fMatrix[1][1]);
				}
			}
		}
		
	}


	for (i = 0; i<nClassCount; ++i)
		free(fMatrix[i]);

	free(fMatrix);

	return(fAveragePercent);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
float ScoreClassLevelNetworkMatrix(int nNetworkType, int nLabelID, structClass *classHead, int **nMatrix, int nDisplayMode, FILE *fpFileOut)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structClass	*classCurRow;
	structClass	*classCurColumn;
	float		**fMatrix;
	float		fTotal = 0.0f;
	float		fAveragePercent;
	int			nCount = 0;
	int			nClassCount = 0;
	int			nMaxLength;
	int			nSum;
	int			i, j;



	if (nNetworkType == COMPLETE_NETWORK)
	{
		for (classCurRow = classHead; classCurRow != NULL; classCurRow = classCurRow->next)
			++nClassCount;
	}
	else if(nNetworkType == CLASS_NETWORK)
	{
		nClassCount = 2;
	}
	else
		nClassCount = 0;


	fMatrix = (float **)calloc(nClassCount, sizeof(float *));
	for (i = 0; i<nClassCount; ++i)
		fMatrix[i] = (float *)calloc(nClassCount, sizeof(float));


	for (i = 0; i<nClassCount; ++i)
	{
		for (j = 0, nSum=0; j<nClassCount; ++j)
			nSum += nMatrix[j][i];

		if (nSum > 0)
		{
			for (j = 0; j<nClassCount; ++j)
				fMatrix[j][i] = (float)nMatrix[j][i] / (float)nSum;
		}
	}
	
	for (i = 0; i<nClassCount; ++i)
		fTotal += fMatrix[i][i];

	fAveragePercent = fTotal / (float)nClassCount;

	
	if (nDisplayMode == SHOW_MATRIX)
	{
		nMaxLength = (int)strlen(classHead->sLabel);
		
		for (classCurRow = classHead->next; classCurRow != NULL; classCurRow = classCurRow->next)
		{
			if(strlen(classCurRow->sLabel) > nMaxLength)
				nMaxLength = (int)strlen(classCurRow->sLabel);
		}
		
		if (nMaxLength < 7)
			nMaxLength = 7;

			
		if (nNetworkType == COMPLETE_NETWORK)
		{
			for (i = 0; i < nMaxLength + 2; ++i)
			{
				printf(" ");
				if(fpFileOut != NULL)
					fprintf(fpFileOut, " ");
			}

			for (classCurRow = classHead; classCurRow != NULL; classCurRow = classCurRow->next)
			{
				printf("%s", classCurRow->sLabel);
				if (fpFileOut != NULL)
					fprintf(fpFileOut, "%s", classCurRow->sLabel);

				for (i = (int)strlen(classCurRow->sLabel); i < nMaxLength + 2; ++i)
				{
					printf(" ");
					if (fpFileOut != NULL)
						fprintf(fpFileOut, " ");
				}
			}
				

			printf("\n");
			if (fpFileOut != NULL)
				fprintf(fpFileOut, "\n");

			for (classCurRow = classHead; classCurRow != NULL; classCurRow = classCurRow->next)
			{
				for (i = 0; i < (nMaxLength - strlen(classCurRow->sLabel)); ++i)
				{
					printf(" ");
					if (fpFileOut != NULL)
						fprintf(fpFileOut, " ");
				}
				printf("%s  ", classCurRow->sLabel);
				if (fpFileOut != NULL)
					fprintf(fpFileOut, "%s  ", classCurRow->sLabel);


				for (classCurColumn = classHead; classCurColumn != NULL; classCurColumn = classCurColumn->next)
				{
					printf("%0.4f", fMatrix[classCurRow->nID][classCurColumn->nID]);
					//printf("%d", nMatrix[classCurRow->nID][classCurColumn->nID]);
					
					if (fpFileOut != NULL)
						fprintf(fpFileOut, "%0.4f", fMatrix[classCurRow->nID][classCurColumn->nID]);

					for (i = 6; i < nMaxLength + 2; ++i)
					{
						printf(" ");
						if (fpFileOut != NULL)
							fprintf(fpFileOut, " ");
					}
				}
					

				printf("\n");
				if (fpFileOut != NULL)
					fprintf(fpFileOut, "\n");
			}
			printf("\n");
			if (fpFileOut != NULL)
				fprintf(fpFileOut, "\n");
		}
		else if (nNetworkType == CLASS_NETWORK)
		{
			for (classCurRow = classHead; classCurRow != NULL; classCurRow = classCurRow->next)
			{
				if (classCurRow->nID == nLabelID)
				{
					printf("\n \t%s \tNOT\n", classCurRow->sLabel);
					printf("%s\t%0.4f\t%0.4f\n", classCurRow->sLabel, fMatrix[0][0], fMatrix[0][1]);
					printf("NOT\t%0.4f\t%0.4f\n\n", fMatrix[1][0], fMatrix[1][1]);

					if (fpFileOut != NULL)
					{
						fprintf(fpFileOut, "\n \t%s \tNOT\n", classCurRow->sLabel);
						fprintf(fpFileOut, "%s\t%0.4f\t%0.4f\n", classCurRow->sLabel, fMatrix[0][0], fMatrix[0][1]);
						fprintf(fpFileOut, "NOT\t%0.4f\t%0.4f\n\n", fMatrix[1][0], fMatrix[1][1]);
					}
				}
			}
		}
		else
		{
			printf("\n");
			if (fpFileOut != NULL)
				fprintf(fpFileOut, "\n");
		}
	}


	for (i = 0; i<nClassCount; ++i)
		free(fMatrix[i]);

	free(fMatrix);

	return(fAveragePercent);
}



/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void ClearClassLevelNetworkMatrix(structClass *classHead, int **nMatrix)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structClass	*classRow;
	structClass	*classColumn;


	for (classRow= classHead; classRow!=NULL; classRow = classRow->next)
	{
		for (classColumn = classHead; classColumn != NULL; classColumn = classColumn->next)
		{
			nMatrix[classRow->nID][classColumn->nID] = 0;
		}
	}
}



/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void ClearMatrix(int **nMatrix, int nClassCount)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	int			i, j;

	for (i = 0; i<nClassCount; ++i)
	{
		for (j = 0; j<nClassCount; ++j)
		{
			nMatrix[i][j] = 0;
		}
	}
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void DeleteMatrix(int **nMatrix, int nClassCount)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	int			i;

	for (i = 0; i < nClassCount; ++i)
		free(nMatrix[i]);

	free(nMatrix);

	nMatrix = NULL;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void RandomizeArray(int	*nIndexArray, int nCount, int nMode)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	int	nValue;
	int	bFlag;
	int	i, j;

	for (i = 0; i<nCount; ++i)
	{
		if (nMode == RANDOMIZE)
		{
			bFlag = 0;
			nValue = (rand() * rand()) % (nCount);
			if (nValue >= 0)
			{
				for (j = 0, bFlag=1; j < i && bFlag; ++j)
					bFlag = (nIndexArray[j] != nValue);
			}

			if (bFlag)
				nIndexArray[i] = nValue;
			else --i;
		}
		else
			nIndexArray[i] = i;
	}
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void Swap(structSort *xp, structSort *yp)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structSort temp = *xp;

	*xp = *yp;
	*yp = temp;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void BubbleSort(structSort arr[], int n)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	int i, j;
	for (i = 0; i < n - 1; i++)

		// Last i elements are already in place   
		for (j = 0; j < n - i - 1; j++)
			if (arr[j].fValue < arr[j + 1].fValue)
				Swap(&arr[j], &arr[j + 1]);
}





/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int ClusterPerceptronWeights(structPerceptron *perceptronCur)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structSynapse		*synapseHead= perceptronCur->synapseHead;
	structSynapse		*synapseCur;
	structData			*dataJenkFish = NULL;
	structData			*dataCluster = NULL;
	int					nWeightCount=0;
	int					p;

	if (!perceptronCur->nClusterCount)
	{
		for (synapseCur = synapseHead; synapseCur != NULL; synapseCur = synapseCur->next)
			*synapseCur->fWeight = 0.0f;
	}
	else
	{
		for (synapseCur = synapseHead->next, perceptronCur->nSynapseCount = 0; synapseCur != NULL; synapseCur = synapseCur->next, ++perceptronCur->nSynapseCount)
			synapseCur->nCluster = -1;

		dataJenkFish = (structData*)calloc(perceptronCur->nSynapseCount, sizeof(structData));

		for (synapseCur = synapseHead->next, nWeightCount = 0; synapseCur != NULL; synapseCur = synapseCur->next, ++nWeightCount)
		{
			dataJenkFish[nWeightCount].nID = nWeightCount;
			dataJenkFish[nWeightCount].fOutput = *synapseCur->fWeight;
		}

		JenkFish(dataJenkFish, &dataCluster, perceptronCur->nClusterCount, nWeightCount);

		nWeightCount = 0;
		for (p = 0; p < perceptronCur->nClusterCount; ++p)
		{
			for (synapseCur = synapseHead->next; synapseCur != NULL; synapseCur = synapseCur->next)
			{
				if (synapseCur->nCluster == -1 && *synapseCur->fWeight >= dataCluster[p].fStart && *synapseCur->fWeight <= dataCluster[p].fEnd)
				{
					*synapseCur->fWeight = dataCluster[p].fAverage;

					synapseCur->nCluster = p;
					++nWeightCount;
				}
			}
		}

		free(dataJenkFish);
		free(dataCluster);
	}

	return(nWeightCount);
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void DescribeClassLevelNetwork(structCLN *cln, FILE *fpFileOut)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	FILE			*fpOut = NULL;
	structLayer		*layerCur;
	int				nTotalPerceptrons = 0;
	int				nTotalSynapes = 0;
	int				nTotalAdditions = 0;
	int				nTotalMultiplications = 0;
	int				nTotalWeights = 0;
	int				bShowMath = (cln->layerHead->nAdditions > 0 || cln->layerHead->nMultiplications > 0);


	if (fpFileOut == NULL)
		fpOut = stdout;
	else
		fpOut = fpFileOut;

	if(bShowMath)
		fprintf(fpOut, "\nLayer     \tFilters\tKernel\tStride\tPerceptrons\tMultiplications\tAdditions\tWeights\tOutputs\n");
	else
		fprintf(fpOut, "\nLayer     \tFilters\tKernel\tStride\tPerceptrons\tConnections\tAdditions\tWeights\tOutputs\n");


	for (layerCur = cln->layerHead; layerCur != NULL; layerCur = layerCur->next)
	{
		if (layerCur->nLayerType == FULLY_CONNECTED_LAYER)
			fprintf(fpOut, "Full    \t");
		else if (layerCur->nLayerType == SINGLE_CONV_LAYER)
			fprintf(fpOut, "2D_Conv \t");
		else if (layerCur->nLayerType == MULTIPLE_CONV_LAYER)
			fprintf(fpOut, "3D_Conv \t");
		else if (layerCur->nLayerType == CLASSIFIER_LAYER)
			fprintf(fpOut, "Classifier\t");
		else if (layerCur->nLayerType == SPARSELY_CONNECTED_LAYER)
			fprintf(fpOut, "Sparse  \t");
		else if (layerCur->nLayerType == COMBINING_CLASSIFIER_LAYER)
			fprintf(fpOut, "Combining  \t");
		else if (layerCur->nLayerType == MAX_POOLING_LAYER)
			fprintf(fpOut, "Max_Pool  \t");

		if (bShowMath)
		{
			if (layerCur->nLayerType == SINGLE_CONV_LAYER || layerCur->nLayerType == MULTIPLE_CONV_LAYER)
				fprintf(fpOut, "%7d\t%6d\t%6d\t%11d\t%11d\t%9d\t%7d\t%7d", layerCur->nKernelCount, layerCur->nKernelRowCount, layerCur->nStrideRow, layerCur->nPerceptronCount, layerCur->nMultiplications, layerCur->nAdditions, layerCur->nWeightCount, layerCur->nOutputArraySize);
			else
				fprintf(fpOut, "\t\t\t%11d\t%11d\t%9d\t%7d\t%7d", layerCur->nPerceptronCount, layerCur->nMultiplications, layerCur->nAdditions, layerCur->nWeightCount, layerCur->nOutputArraySize);
		
			nTotalMultiplications += layerCur->nMultiplications;
			nTotalAdditions += layerCur->nAdditions;
		}
		else
		{
			if (layerCur->nLayerType == SINGLE_CONV_LAYER || layerCur->nLayerType == MULTIPLE_CONV_LAYER)
				fprintf(fpOut, "%7d\t%6d\t%6d\t%11d\t%11d\t%9d\t%7d\t%7d", layerCur->nKernelCount, layerCur->nKernelRowCount, layerCur->nStrideRow, layerCur->nPerceptronCount, layerCur->nConnectionCount, layerCur->nAdditionCount, layerCur->nWeightCount, layerCur->nOutputArraySize);
			else if(layerCur->nLayerType == MAX_POOLING_LAYER)
				fprintf(fpOut, "%7d\t%6d\t%6d\t%11d\t%11d\t%9d\t%7d\t%7d", layerCur->nKernelCount, layerCur->nKernelRowCount, layerCur->nStrideRow, layerCur->nPerceptronCount, layerCur->nConnectionCount, layerCur->nAdditionCount, layerCur->nWeightCount, layerCur->nOutputArraySize);
			else
				fprintf(fpOut, "\t\t\t%11d\t%11d\t%9d\t%7d\t%7d", layerCur->nPerceptronCount, layerCur->nConnectionCount, layerCur->nAdditionCount, layerCur->nWeightCount, layerCur->nOutputArraySize);
		
		
			nTotalAdditions += layerCur->nAdditionCount;
			nTotalSynapes += layerCur->nConnectionCount;
		}

		nTotalPerceptrons += layerCur->nPerceptronCount;
		nTotalWeights += layerCur->nWeightCount;

		fprintf(fpOut, "\n");
	}

	if (bShowMath)
		fprintf(fpOut, "\t\t\t\t\t%11d\t%11d\t%9d\t%7d", nTotalPerceptrons, nTotalMultiplications, nTotalAdditions, nTotalWeights);
	else
		fprintf(fpOut, "\t\t\t\t\t%11d\t%11d\t%9d\t%7d", nTotalPerceptrons, nTotalSynapes, nTotalAdditions, nTotalWeights);

	fprintf(fpOut, "\n\n");
}




/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void GetClassLevelNetworkWeightCount(structCLN *cln)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structLayer			*layerCur;
	structPerceptron	*perceptronHeadCur = NULL;
	structPerceptron	*perceptronHead = NULL;
	structPerceptron	*perceptronCur;
	structSynapse		*synapseCur;

	cln->nWeightCount = 0;
	cln->nLayerCount = 0;

	for (layerCur = cln->layerHead; layerCur != NULL; layerCur = layerCur->next, ++cln->nLayerCount)
	{
		layerCur->nPerceptronCount = 0;
		layerCur->nWeightCount = 0;
		layerCur->nConnectionCount = 0;

		for (perceptronHeadCur = layerCur->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
		{
			for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
			{
				++layerCur->nPerceptronCount;
				perceptronCur->nWeightCount = 0;

				if (perceptronCur->synapseHead != NULL)
				{
					for (synapseCur = perceptronCur->synapseHead->next; synapseCur != NULL; synapseCur = synapseCur->next)
					{
						if (layerCur->nLayerType == DENSE_LAYER || layerCur->nLayerType == CLASSIFIER_LAYER || ((layerCur->nLayerType == CONV_2D_LAYER || layerCur->nLayerType == CONV_3D_LAYER) && perceptronCur == perceptronHeadCur))
						{
							++layerCur->nWeightCount;
							++perceptronCur->nWeightCount;
						}

						++layerCur->nConnectionCount;
					}
				}
			}
		}

		cln->nWeightCount += layerCur->nWeightCount;
	}

	return;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int GetClassLevelNetworkWeightZeroCount(structCLN *cln)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structLayer			*layerCur;
	structPerceptron	*perceptronHeadCur = NULL;
	structPerceptron	*perceptronHead = NULL;
	structPerceptron	*perceptronCur;
	structSynapse		*synapseCur;
	int					nWeightCount;
	int					nErrorCount;

	nWeightCount = 0;
	nErrorCount = 0;

	for (layerCur = cln->layerHead; layerCur != NULL; layerCur = layerCur->next)
	{
		layerCur->nPerceptronCount = 0;

		if (layerCur->perceptronHead != NULL)
		{
			layerCur->nConnectionCount = 0;
			layerCur->nWeightCount = 0;

			for (perceptronHeadCur = layerCur->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
			{
				for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
				{
					++layerCur->nPerceptronCount;

					for (synapseCur = perceptronCur->synapseHead; synapseCur != NULL; synapseCur = synapseCur->next)
					{
						if (layerCur->nLayerType == FULLY_CONNECTED_LAYER || layerCur->nLayerType == CLASSIFIER_LAYER || layerCur->nLayerType == COMBINING_CLASSIFIER_LAYER)
						{
							if(*synapseCur->fWeight != 0.0f)
								++nWeightCount;
							++cln->nWeightCount;
							++layerCur->nWeightCount;

							if (synapseCur != perceptronCur->synapseHead)
								++layerCur->nConnectionCount;
						}
						else
						{
							if (perceptronCur == perceptronHeadCur)
							{
								++layerCur->nWeightCount;
								++cln->nWeightCount;
								
								if (*synapseCur->fWeight != 0.0f)
									++nWeightCount;
							}

							++layerCur->nConnectionCount;
						}
					}
				}
			}
		}
		else
		{
			cln->nWeightCount += layerCur->nWeightCount;
		}
	}
	
	
	return(nWeightCount);
}




/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void CopyClassLevelNetworkWeights(structCLN *cln, float *fWeights, float *fThresholds, int nMode)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structLayer			*layerCur;
	int					nWeightCount;
	int					i=0;

	for (layerCur = cln->layerHead, nWeightCount=0; layerCur != NULL; layerCur = layerCur->next)
	{
		for (i = 0; i < layerCur->nWeightCount; ++i)
		{
			if (nMode == NETWORK_TO_MEMORY)
				fWeights[nWeightCount++] = layerCur->fWeightArray[i];
			else
				layerCur->fWeightArray[i] = fWeights[nWeightCount++];
		}
	}


	return;
}




/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
float Slope(float fY2, float fY1, float fX2, float fX1)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	float	fSlope;

	fSlope = (fY2 - fY1) / (fX2 - fX1);

	return(fSlope);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
float Intercept(float fY2, float fY1, float fX2, float fX1)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	float	fSlope;
	float	fIntercept;

	fSlope = (fY2 - fY1) / (fX2 - fX1);

	fIntercept = fY2 - (fSlope * fX2);

	return(fIntercept);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
float CalculateWeight(float fY2, float fY1, float fX2, float fX1, float fTarget)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	float	fSlope;
	float	fIntercept;
	float	fWeight;

	fSlope = Slope(fY2, fY1, fX2, fX1);
	fIntercept = Intercept(fY2, fY1, fX2, fX1);
	fWeight = (fSlope *fTarget) + fIntercept;

	return(fWeight);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void DisplayMessage(const char *sMessage, int nMode)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	printf("\n***** %s *****\n", sMessage);

	if (nMode == PAUSE)
		while (1);
}
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void CalculateStandardDeviationArray(float *fArray, float *fSD, float *fAverage, int nCount)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	float	fSum = 0.0f;
	int		i;

	for (i = 0; i < nCount; ++i)
	{
		fSum += fArray[i];
	}

	*fAverage = fSum / (float)nCount;
	fSum = 0.0f;

	for (i = 0; i < nCount; ++i)
	{
		fSum += ((fArray[i] - *fAverage) * (fArray[i] - *fAverage));
	}

	*fSD = sqrtf(fSum / (float)(nCount - 1));

}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void CalculateStandardDeviationMACArray(float **fArray, float *fSD, float *fAverage, int nCount)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	float	fSum = 0.0f;
	int		i;

	for (i = 0; i < nCount; ++i)
	{
		fSum += (*fArray)[i];
	}

	*fAverage = fSum / (float)nCount;
	fSum = 0.0f;

	for (i = 0; i < nCount; ++i)
	{
		fSum += (((*fArray)[i] - *fAverage) * ((*fArray)[i] - *fAverage));
	}

	*fSD = sqrtf(fSum / (float)(nCount - 1));

}



#ifdef _WINDOWS
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void StartTimer(LARGE_INTEGER *lnFrequency, LARGE_INTEGER *lnStart)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	QueryPerformanceFrequency(lnFrequency);
	QueryPerformanceCounter(lnStart);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
float EndTimer(LARGE_INTEGER *lnFrequency, LARGE_INTEGER *lnStart)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	LARGE_INTEGER	lnEnd;
	float			fDelta;

	QueryPerformanceCounter(&lnEnd);
	fDelta = (float)(lnEnd.QuadPart - (*lnStart).QuadPart) / (*lnFrequency).QuadPart;

	return(fDelta);
}
#endif


#ifdef _VXWORKS
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void StartTimer(uint64_t *lnFrequency, uint64_t *lnStart)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	*lnFrequency = (uint64_t)plMiscRegs_getSysClockFreqInHz();
	*lnStart = plMiscRegs_getSysTimeInTicks();
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
float EndTimer(uint64_t *lnFrequency, uint64_t *lnStart)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	uint64_t	lnEnd;
	float		fDelta;

	lnEnd = plMiscRegs_getSysTimeInTicks();
	fDelta = (float)(lnEnd - *lnStart) / (float)(*lnFrequency);

	return(fDelta);
}
#endif

#ifdef _LINUX
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void StartTimer(uint64_t *lnFrequency, struct timespec *timeStart)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	/* Get current time, */
	clock_gettime(CLOCK_REALTIME, timeStart);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
float EndTimer(uint64_t *lnFrequency, struct timespec *timeStart)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	struct timespec timeEnd;
	float		fDelta;

	/* Get current time, */
	clock_gettime(CLOCK_REALTIME, &timeEnd);

	fDelta = (timeEnd.tv_sec - timeStart->tv_sec) + (timeEnd.tv_nsec - timeStart->tv_nsec) / BILLION;

	return(fDelta);
}
#endif

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void FormatTime(float fSeconds, char *sBuffer)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	if (fSeconds > 1.0f)
	{
		sprintf(sBuffer, "%0.3f ", fSeconds);
	}
	else if (fSeconds > 0.001f)
	{
		sprintf(sBuffer, "%0.3fms", fSeconds*1000.0f);
	}
	else if (fSeconds > 0.000001f)
	{
		sprintf(sBuffer, "%0.3fus", fSeconds*1000000.0f);
	}
	else
		sprintf(sBuffer, "%0.3fs ", fSeconds);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
char *ReverseString(char *str)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	char *p1, *p2;

	if (!str || !*str)
		return str;
	for (p1 = str, p2 = str + strlen(str) - 1; p2 > p1; ++p1, --p2)
	{
		*p1 ^= *p2;
		*p2 ^= *p1;
		*p1 ^= *p2;
	}
	return str;
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
//void ShuffleSwap(int* xp, int* yp)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
/*
{
	int temp = *xp;

	*xp = *yp;
	*yp = temp;
}
*/
void ShuffleSwap(int* xp, int* yp)
{
    if (xp == NULL || yp == NULL) {
        printf("Error: Null pointer passed to ShuffleSwap. xp=%p, yp=%p\n", (void*)xp, (void*)yp);
        return;
    }

    //printf("Before swap: xp=%p, *xp=%d, yp=%p, *yp=%d\n", (void*)xp, *xp, (void*)yp, *yp);

    int temp = *xp;
    *xp = *yp;
    *yp = temp;

    //printf("After swap: xp=%p, *xp=%d, yp=%p, *yp=%d\n", (void*)xp, *xp, (void*)yp, *yp);
}
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void ShuffleArray(int* nIndexArray, int nCount, int nMode)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	int	i;
	//int p;
  unsigned int p;
	if (!nMode)
	{
		for (i = 0; i < nCount; ++i)
			nIndexArray[i] = i;
	}

	for (i = nCount; i > 1; i--)
	{
		p = (unsigned int )(rand() * rand()) % (i);
		//printf("ShuffleArray: i=%d, p=%d, nIndexArray[i-1]=%d, nIndexArray[p]=%d\n", i, p, nIndexArray[i-1], nIndexArray[p]);

		ShuffleSwap(&nIndexArray[i - 1], &nIndexArray[p]); // swap the values at i-1 and p
	}
}

/*
void ShuffleArray(int* nIndexArray, int nCount, int nMode)
{
    int i;
    int p;

    if (!nMode) {
        for (i = 0; i < nCount; ++i)
            nIndexArray[i] = i;
    }

    srand(time(NULL)); // Seed the random number generator

    for (i = nCount; i > 1; i--) {
        p = rand() % i; // Generate a random number in the range [0, i-1]
        
        //printf("ShuffleArray: i=%d, p=%d, nIndexArray[i-1]=%d, nIndexArray[p]=%d\n", i, p, nIndexArray[i-1], nIndexArray[p]);

        ShuffleSwap(&nIndexArray[i - 1], &nIndexArray[p]); // swap the values at i-1 and p
    }
}
*/

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int CompareMissCounts(const void* a, const void* b)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	int		nA = ((structInputData*)a)->nMissCount;
	int		nB = ((structInputData*)b)->nMissCount;


	if (nA > nB)
		return -1;

	return nA < nB;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int CompareDifference(const void* a, const void* b)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	float	fA = ((structInputData*)a)->fDifference;
	float	fB = ((structInputData*)b)->fDifference;


	if (fA < fB)
		return -1;

	return fA > fB;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int CompareFloatAscend(const void* a, const void* b)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	float	fA = *(const float*)a;
	float	fB = *(const float*)b;

	return (fA > fB) - (fA < fB);
}
