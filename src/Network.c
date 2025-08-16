
#include "main.h"

extern int	glblSegmentCount;

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void Initialize_Network(structNetwork *networkMain)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	networkMain->nID = 0;
	networkMain->clnHead = NULL;
	networkMain->clnCur = NULL;
	networkMain->classHead = NULL;
	networkMain->layerNew = NULL;
	networkMain->layerInput = NULL;
	networkMain->architecture = NULL;
	strcpy(networkMain->sTitle, "");
	strcpy(networkMain->sDrive, "");
	strcpy(networkMain->sDataSource, "");
	strcpy(networkMain->sTrainingFilePath, "");
	strcpy(networkMain->sTestingFilePath, "");
	strcpy(networkMain->sNetworkFilePath, "");
	strcpy(networkMain->sConfigFilePath, "");
	strcpy(networkMain->sFilePath, "");
	networkMain->nMatrix = NULL;
	networkMain->nClassMemberCount = NULL;
	networkMain->nClassCount = 0;
	networkMain->nDataSource = 0;
	networkMain->nRowCount = 0;
	networkMain->nColumnCount = 0;
	networkMain->nWeightCount = 0;
	networkMain->nPerceptronID = 0;
	networkMain->nSynapseID = 0;
	networkMain->nTrainVerifySplit = 50;
	networkMain->nPrimingCycles = 0;
	networkMain->nTrainingCycles = 0;
	networkMain->nClassLevelNetworkCount = 0;


	networkMain->fInputArray = NULL;
	networkMain->fMaximumThreshold = 1.0f;
	networkMain->fMinimumThreshold = -1.0f;
	networkMain->fLearningRate = 0.001f;
	networkMain->fInitialError = 1.0f;
	networkMain->fThreshold = 0.000f;

	networkMain->nLayerCount = 0;
	networkMain->nLayerTypeArray = NULL;
	networkMain->nLayerMapCountArray = NULL;
	networkMain->nLayerRowCountArray = NULL;
	networkMain->nLayerColumnCountArray = NULL;
	networkMain->nLayerRowStrideArray = NULL;
	networkMain->nLayerColumnStrideArray = NULL;
	networkMain->nLayerNeuronsArray = NULL;

	networkMain->nKernelCountStart = 0;
	networkMain->nKernelCountEnd = 0;
	networkMain->nSubNetCount = 0;
	networkMain->nBuildSort = 1;
	networkMain->nBuildThreshold = 0;
	networkMain->nTrainSort = 0;
	networkMain->nTrainThreshold = 0;
	networkMain->nPostTrain = 0;

	networkMain->bAdjustGlobalLearningRate = 0;
	networkMain->fLearningRateMinimum = 0.001f;
	networkMain->fLearningRateMaximum = 0.001f;
	networkMain->bAdjustPerceptronLearningRate;
	networkMain->bAdjustThreshold = 0;
	networkMain->fTargetWeight = 0.000065f;
	networkMain->nOpMode = 0;
	networkMain->bPruneNetwork = 0;
	networkMain->fPruneFCThreshold = 0.001f;
	networkMain->fPruneConvThreshold = 0.001f;
	networkMain->bLearnRateInitialization = 0;
}



/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void CreateCombiningClassifier_Network(structNetwork *networkMain, float *fRandomWeightarray, int nRandomizeMode)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structCLN			*clnNew = NULL;
	structLayer			*layerNew = NULL;
	structCLN			*clnCur = NULL;
	structPerceptron	*perceptronNewHead = NULL;
	structPerceptron	*perceptronNew = NULL;
	structPerceptron	*perceptronCur = NULL;
	structClass			*curClass = NULL;
	structSynapse		*synapseNew = NULL;
	int					nSynapseIndex;
	int					nNeuralPreviousCount;
	int					nNeuronCount;
	int					nCLNCount;

	nCLNCount = 0;
	nSynapseIndex = 0;
	nNeuralPreviousCount = 0;
	nNeuronCount = 0;

	for (clnCur = networkMain->clnHead; clnCur != NULL; clnCur = clnCur->next, ++nCLNCount)
	{
		for (perceptronCur = clnCur->layerClassifier->perceptronHead; perceptronCur != NULL; perceptronCur = perceptronCur->next, ++nNeuralPreviousCount);
	}

	for (curClass = networkMain->classHead, nNeuronCount = 0; curClass != NULL; curClass = curClass->next, ++nNeuronCount);

	// Create Weight Array
	if ((layerNew = (structLayer *)calloc(1, sizeof(structLayer))) == NULL)
		exit(0);

	layerNew->nLayerType = COMBINING_CLASSIFIER_LAYER;
	layerNew->nInputRowCount = 0;
	layerNew->nInputColumnCount = 0;
	layerNew->nPerceptronCount = 0;
	layerNew->nWeightCount = nNeuronCount * (nNeuralPreviousCount + 1);
	layerNew->fWeightArray = (float *)calloc(layerNew->nWeightCount, sizeof(float *));

	if ((clnNew = (structCLN *)calloc(1, sizeof(structCLN))) == NULL)
		exit(0);

	clnNew->nID = 0;
	clnNew->fLearningRate = 0.001f; //networkMain->fLearningRate;
	clnNew->fInitialError = networkMain->fInitialError;
	clnNew->fThreshold = networkMain->fThreshold;
	clnNew->nSize = 0;
	clnNew->nPerceptronLayerCount = 0;

	for (curClass = networkMain->classHead; curClass != NULL; curClass = curClass->next)
	{
		perceptronNew = (structPerceptron *)calloc(1, sizeof(structPerceptron));
		perceptronNew->nID = networkMain->nPerceptronID++;

		perceptronNew->nLayerType = COMBINING_CLASSIFIER_LAYER;
		perceptronNew->nIndex = curClass->nID;
		perceptronNew->fLearningRate = 0.001f; //networkMain->fLearningRate;

		//Add Bias Synapse ////////////////////////////////////////////////////////////////////////
		synapseNew = (structSynapse *)calloc(1, sizeof(structSynapse));
		synapseNew->nID = networkMain->nSynapseID;
		synapseNew->nIndex = nSynapseIndex;
		synapseNew->fWeight = &layerNew->fWeightArray[synapseNew->nIndex];
		synapseNew->fInput = (float *)calloc(1, sizeof(float));
		*(synapseNew->fInput) = 1; // Bias
		synapseNew->nInputArrayIndex = -1;

		AddNew_Synapse(&perceptronNew->synapseHead, synapseNew);

		++networkMain->nSynapseID;
		++nSynapseIndex;

		for (clnCur = networkMain->clnHead; clnCur != NULL; clnCur = clnCur->next, ++nCLNCount)
		{
			for (perceptronCur = clnCur->layerClassifier->perceptronHead; perceptronCur != NULL; perceptronCur = perceptronCur->next, ++networkMain->nSynapseID, ++nSynapseIndex)
			{
				synapseNew = (structSynapse *)calloc(1, sizeof(structSynapse));

				synapseNew->nID = networkMain->nSynapseID;
				synapseNew->nIndex = nSynapseIndex;

				synapseNew->fWeight = &layerNew->fWeightArray[synapseNew->nIndex];
				synapseNew->perceptronConnectTo = perceptronCur;
				synapseNew->fInput = &synapseNew->perceptronConnectTo->fOutput;
				synapseNew->nInputArrayIndex = -999;

				AddNew_Synapse(&perceptronNew->synapseHead, synapseNew);
			}
		}

		++layerNew->nPerceptronCount;
		perceptronNew->nWeightCount = nSynapseIndex;
		AddNewV2_Perceptron(&perceptronNewHead, perceptronNew);
	}


	perceptronNewHead->nDimX = 1;
	perceptronNewHead->nDimY = nNeuronCount;

	AddToLayer_Perceptron(&layerNew->perceptronHead, perceptronNewHead, 0);

	layerNew->nConnectionCount = layerNew->nWeightCount - layerNew->nPerceptronCount;
	layerNew->nOutputArraySize = layerNew->nPerceptronCount;

	AddNew_Layer(&clnNew->layerHead, layerNew);

	
	for (perceptronCur = layerNew->perceptronHead; perceptronCur != NULL; perceptronCur = perceptronCur->next)
		perceptronCur->nLayerID = layerNew->nID;

	clnNew->perceptronClassifier = layerNew->perceptronHead;

	GetClassLevelNetworkWeightCount(clnNew);
	CreateMACArray_ClassLevelNetworks(clnNew);
	InitializeWeights_ClassLevelNetworks(clnNew, nRandomizeMode, 0.0f);
	AddNew_ClassLevelNetworks(&networkMain->clnHead, clnNew);

	return;
}



/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void DeleteNetwork(int nLayers, structPerceptron *m_perceptronLayerHead[])
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	int	i;

	for(i=0; i<nLayers; ++i) 
	{
		Free_Layer(m_perceptronLayerHead[i]);
	}
}






/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void WriteDNANetwork(structNetwork* networkMain, char* sFilePath)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	FILE				*pFile;
	structCLN			*clnCur = NULL;
	structLayer			*layerCur;
	structPerceptron	*perceptronHeadCur = NULL;
	structPerceptron	*perceptronHead = NULL;
	structPerceptron	*perceptronCur;
	structSynapse		*synapseCur;
	float				fZero = 0.0f;
	char				sTemp[256];
	int					nLayerCount;
	int					nHeadPerceptronCount;
	int					nPerceptronCount;
	int					nSynapseCount;
	int					nActivationMode=2;
	int					nValue =1;

	if (strlen(networkMain->sOutputStructurePath) == 0)
		sprintf(networkMain->sOutputStructurePath, "%s", sFilePath);


	if ((strstr(networkMain->sOutputStructurePath, ".dna")) == NULL)
	{
		sprintf(sTemp, "%s/%s_x.dna", networkMain->sOutputStructurePath, networkMain->sConfigFile);
		strcpy(networkMain->sOutputStructurePath, sTemp);
		SetFilePath(networkMain, networkMain->sOutputStructurePath);
	}


	if ((pFile = FOpenMakeDirectory(networkMain->sOutputStructurePath, "wb")) == NULL)
	{
		printf("Could not write file -- %s\n\n", sFilePath);
		while (1);
	}

	fwrite(&networkMain->nRowCount, sizeof(int), 1, pFile);
	fwrite(&networkMain->nColumnCount, sizeof(int), 1, pFile);

	for (layerCur = networkMain->clnHead->layerHead, nLayerCount = 0; layerCur != NULL; layerCur = layerCur->next, ++nLayerCount);
	fwrite(&nLayerCount, sizeof(int), 1, pFile);

	for (layerCur = networkMain->clnHead->layerHead; layerCur != NULL; layerCur = layerCur->next)
	{
		fwrite(&layerCur->nLayerType, sizeof(int), 1, pFile);
		fwrite(&nActivationMode, sizeof(int), 1, pFile);

		if (layerCur->nLayerType == CONV_2D_LAYER || layerCur->nLayerType == CONV_3D_LAYER)
		{
			for (perceptronHeadCur = layerCur->perceptronHead, nHeadPerceptronCount = 0; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead, ++nHeadPerceptronCount);
			fwrite(&nHeadPerceptronCount, sizeof(int), 1, pFile);

			// Filters
			for (perceptronHeadCur = layerCur->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
			{
				for (perceptronCur = perceptronHeadCur, nPerceptronCount = 0; perceptronCur != NULL; perceptronCur = perceptronCur->next, ++nPerceptronCount);
				fwrite(&nPerceptronCount, sizeof(int), 1, pFile);

				for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
				{
					if (perceptronCur == perceptronHeadCur)  // bias
						fwrite(perceptronCur->synapseHead->fWeight, sizeof(float), 1, pFile);

					for (synapseCur = perceptronCur->synapseHead->next, nSynapseCount = 0; synapseCur != NULL; synapseCur = synapseCur->next, ++nSynapseCount);
					fwrite(&nSynapseCount, sizeof(int), 1, pFile);

					for (synapseCur = perceptronCur->synapseHead->next; synapseCur != NULL; synapseCur = synapseCur->next)
					{
						if (perceptronCur == perceptronHeadCur)
							fwrite(synapseCur->fWeight, sizeof(float), 1, pFile);

						fwrite(&nValue, sizeof(int), 1, pFile);
						
						if (synapseCur->nInputArrayIndex == -999) // Connect to perceptron output
							fwrite(&synapseCur->perceptronConnectTo->nID, sizeof(int), 1, pFile);
						else
							fwrite(&synapseCur->nInputArrayIndex, sizeof(int), 1, pFile);
					}
				}
			}
		}
		else if (layerCur->nLayerType == MAX_POOLING_LAYER)
		{
			for (perceptronHeadCur = layerCur->perceptronHead, nHeadPerceptronCount = 0; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead, ++nHeadPerceptronCount);
			fwrite(&nHeadPerceptronCount, sizeof(int), 1, pFile);

			// Filters
			for (perceptronHeadCur = layerCur->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
			{
				for (perceptronCur = perceptronHeadCur, nPerceptronCount = 0; perceptronCur != NULL; perceptronCur = perceptronCur->next, ++nPerceptronCount);
				fwrite(&nPerceptronCount, sizeof(int), 1, pFile);

				for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
				{
					for (synapseCur = perceptronCur->synapseHead, nSynapseCount = 0; synapseCur != NULL; synapseCur = synapseCur->next, ++nSynapseCount);
					fwrite(&nSynapseCount, sizeof(int), 1, pFile);

					for (synapseCur = perceptronCur->synapseHead; synapseCur != NULL; synapseCur = synapseCur->next)
					{
						fwrite(&nValue, sizeof(int), 1, pFile);

						if (synapseCur->nInputArrayIndex == -999) // Connect to perceptron output
							fwrite(&synapseCur->perceptronConnectTo->nID, sizeof(int), 1, pFile);
						else
							fwrite(&synapseCur->nInputArrayIndex, sizeof(int), 1, pFile);
					}
				}
			}
		}
		else if (layerCur->nLayerType == FULLY_CONNECTED_LAYER || layerCur->nLayerType == CLASSIFIER_LAYER)
		{
			for (perceptronCur = perceptronHeadCur, nPerceptronCount = 0; perceptronCur != NULL; perceptronCur = perceptronCur->next, ++nPerceptronCount);
			fwrite(&nPerceptronCount, sizeof(int), 1, pFile);

			for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
			{
				if (perceptronCur == perceptronHeadCur)  // bias
					fwrite(perceptronCur->synapseHead->fWeight, sizeof(float), 1, pFile);

				for (synapseCur = perceptronCur->synapseHead, nSynapseCount = 0; synapseCur != NULL; synapseCur = synapseCur->next, ++nSynapseCount);
				fwrite(&nSynapseCount, sizeof(int), 1, pFile);

				for (synapseCur = perceptronCur->synapseHead; synapseCur != NULL; synapseCur = synapseCur->next)
				{
					fwrite(synapseCur->fWeight, sizeof(float), 1, pFile);

					fwrite(&nValue, sizeof(int), 1, pFile);

					if (synapseCur->nInputArrayIndex == -999) // Connect to perceptron output
						fwrite(&synapseCur->perceptronConnectTo->nID, sizeof(int), 1, pFile);
					else
						fwrite(&synapseCur->nInputArrayIndex, sizeof(int), 1, pFile);
				}
			}
		}
	}

	fclose(pFile);
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void Write_DNA_Network_V2(structNetwork *networkMain, char *sFilePath)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	FILE				*pFile;
	structCLN			*clnCur = NULL;
	structLayer			*layerCur;
	structPerceptron	*perceptronHeadCur = NULL;
	structPerceptron	*perceptronHead = NULL;
	structPerceptron	*perceptronCur;
	structSynapse		*synapseCur;
	float				fZero = 0.0f;
	char				sTemp[256];
	int					nLayerCount;
	int					nPerceptronCount;
	int					nSynapseCount;
	int					nInputCount=1;
	int					nActivationMode= MTANH;

	if(strlen(networkMain->sDNAOutputPath) == 0)
		sprintf(networkMain->sDNAOutputPath, "%s", sFilePath);

	
	if ((strstr(networkMain->sDNAOutputPath, ".dna")) == NULL)
	{
		sprintf(sTemp, "%s/%s_x.dna", networkMain->sDNAOutputPath, networkMain->sConfigFile);
		strcpy(networkMain->sDNAOutputPath, sTemp);
		SetFilePath(networkMain, networkMain->sDNAOutputPath);
	}
	
	
	if ((pFile = FOpenMakeDirectory(networkMain->sDNAOutputPath, "wb")) == NULL)
	{
		printf("Could not write file -- %s\n\n", sFilePath);
		while (1);
	}

	strcpy(sTemp, "DNA\n");
	fwrite(sTemp, sizeof(char), 4, pFile);
	//printf("sTemp: %s\n", sTemp);
	fwrite(&networkMain->nRowCount, sizeof(int), 1, pFile);
	//printf("nRowCount: %d\n", &networkMain->nRowCount);
	fwrite(&networkMain->nColumnCount, sizeof(int), 1, pFile);
	//printf("nColumnCount: %d\n", &networkMain->nColumnCount);

	for (layerCur = networkMain->clnHead->layerHead, nLayerCount = 0; layerCur != NULL; layerCur = layerCur->next, ++nLayerCount){
		//printf("nLayerCount: %d\n", &networkMain->nColumnCount); 
		fwrite(&nLayerCount, sizeof(int), 1, pFile);
	}

	for (layerCur = networkMain->clnHead->layerHead; layerCur != NULL; layerCur = layerCur->next)
	{
		fwrite(&layerCur->nLayerType, sizeof(int), 1, pFile);
		fwrite(&nActivationMode, sizeof(int), 1, pFile);

		if (layerCur->nLayerType == MAX_POOL_LAYER)
		{
			for (perceptronHeadCur = layerCur->perceptronHead, nPerceptronCount = 0; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
			{
				for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next, ++nPerceptronCount);
			}

			fwrite(&nPerceptronCount, sizeof(int), 1, pFile);

			for (perceptronHeadCur = layerCur->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
			{
				for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
				{
					for (synapseCur = perceptronCur->synapseHead, nSynapseCount = 0; synapseCur != NULL; synapseCur = synapseCur->next, ++nSynapseCount);
					fwrite(&nSynapseCount, sizeof(int), 1, pFile);

					for (synapseCur = perceptronCur->synapseHead; synapseCur != NULL; synapseCur = synapseCur->next)
					{
						fwrite(&nInputCount, sizeof(int), 1, pFile);
						fwrite(&synapseCur->nInputArrayIndex, sizeof(int), 1, pFile);
					}
				}
			}
		}
		else if (layerCur->nLayerType == FULLY_CONNECTED_LAYER || layerCur->nLayerType == DENSE_LAYER || layerCur->nLayerType == CLASSIFIER_LAYER)
		{
			for (perceptronHeadCur = layerCur->perceptronHead, nPerceptronCount = 0; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
			{
				for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next, ++nPerceptronCount);
			}

			fwrite(&nPerceptronCount, sizeof(int), 1, pFile);

			for (perceptronHeadCur = layerCur->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
			{
				for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
				{
					synapseCur = perceptronCur->synapseHead;
					fwrite(synapseCur->fWeight, sizeof(float), 1, pFile);

					for (synapseCur = synapseCur->next, nSynapseCount = 0; synapseCur != NULL; synapseCur = synapseCur->next, ++nSynapseCount);
					fwrite(&nSynapseCount, sizeof(int), 1, pFile);

					for (synapseCur = perceptronCur->synapseHead->next; synapseCur != NULL; synapseCur = synapseCur->next)
					{
						if (synapseCur->fWeight != NULL)
							fwrite(synapseCur->fWeight, sizeof(float), 1, pFile);
						else
							fwrite(&fZero, sizeof(float), 1, pFile);

						fwrite(&nInputCount, sizeof(int), 1, pFile);

						if (synapseCur->nInputArrayIndex == -999) // Connect to perceptron output
							fwrite(&synapseCur->perceptronConnectTo->nID, sizeof(int), 1, pFile);
						else
							fwrite(&synapseCur->nInputArrayIndex, sizeof(int), 1, pFile);
					}
				}
			}
		}
	}


	fclose(pFile);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void Write_DNA_Network(structNetwork* networkMain, char* sFilePath)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	FILE* pFile;
	structCLN* clnCur = NULL;
	structLayer* layerCur;
	structPerceptron* perceptronHeadCur = NULL;
	structPerceptron* perceptronHead = NULL;
	structPerceptron* perceptronCur;
	structSynapse* synapseCur;
	float				fZero = 0.0f;
	int					nCLNCount;
	int					nLayerCount;
	int					nHeadPerceptronCount;
	int					nPerceptronCount;
	int					nSynapseCount;


	if ((pFile = FOpenMakeDirectory(sFilePath, "wb")) == NULL)
	{
		printf("Could not write file -- %s\n\n", sFilePath);
		while (1);
	}

	fwrite(&networkMain->nRowCount, sizeof(int), 1, pFile);
	fwrite(&networkMain->nColumnCount, sizeof(int), 1, pFile);

	for (clnCur = networkMain->clnHead, nCLNCount = 0; clnCur != NULL; clnCur = clnCur->next, ++nCLNCount);
	fwrite(&nCLNCount, sizeof(int), 1, pFile);

	for (clnCur = networkMain->clnHead; clnCur != NULL; clnCur = clnCur->next)
	{
		fwrite(&clnCur->nID, sizeof(int), 1, pFile);
		fwrite(&clnCur->fLearningRate, sizeof(float), 1, pFile);
		fwrite(&clnCur->fInitialError, sizeof(float), 1, pFile);
		fwrite(&clnCur->fThreshold, sizeof(float), 1, pFile);
		fwrite(&clnCur->nLabelID, sizeof(int), 1, pFile);
		fwrite(&clnCur->nSize, sizeof(int), 1, pFile);
		fwrite(&clnCur->nPerceptronLayerCount, sizeof(int), 1, pFile);

		for (layerCur = clnCur->layerHead, nLayerCount = 0; layerCur != NULL; layerCur = layerCur->next, ++nLayerCount);
		fwrite(&nLayerCount, sizeof(int), 1, pFile);

		for (layerCur = clnCur->layerHead; layerCur != NULL; layerCur = layerCur->next)
		{
			fwrite(&layerCur->nID, sizeof(int), 1, pFile);
			fwrite(&layerCur->nLayerType, sizeof(int), 1, pFile);
			fwrite(&layerCur->nKernelCount, sizeof(int), 1, pFile);
			fwrite(&layerCur->nInputRowCount, sizeof(int), 1, pFile);
			fwrite(&layerCur->nInputColumnCount, sizeof(int), 1, pFile);
			fwrite(&layerCur->nKernelRowCount, sizeof(int), 1, pFile);
			fwrite(&layerCur->nKernelColumnCount, sizeof(int), 1, pFile);
			fwrite(&layerCur->nStrideRow, sizeof(int), 1, pFile);
			fwrite(&layerCur->nStrideColumn, sizeof(int), 1, pFile);
			fwrite(&layerCur->nWeightCount, sizeof(int), 1, pFile);
			fwrite(&layerCur->nOutputArraySize, sizeof(int), 1, pFile);
			fwrite(&layerCur->nOffset, sizeof(int), 1, pFile);
			fwrite(&layerCur->nPerceptronCount, sizeof(int), 1, pFile);
			fwrite(&layerCur->nConnectionCount, sizeof(int), 1, pFile);
			fwrite(&layerCur->fGamma, sizeof(float), 1, pFile);
			fwrite(&layerCur->fLambda, sizeof(float), 1, pFile);

			for (perceptronHeadCur = layerCur->perceptronHead, nHeadPerceptronCount = 0; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead, ++nHeadPerceptronCount);
			fwrite(&nHeadPerceptronCount, sizeof(int), 1, pFile);

			for (perceptronHeadCur = layerCur->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
			{
				for (perceptronCur = perceptronHeadCur, nPerceptronCount = 0; perceptronCur != NULL; perceptronCur = perceptronCur->next, ++nPerceptronCount);
				fwrite(&nPerceptronCount, sizeof(int), 1, pFile);

				for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
				{
					fwrite(&perceptronCur->nLayerType, sizeof(int), 1, pFile);
					fwrite(&perceptronCur->nID, sizeof(int), 1, pFile);
					fwrite(&perceptronCur->nHeadIndex, sizeof(int), 1, pFile);
					fwrite(&perceptronCur->nIndex, sizeof(int), 1, pFile);
					fwrite(&perceptronCur->fOutput, sizeof(float), 1, pFile);
					fwrite(&perceptronCur->nDimX, sizeof(int), 1, pFile);
					fwrite(&perceptronCur->nDimY, sizeof(int), 1, pFile);
					fwrite(&perceptronCur->nConnectionCount, sizeof(int), 1, pFile);
					fwrite(&perceptronCur->nWeightCount, sizeof(int), 1, pFile);
					fwrite(&perceptronCur->fBias, sizeof(float), 1, pFile);
					fwrite(&perceptronCur->fError, sizeof(float), 1, pFile);
					fwrite(&perceptronCur->fDifferential, sizeof(float), 1, pFile);
					fwrite(&perceptronCur->fLearningRate, sizeof(float), 1, pFile);
					fwrite(&perceptronCur->fThreshold, sizeof(float), 1, pFile);
					fwrite(&perceptronCur->nLayerID, sizeof(int), 1, pFile);
					fwrite(&perceptronCur->nClusterCount, sizeof(int), 1, pFile);

					for (synapseCur = perceptronCur->synapseHead, nSynapseCount = 0; synapseCur != NULL; synapseCur = synapseCur->next, ++nSynapseCount);
					fwrite(&nSynapseCount, sizeof(int), 1, pFile);

					for (synapseCur = perceptronCur->synapseHead; synapseCur != NULL; synapseCur = synapseCur->next)
					{
						fwrite(&synapseCur->nID, sizeof(int), 1, pFile);
						fwrite(&synapseCur->nIndex, sizeof(int), 1, pFile);
						fwrite(&synapseCur->nCluster, sizeof(int), 1, pFile);
						fwrite(&synapseCur->bAdjust, sizeof(int), 1, pFile);

						if (synapseCur->fWeight != NULL)
							fwrite(synapseCur->fWeight, sizeof(float), 1, pFile);
						else
							fwrite(&fZero, sizeof(float), 1, pFile);

						fwrite(&synapseCur->nInputArrayIndex, sizeof(int), 1, pFile);

						if (synapseCur->nInputArrayIndex == -999) // Connect to perceptron output
							fwrite(&synapseCur->perceptronConnectTo->nID, sizeof(int), 1, pFile);
						else
							fwrite(&synapseCur->nInputArrayIndex, sizeof(int), 1, pFile);
					}
				}
			}
		}
	}




	fclose(pFile);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void Write_TXT_Network(structNetwork* networkMain, char* sFilePath)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	FILE* pFile;
	structCLN* clnCur = NULL;
	structLayer* layerCur;
	structPerceptron* perceptronHeadCur = NULL;
	structPerceptron* perceptronHead = NULL;
	structPerceptron* perceptronCur;
	structSynapse* synapseCur;
	float				fZero = 0.0f;
	int					nCLNCount;
	int					nLayerCount;
	int					nHeadPerceptronCount;
	int					nPerceptronCount;
	int					nSynapseCount;

	if (sFilePath != NULL){
		pFile = fopen(sFilePath,"w");
	}
	else if ((pFile = FOpenMakeDirectory(sFilePath, "wb")) == NULL) {
		printf("Could not write file -- %s\n\n", sFilePath);
		while (1);
	}

	fprintf(pFile, "Network Row Count: %d\n", networkMain->nRowCount);
	fprintf(pFile, "Network Column Count: %d\n", networkMain->nColumnCount);

	for (clnCur = networkMain->clnHead, nCLNCount = 0; clnCur != NULL; clnCur = clnCur->next, ++nCLNCount);
	fprintf(pFile, "Number of CLNs: %d\n", nCLNCount);

	for (clnCur = networkMain->clnHead; clnCur != NULL; clnCur = clnCur->next)
	{
		fprintf(pFile, "CLN ID: %d\n", clnCur->nID);
		fprintf(pFile, "Learning Rate: %f\n", clnCur->fLearningRate);
		fprintf(pFile, "Initial Error: %f\n", clnCur->fInitialError);
		fprintf(pFile, "Threshold: %f\n", clnCur->fThreshold);
		fprintf(pFile, "Label ID: %d\n", clnCur->nLabelID);
		fprintf(pFile, "Size: %d\n", clnCur->nSize);
		fprintf(pFile, "Perceptron Layer Count: %d\n", clnCur->nPerceptronLayerCount);

		for (layerCur = clnCur->layerHead, nLayerCount = 0; layerCur != NULL; layerCur = layerCur->next, ++nLayerCount);
		fprintf(pFile, "Number of Layers in CLN %d: %d\n", clnCur->nID, nLayerCount);

		for (layerCur = clnCur->layerHead; layerCur != NULL; layerCur = layerCur->next)
		{
			fprintf(pFile, "  Layer ID: %d\n", layerCur->nID);
			fprintf(pFile, "  Layer Type: %d\n", layerCur->nLayerType);
			fprintf(pFile, "  Kernel Count: %d\n", layerCur->nKernelCount);
			fprintf(pFile, "  Input Row Count: %d\n", layerCur->nInputRowCount);
			fprintf(pFile, "  Input Column Count: %d\n", layerCur->nInputColumnCount);
			fprintf(pFile, "  Kernel Row Count: %d\n", layerCur->nKernelRowCount);
			fprintf(pFile, "  Kernel Column Count: %d\n", layerCur->nKernelColumnCount);
			fprintf(pFile, "  Stride Row: %d\n", layerCur->nStrideRow);
			fprintf(pFile, "  Stride Column: %d\n", layerCur->nStrideColumn);
			fprintf(pFile, "  Weight Count: %d\n", layerCur->nWeightCount);
			fprintf(pFile, "  Output Array Size: %d\n", layerCur->nOutputArraySize);
			fprintf(pFile, "  Offset: %d\n", layerCur->nOffset);
			fprintf(pFile, "  Perceptron Count: %d\n", layerCur->nPerceptronCount);
			fprintf(pFile, "  Connection Count: %d\n", layerCur->nConnectionCount);
			fprintf(pFile, "  Gamma: %f\n", layerCur->fGamma);
			fprintf(pFile, "  Lambda: %f\n", layerCur->fLambda);

			for (perceptronHeadCur = layerCur->perceptronHead, nHeadPerceptronCount = 0; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead, ++nHeadPerceptronCount);
			fprintf(pFile, "  Number of Head Perceptrons in Layer %d: %d\n", layerCur->nID, nHeadPerceptronCount);

			for (perceptronHeadCur = layerCur->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
			{
				for (perceptronCur = perceptronHeadCur, nPerceptronCount = 0; perceptronCur != NULL; perceptronCur = perceptronCur->next, ++nPerceptronCount);
				fprintf(pFile, "    Number of Perceptrons in Head %d: %d\n", perceptronHeadCur->nID, nPerceptronCount);

				for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
				{
					fprintf(pFile, "    Perceptron Layer Type: %d\n", perceptronCur->nLayerType);
					fprintf(pFile, "    Perceptron ID: %d\n", perceptronCur->nID);
					fprintf(pFile, "    Head Index: %d\n", perceptronCur->nHeadIndex);
					fprintf(pFile, "    Index: %d\n", perceptronCur->nIndex);
					fprintf(pFile, "    Output: %f\n", perceptronCur->fOutput);
					fprintf(pFile, "    Dimension X: %d\n", perceptronCur->nDimX);
					fprintf(pFile, "    Dimension Y: %d\n", perceptronCur->nDimY);
					fprintf(pFile, "    Connection Count: %d\n", perceptronCur->nConnectionCount);
					fprintf(pFile, "    Weight Count: %d\n", perceptronCur->nWeightCount);
					fprintf(pFile, "    Bias: %f\n", perceptronCur->fBias);
					fprintf(pFile, "    Error: %f\n", perceptronCur->fError);
					fprintf(pFile, "    Differential: %f\n", perceptronCur->fDifferential);
					fprintf(pFile, "    Learning Rate: %f\n", perceptronCur->fLearningRate);
					fprintf(pFile, "    Threshold: %f\n", perceptronCur->fThreshold);
					fprintf(pFile, "    Layer ID: %d\n", perceptronCur->nLayerID);
					fprintf(pFile, "    Cluster Count: %d\n", perceptronCur->nClusterCount);

					for (synapseCur = perceptronCur->synapseHead, nSynapseCount = 0; synapseCur != NULL; synapseCur = synapseCur->next, ++nSynapseCount);
					fprintf(pFile, "    Number of Synapses in Perceptron %d: %d\n", perceptronCur->nID, nSynapseCount);

					for (synapseCur = perceptronCur->synapseHead; synapseCur != NULL; synapseCur = synapseCur->next)
					{
						fprintf(pFile, "      Synapse ID: %d\n", synapseCur->nID);
						fprintf(pFile, "      Synapse Index: %d\n", synapseCur->nIndex);
						fprintf(pFile, "      Synapse Cluster: %d\n", synapseCur->nCluster);
						fprintf(pFile, "      Adjust: %d\n", synapseCur->bAdjust);

						if (synapseCur->fWeight != NULL)
							fprintf(pFile, "      Weight: %f\n", *(synapseCur->fWeight));
						else
							fprintf(pFile, "      Weight: %f\n", fZero);

						//fprintf(pFile, "      Input Array Index: %d\n", synapseCur->nInputArrayIndex);

						if (synapseCur->nInputArrayIndex == -999) // Connect to perceptron output
							fprintf(pFile, "      Connect to Perceptron ID: %d\n", synapseCur->perceptronConnectTo->nID);
						else
							fprintf(pFile, "      Input Array Index: %d\n", synapseCur->nInputArrayIndex);
					}
				}
			}
		}
	}
	fclose(pFile);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void Write_DNX_Network(structNetwork *networkMain, char *sFilePath)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	FILE				*pFile;
	structCLN			*clnCur = NULL;
	structLayer			*layerCur;
	structPerceptron	*perceptronHeadCur = NULL;
	structPerceptron	*perceptronHead = NULL;
	structPerceptron	*perceptronCur;
	structSynapse		*synapseCur;
	float				fZero = 0.0f;
	char				sTemp[256];
	int					nCLNCount;
	int					nLayerCount;
	int					nHeadPerceptronCount;
	int					nPerceptronCount;
	int					nSynapseCount;
	int					i;

	if ((strstr(networkMain->sDNXOutputPath, ".dnx")) == NULL)
	{
		sprintf(sTemp, "%s/%s_x.dnx", networkMain->sDNXOutputPath, networkMain->sConfigFile);
		strcpy(networkMain->sDNXOutputPath, sTemp);
		SetFilePath(networkMain, networkMain->sDNXOutputPath);
	}


	if ((pFile = FOpenMakeDirectory(networkMain->sDNXOutputPath, "wb")) == NULL)
	{
		printf("Could not write file -- %s\n\n", sFilePath);
		while (1);
	}

	fwrite(&networkMain->nRowCount, sizeof(int), 1, pFile);
	fwrite(&networkMain->nColumnCount, sizeof(int), 1, pFile);



	for (clnCur = networkMain->clnHead, nCLNCount = 0; clnCur != NULL; clnCur = clnCur->next, ++nCLNCount);
	fwrite(&nCLNCount, sizeof(int), 1, pFile);

	for (clnCur = networkMain->clnHead; clnCur != NULL; clnCur = clnCur->next)
	{
		fwrite(&clnCur->nID, sizeof(int), 1, pFile);
		fwrite(&clnCur->fLearningRate, sizeof(float), 1, pFile);
		fwrite(&clnCur->fInitialError, sizeof(float), 1, pFile);
		fwrite(&clnCur->fThreshold, sizeof(float), 1, pFile);
		fwrite(&clnCur->nLabelID, sizeof(int), 1, pFile);
		fwrite(&clnCur->nSize, sizeof(int), 1, pFile);
		fwrite(&clnCur->nPerceptronLayerCount, sizeof(int), 1, pFile);

		for (layerCur = clnCur->layerHead, nLayerCount = 0; layerCur != NULL; layerCur = layerCur->next, ++nLayerCount);
		fwrite(&nLayerCount, sizeof(int), 1, pFile);

		for (layerCur = clnCur->layerHead; layerCur != NULL; layerCur = layerCur->next)
		{
			fwrite(&layerCur->nID, sizeof(int), 1, pFile);
			fwrite(&layerCur->nLayerType, sizeof(int), 1, pFile);
			fwrite(&layerCur->nKernelCount, sizeof(int), 1, pFile);
			fwrite(&layerCur->nInputRowCount, sizeof(int), 1, pFile);
			fwrite(&layerCur->nInputColumnCount, sizeof(int), 1, pFile);
			fwrite(&layerCur->nKernelRowCount, sizeof(int), 1, pFile);
			fwrite(&layerCur->nKernelColumnCount, sizeof(int), 1, pFile);
			fwrite(&layerCur->nStrideRow, sizeof(int), 1, pFile);
			fwrite(&layerCur->nStrideColumn, sizeof(int), 1, pFile);
			fwrite(&layerCur->nWeightCount, sizeof(int), 1, pFile);
			fwrite(&layerCur->nOutputArraySize, sizeof(int), 1, pFile);
			fwrite(&layerCur->nOffset, sizeof(int), 1, pFile);
			fwrite(&layerCur->nPerceptronCount, sizeof(int), 1, pFile);
			fwrite(&layerCur->nConnectionCount, sizeof(int), 1, pFile);
			fwrite(&layerCur->fGamma, sizeof(float), 1, pFile);
			fwrite(&layerCur->fLambda, sizeof(float), 1, pFile);

			for (perceptronHeadCur = layerCur->perceptronHead, nHeadPerceptronCount = 0; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead, ++nHeadPerceptronCount);
			fwrite(&nHeadPerceptronCount, sizeof(int), 1, pFile);

			for (perceptronHeadCur = layerCur->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
			{
				for (perceptronCur = perceptronHeadCur, nPerceptronCount = 0; perceptronCur != NULL; perceptronCur = perceptronCur->next, ++nPerceptronCount);
				fwrite(&nPerceptronCount, sizeof(int), 1, pFile);

				for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
				{
					fwrite(&perceptronCur->nLayerType, sizeof(int), 1, pFile);
					fwrite(&perceptronCur->nID, sizeof(int), 1, pFile);
					fwrite(&perceptronCur->nHeadIndex, sizeof(int), 1, pFile);
					fwrite(&perceptronCur->nIndex, sizeof(int), 1, pFile);
					fwrite(&perceptronCur->fOutput, sizeof(float), 1, pFile);
					fwrite(&perceptronCur->nDimX, sizeof(int), 1, pFile);
					fwrite(&perceptronCur->nDimY, sizeof(int), 1, pFile);
					fwrite(&perceptronCur->nConnectionCount, sizeof(int), 1, pFile);
					fwrite(&perceptronCur->nWeightCount, sizeof(int), 1, pFile);
					fwrite(&perceptronCur->fBias, sizeof(float), 1, pFile);
					fwrite(&perceptronCur->fError, sizeof(float), 1, pFile);
					fwrite(&perceptronCur->fDifferential, sizeof(float), 1, pFile);
					fwrite(&perceptronCur->fLearningRate, sizeof(float), 1, pFile);
					fwrite(&perceptronCur->fThreshold, sizeof(float), 1, pFile);
					fwrite(&perceptronCur->nLayerID, sizeof(int), 1, pFile);
					fwrite(&perceptronCur->nClusterCount, sizeof(int), 1, pFile);

					for (synapseCur = perceptronCur->synapseHead, nSynapseCount = 0; synapseCur != NULL; synapseCur = synapseCur->next, ++nSynapseCount);
					fwrite(&nSynapseCount, sizeof(int), 1, pFile);

					for (synapseCur = perceptronCur->synapseHead; synapseCur != NULL; synapseCur = synapseCur->next)
					{
						fwrite(&synapseCur->nID, sizeof(int), 1, pFile);
						fwrite(&synapseCur->nIndex, sizeof(int), 1, pFile);
						fwrite(&synapseCur->nCluster, sizeof(int), 1, pFile);
						fwrite(&synapseCur->bAdjust, sizeof(int), 1, pFile);
						
						if(layerCur->nLayerType == MAX_POOL_LAYER)
							fwrite(&fZero, sizeof(float), 1, pFile);
						else
							fwrite(synapseCur->fWeight, sizeof(float), 1, pFile);
						
						
						fwrite(&synapseCur->nInputArrayIndex, sizeof(int), 1, pFile);

						if (synapseCur->nInputArrayIndex == -999) // Connect to perceptron output
							fwrite(&synapseCur->perceptronConnectTo->nID, sizeof(int), 1, pFile);
						else
							fwrite(&synapseCur->nInputArrayIndex, sizeof(int), 1, pFile);

						fwrite(&synapseCur->nInputCount, sizeof(int), 1, pFile);
						
						for(i=0; i< synapseCur->nInputCount; ++i)
							fwrite(&synapseCur->nInputArray[i], sizeof(int), 1, pFile);
					}
				}
			}
		}
	}

	fclose(pFile);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void Read_DNX_Network(structNetwork *networkMain, char *sFilePath)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	FILE				*pFile;
	structCLN			*clnNew = NULL;
	structLayer			*layerNew = NULL;
	structPerceptron	*perceptronHeadCur = NULL;
	structPerceptron	*perceptronHeadNew = NULL;
	structPerceptron	*perceptronNewHead = NULL;
	structPerceptron	*perceptronNew = NULL;
	structPerceptron	*perceptronCur = NULL;
	structPerceptron	*perceptronDuplicate = NULL;
	structSynapse		*synapseNew = NULL;
	structSynapse		*synapseDuplicate = NULL;
	float				fValue;
	int					nCLNCount;
	int					nLayerCount;
	int					nHeadPerceptronCount;
	int					nPerceptronCount;
	int					nSynapseCount;
	int					nPlaceHolder;
	int					nIndex;
	int					nSynapseIndex;
	int					nMaxSynapseIndex;

	int					i, j, k, m, n;
	
  char cwd[1024];
  if (getcwd(cwd, sizeof(cwd)) != NULL) {
      printf("Current working dir: %s\n", cwd);
  } else {
      perror("getcwd() error");
  }
	if ((pFile = FOpenMakeDirectory(sFilePath, "rb")) == NULL)
	{
		printf("Could not read file -- %s\n\n", sFilePath);
		while (1);
	}


	fread(&networkMain->nRowCount, sizeof(int), 1, pFile);
	fread(&networkMain->nColumnCount, sizeof(int), 1, pFile);

	if (networkMain->fInputArray != NULL)
		free(networkMain->fInputArray);

	if ((networkMain->fInputArray = (float *)calloc(networkMain->nRowCount * networkMain->nColumnCount, sizeof(float))) == NULL)
		exit(0);



	fread(&nCLNCount, sizeof(int), 1, pFile);

	for (i = 0; i < nCLNCount; ++i)
	{
		if ((clnNew = (structCLN *)calloc(1, sizeof(structCLN))) == NULL)
			exit(0);

		fread(&clnNew->nID, sizeof(int), 1, pFile);
		fread(&clnNew->fLearningRate, sizeof(float), 1, pFile);
		fread(&clnNew->fInitialError, sizeof(float), 1, pFile);
		fread(&clnNew->fThreshold, sizeof(float), 1, pFile);
		fread(&clnNew->nLabelID, sizeof(int), 1, pFile);
		fread(&clnNew->nSize, sizeof(int), 1, pFile);
		fread(&clnNew->nPerceptronLayerCount, sizeof(int), 1, pFile);

		fread(&nLayerCount, sizeof(int), 1, pFile);
		for (j = 0; j < nLayerCount; ++j)
		{
			if ((layerNew = (structLayer *)calloc(1, sizeof(structLayer))) == NULL)
				exit(0);

			fread(&layerNew->nID, sizeof(int), 1, pFile);
			fread(&layerNew->nLayerType, sizeof(int), 1, pFile);
			fread(&layerNew->nKernelCount, sizeof(int), 1, pFile);
			fread(&layerNew->nInputRowCount, sizeof(int), 1, pFile);
			fread(&layerNew->nInputColumnCount, sizeof(int), 1, pFile);
			fread(&layerNew->nKernelRowCount, sizeof(int), 1, pFile);
			fread(&layerNew->nKernelColumnCount, sizeof(int), 1, pFile);
			fread(&layerNew->nStrideRow, sizeof(int), 1, pFile);
			fread(&layerNew->nStrideColumn, sizeof(int), 1, pFile);
			fread(&layerNew->nWeightCount, sizeof(int), 1, pFile);
			fread(&layerNew->nOutputArraySize, sizeof(int), 1, pFile);
			fread(&layerNew->nOffset, sizeof(int), 1, pFile);
			fread(&layerNew->nPerceptronCount, sizeof(int), 1, pFile);
			fread(&layerNew->nConnectionCount, sizeof(int), 1, pFile);
			fread(&layerNew->fGamma, sizeof(float), 1, pFile);
			fread(&layerNew->fLambda, sizeof(float), 1, pFile);

			if(layerNew->nWeightCount > 0)
				layerNew->fWeightArray = (float *)calloc(300000, sizeof(float));  // layerNew->nWeightCount
			
			nSynapseIndex = 0;
			nMaxSynapseIndex = 0;

			AddNew_Layer(&clnNew->layerHead, layerNew);

			fread(&nHeadPerceptronCount, sizeof(int), 1, pFile);
			for (k = 0; k < nHeadPerceptronCount; ++k)
			{
				perceptronNewHead = NULL;

				fread(&nPerceptronCount, sizeof(int), 1, pFile);
				for (m = 0; m < nPerceptronCount; ++m)
				{
					perceptronNew = (structPerceptron *)calloc(1, sizeof(structPerceptron));

					fread(&perceptronNew->nLayerType, sizeof(int), 1, pFile);
					fread(&perceptronNew->nID, sizeof(int), 1, pFile);
					fread(&perceptronNew->nHeadIndex, sizeof(int), 1, pFile);
					fread(&perceptronNew->nIndex, sizeof(int), 1, pFile);
					fread(&perceptronNew->fOutput, sizeof(float), 1, pFile);
					fread(&perceptronNew->nDimX, sizeof(int), 1, pFile);
					fread(&perceptronNew->nDimY, sizeof(int), 1, pFile);
					fread(&perceptronNew->nConnectionCount, sizeof(int), 1, pFile);
					fread(&perceptronNew->nWeightCount, sizeof(int), 1, pFile);
					fread(&perceptronNew->fBias, sizeof(float), 1, pFile);
					fread(&perceptronNew->fError, sizeof(float), 1, pFile);
					fread(&perceptronNew->fDifferential, sizeof(float), 1, pFile);
					fread(&perceptronNew->fLearningRate, sizeof(float), 1, pFile);
					fread(&perceptronNew->fThreshold, sizeof(float), 1, pFile);
					fread(&perceptronNew->nLayerID, sizeof(int), 1, pFile);
					fread(&perceptronNew->nClusterCount, sizeof(int), 1, pFile);

					fread(&nSynapseCount, sizeof(int), 1, pFile);

					for (n = 0; n < nSynapseCount; ++n)
					{
						synapseNew = (structSynapse *)calloc(1, sizeof(structSynapse));

						fread(&synapseNew->nID, sizeof(int), 1, pFile);
						fread(&synapseNew->nIndex, sizeof(int), 1, pFile);

						if (layerNew->nLayerType == FULLY_CONNECTED_LAYER || layerNew->nLayerType == CLASSIFIER_LAYER)
							synapseNew->nIndex = nSynapseIndex++;

						if (synapseNew->nIndex > layerNew->nWeightCount)
							nMaxSynapseIndex = synapseNew->nIndex;

						fread(&synapseNew->nCluster, sizeof(int), 1, pFile);
						fread(&synapseNew->bAdjust, sizeof(int), 1, pFile);
						fread(&fValue, sizeof(float), 1, pFile);
						
						if (layerNew->nWeightCount)
						{
							layerNew->fWeightArray[synapseNew->nIndex] = fValue;
							synapseNew->fWeight = &layerNew->fWeightArray[synapseNew->nIndex];
						}
						
						fread(&synapseNew->nInputArrayIndex, sizeof(int), 1, pFile);


						if (synapseNew->nInputArrayIndex == -1) // Bias
						{
							fread(&nPlaceHolder, sizeof(int), 1, pFile);
							synapseNew->fInput = (float *)calloc(1, sizeof(float));
							*(synapseNew->fInput) = 1;
						}
						else if (synapseNew->nInputArrayIndex != -999) // Connect to input array
						{
							fread(&synapseNew->nInputArrayIndex, sizeof(int), 1, pFile);
							synapseNew->fInput = &networkMain->fInputArray[synapseNew->nInputArrayIndex];
						}
						else
						{
							fread(&nPlaceHolder, sizeof(int), 1, pFile);

							for (perceptronHeadCur = layerNew->prev->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
							{
								for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
								{
									if (perceptronCur->nID == nPlaceHolder)
									{
										synapseNew->fInput = &perceptronCur->fOutput;
										synapseNew->perceptronConnectTo = perceptronCur;

										break;
									}
								}

								if (perceptronCur != NULL)
									break;
								else
									nIndex = 0;
							}

							if (perceptronHeadCur == NULL)
							{
								nIndex = 0;
							}
						}


						fread(&synapseNew->nInputCount, sizeof(int), 1, pFile);

						if (synapseNew->nInputCount > 0)
						{
							synapseNew->nInputArray = (int *)calloc(synapseNew->nInputCount, sizeof(int));
							synapseNew->fInputArray = (float **)calloc(synapseNew->nInputCount, sizeof(float *));

							for (i = 0; i < synapseNew->nInputCount; ++i)
							{
								fread(&synapseNew->nInputArray[i], sizeof(int), 1, pFile);

								if (synapseNew->nInputArrayIndex != -999) // Connect to input array
								{
									synapseNew->fInputArray[i] = &networkMain->fInputArray[synapseNew->nInputArray[i]];
								}
								else
								{
									for (perceptronHeadCur = layerNew->prev->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
									{
										for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
										{
											if (perceptronCur->nID == synapseNew->nInputArray[i])
											{
												synapseNew->fInputArray[i] = &perceptronCur->fOutput;
												break;
											}
										}

										if (perceptronCur != NULL)
											break;
										else
											nIndex = 0;
									}
								}
							}
						}

						AddNew_Synapse(&perceptronNew->synapseHead, synapseNew);
					}

					AddNew_Perceptron(&perceptronNewHead, perceptronNew);
				}

				AddToLayerV2_Perceptron(&layerNew->perceptronHead, perceptronNewHead);
			}


			if (layerNew->nLayerType == CLASSIFIER_LAYER)
			{
				clnNew->perceptronClassifier = layerNew->perceptronHead;
				clnNew->layerClassifier = layerNew;
			}
			else if (layerNew->nLayerType == CONV_2D_LAYER || layerNew->nLayerType == CONV_3D_LAYER)
			{
				for (perceptronHeadNew = layerNew->perceptronHead; perceptronHeadNew != NULL; perceptronHeadNew = perceptronHeadNew->nextHead)
				{
					for (perceptronDuplicate = perceptronHeadNew->next, perceptronCur = perceptronHeadNew; perceptronDuplicate != NULL; perceptronDuplicate = perceptronDuplicate->next, perceptronCur = perceptronCur->next)
					{
						perceptronDuplicate->nClusterCount = perceptronCur->nClusterCount;
					}
				}



				for (perceptronHeadNew = layerNew->perceptronHead; perceptronHeadNew != NULL; perceptronHeadNew = perceptronHeadNew->nextHead)
				{
					for (perceptronDuplicate = perceptronHeadNew->next; perceptronDuplicate != NULL; perceptronDuplicate = perceptronDuplicate->next)
					{
						for (synapseNew = perceptronHeadNew->synapseHead->next, synapseDuplicate = perceptronDuplicate->synapseHead->next; synapseNew != NULL; synapseNew = synapseNew->next, synapseDuplicate = synapseDuplicate->next)
						{
							synapseDuplicate->nCluster = synapseNew->nCluster;
						}
					}
				}

			}
		}

		AddNew_ClassLevelNetworks(&networkMain->clnHead, clnNew);
	}

	if (fclose(pFile)) 
	{ 
		printf("error closing file."); 
		exit(-1); 
	}

	networkMain->clnCur = networkMain->clnHead;

	return;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void Read_Network(structNetwork *networkMain, char *sFilePath)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	FILE				*pFile;
	structCLN			*clnNew = NULL;
	structLayer			*layerNew = NULL;
	structPerceptron	*perceptronHeadCur = NULL;
	structPerceptron	*perceptronHeadNew = NULL;
	structPerceptron	*perceptronNewHead = NULL;
	structPerceptron	*perceptronNew = NULL;
	structPerceptron	*perceptronCur = NULL;
	structPerceptron	*perceptronDuplicate = NULL;
	structSynapse		*synapseNew = NULL;
	structSynapse		*synapseDuplicate = NULL;
	float				fValue;
	int					nCLNCount;
	int					nLayerCount;
	int					nHeadPerceptronCount;
	int					nPerceptronCount;
	int					nSynapseCount;
	int					nPlaceHolder;
	int					nIndex;
	int					nSynapseIndex;
	int					nMaxSynapseIndex;
	int					nCluster=0;
	int					nWeightCount =0;
	int					i, j, k, m, n;

  printf("Attempting to open file: %s\n",sFilePath);

	if ((pFile = FOpenMakeDirectory(sFilePath, "rb")) == NULL)
	{
		printf("Could not read file -- %s\n\n", sFilePath);
		while (1);
	}

	fread(&networkMain->nRowCount, sizeof(int), 1, pFile);
	fread(&networkMain->nColumnCount, sizeof(int), 1, pFile);

	if (networkMain->fInputArray != NULL)
		free(networkMain->fInputArray);

	if ((networkMain->fInputArray = (float *)calloc(networkMain->nRowCount * networkMain->nColumnCount, sizeof(float))) == NULL)
		exit(0);

	fread(&nCLNCount, sizeof(int), 1, pFile);

	for (i = 0; i<nCLNCount; ++i)
	{
		if ((clnNew = (structCLN *)calloc(1, sizeof(structCLN))) == NULL)
			exit(0);

		fread(&clnNew->nID, sizeof(int), 1, pFile);
		fread(&clnNew->fLearningRate, sizeof(float), 1, pFile);
		fread(&clnNew->fInitialError, sizeof(float), 1, pFile);
		fread(&clnNew->fThreshold, sizeof(float), 1, pFile);
		fread(&clnNew->nLabelID, sizeof(int), 1, pFile);
		fread(&clnNew->nSize, sizeof(int), 1, pFile);
		fread(&clnNew->nPerceptronLayerCount, sizeof(int), 1, pFile);

		fread(&nLayerCount, sizeof(int), 1, pFile);
		for (j = 0; j<nLayerCount; ++j)
		{
			if ((layerNew = (structLayer *)calloc(1, sizeof(structLayer))) == NULL)
				exit(0);

			fread(&layerNew->nID, sizeof(int), 1, pFile);
			fread(&layerNew->nLayerType, sizeof(int), 1, pFile);
			fread(&layerNew->nKernelCount, sizeof(int), 1, pFile);
			fread(&layerNew->nInputRowCount, sizeof(int), 1, pFile);
			fread(&layerNew->nInputColumnCount, sizeof(int), 1, pFile);
			fread(&layerNew->nKernelRowCount, sizeof(int), 1, pFile);
			fread(&layerNew->nKernelColumnCount, sizeof(int), 1, pFile);
			fread(&layerNew->nStrideRow, sizeof(int), 1, pFile);
			fread(&layerNew->nStrideColumn, sizeof(int), 1, pFile);
			fread(&layerNew->nWeightCount, sizeof(int), 1, pFile);
			fread(&layerNew->nOutputArraySize, sizeof(int), 1, pFile);
			fread(&layerNew->nOffset, sizeof(int), 1, pFile);
			fread(&layerNew->nPerceptronCount, sizeof(int), 1, pFile);
			fread(&layerNew->nConnectionCount, sizeof(int), 1, pFile);
			fread(&layerNew->fGamma, sizeof(float), 1, pFile);
			fread(&layerNew->fLambda, sizeof(float), 1, pFile);

			if (layerNew->nLayerType == CONV_2D_LAYER || layerNew->nLayerType == CONV_3D_LAYER)
			{
				nWeightCount = (layerNew->nWeightCount + (layerNew->nWeightCount / (layerNew->nKernelRowCount * layerNew->nKernelColumnCount)));
			}
			else if (layerNew->nLayerType == FULLY_CONNECTED_LAYER || layerNew->nLayerType == CLASSIFIER_LAYER)
			{
				nWeightCount = layerNew->nWeightCount + layerNew->nPerceptronCount;
			}
			else
			{

			}

			layerNew->fWeightArray = (float*)calloc(nWeightCount, sizeof(float));  // layerNew->nWeightCount

			
			
			
			nSynapseIndex = 0;
			nMaxSynapseIndex = 0;

			AddNew_Layer(&clnNew->layerHead, layerNew);


			fread(&nHeadPerceptronCount, sizeof(int), 1, pFile);
			for (k = 0; k < nHeadPerceptronCount; ++k)
			{
				perceptronNewHead = NULL;

				fread(&nPerceptronCount, sizeof(int), 1, pFile);
				for (m = 0; m<nPerceptronCount; ++m)
				{
					perceptronNew = (structPerceptron *)calloc(1, sizeof(structPerceptron));

					fread(&perceptronNew->nLayerType, sizeof(int), 1, pFile);
					fread(&perceptronNew->nID, sizeof(int), 1, pFile);
					fread(&perceptronNew->nHeadIndex, sizeof(int), 1, pFile);
					fread(&perceptronNew->nIndex, sizeof(int), 1, pFile);
					fread(&perceptronNew->fOutput, sizeof(float), 1, pFile);
					fread(&perceptronNew->nDimX, sizeof(int), 1, pFile);
					fread(&perceptronNew->nDimY, sizeof(int), 1, pFile);
					fread(&perceptronNew->nConnectionCount, sizeof(int), 1, pFile);
					fread(&perceptronNew->nWeightCount, sizeof(int), 1, pFile);
					fread(&perceptronNew->fBias, sizeof(float), 1, pFile);
					fread(&perceptronNew->fError, sizeof(float), 1, pFile);
					fread(&perceptronNew->fDifferential, sizeof(float), 1, pFile);
					fread(&perceptronNew->fLearningRate, sizeof(float), 1, pFile);
					fread(&perceptronNew->fThreshold, sizeof(float), 1, pFile);
					fread(&perceptronNew->nLayerID, sizeof(int), 1, pFile);
					fread(&perceptronNew->nClusterCount, sizeof(int), 1, pFile);

					fread(&nSynapseCount, sizeof(int), 1, pFile);

					for (n = 0; n < nSynapseCount; ++n)
					{
						synapseNew = (structSynapse *)calloc(1, sizeof(structSynapse));
							
						fread(&synapseNew->nID, sizeof(int), 1, pFile);
						fread(&synapseNew->nIndex, sizeof(int), 1, pFile);

						if(layerNew->nLayerType == FULLY_CONNECTED_LAYER || layerNew->nLayerType == CLASSIFIER_LAYER)
							synapseNew->nIndex = nSynapseIndex++;
						
						if (synapseNew->nIndex > nWeightCount)
							nMaxSynapseIndex = synapseNew->nIndex;
						
						fread(&synapseNew->nCluster, sizeof(int), 1, pFile);
						synapseNew->nCluster = nCluster++;

						fread(&synapseNew->bAdjust, sizeof(int), 1, pFile);
						fread(&fValue, sizeof(float), 1, pFile);
						
						if (layerNew->nLayerType != MAX_POOLING_LAYER)
							layerNew->fWeightArray[synapseNew->nIndex] = fValue;
						
						
						fread(&synapseNew->nInputArrayIndex, sizeof(int), 1, pFile);

						synapseNew->fWeight = &layerNew->fWeightArray[synapseNew->nIndex];

						if (synapseNew->nInputArrayIndex == -1) // Bias
						{
							fread(&nPlaceHolder, sizeof(int), 1, pFile);
							synapseNew->fInput = (float *)calloc(1, sizeof(float));
							*(synapseNew->fInput) = 1;
						}
						else if (synapseNew->nInputArrayIndex != -999) // Connect to input array
						{
							fread(&synapseNew->nInputArrayIndex, sizeof(int), 1, pFile);
							synapseNew->fInput = &networkMain->fInputArray[synapseNew->nInputArrayIndex];
						}
						else
						{
							fread(&nPlaceHolder, sizeof(int), 1, pFile);

							for (perceptronHeadCur = layerNew->prev->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
							{
								for (perceptronCur= perceptronHeadCur; perceptronCur!=NULL; perceptronCur= perceptronCur->next)
								{
									if (perceptronCur->nID == nPlaceHolder)
									{
										synapseNew->fInput = &perceptronCur->fOutput;
										synapseNew->perceptronConnectTo = perceptronCur;

										break;
									}
								}

								if (perceptronCur != NULL)
									break;
								else
									nIndex = 0;
							}

							if (perceptronHeadCur == NULL)
							{
								nIndex = 0;
							}
						}

						AddNew_Synapse(&perceptronNew->synapseHead, synapseNew);
					}

					AddNew_Perceptron(&perceptronNewHead, perceptronNew);
				}

				AddToLayerV2_Perceptron(&layerNew->perceptronHead, perceptronNewHead);
			}		

			if (layerNew->nLayerType == CLASSIFIER_LAYER)
			{
				clnNew->perceptronClassifier = layerNew->perceptronHead;
				clnNew->layerClassifier = layerNew;
			}
			else if (layerNew->nLayerType == CONV_2D_LAYER || layerNew->nLayerType == CONV_3D_LAYER)
			{
				for (perceptronHeadNew = layerNew->perceptronHead; perceptronHeadNew != NULL; perceptronHeadNew = perceptronHeadNew->nextHead)
				{
					for (perceptronDuplicate = perceptronHeadNew->next, perceptronCur= perceptronHeadNew; perceptronDuplicate != NULL; perceptronDuplicate = perceptronDuplicate->next, perceptronCur= perceptronCur->next)
					{
						perceptronDuplicate->nClusterCount = perceptronCur->nClusterCount;
					}
				}

				
				
				for (perceptronHeadNew = layerNew->perceptronHead; perceptronHeadNew != NULL; perceptronHeadNew = perceptronHeadNew->nextHead)
				{
					for (perceptronDuplicate = perceptronHeadNew->next; perceptronDuplicate != NULL; perceptronDuplicate = perceptronDuplicate->next)
					{
						for (synapseNew = perceptronHeadNew->synapseHead->next, synapseDuplicate = perceptronDuplicate->synapseHead->next; synapseNew != NULL; synapseNew = synapseNew->next, synapseDuplicate = synapseDuplicate->next)
						{
							synapseDuplicate->nCluster = synapseNew->nCluster;
						}
					}
				}

			}
		}

		AddNew_ClassLevelNetworks(&networkMain->clnHead, clnNew);
	}


	networkMain->clnCur = networkMain->clnHead;

	fclose(pFile);

	return;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void CopyWeights(structCLN *clnHead, int nMode)
{
	structCLN			*clnCur = NULL;
	structLayer			*layerCur = NULL;
	structPerceptron	*perceptronHeadCur = NULL;
	structPerceptron	*perceptronHead = NULL;
	structPerceptron	*perceptronCur;
	structSynapse		*synapseCur;

	//char fileName[12] = "CLN2_";
	//char fileNumber[20];

	//sprintf(fileNumber, "%d", counter);
	//strcat(fileName, fileNumber);
	//strcat(fileName, ".txt");

	//FILE *tempTest = fopen(fileName, "w");
	//printf("%s\n", fileName);

	for (clnCur = clnHead; clnCur != NULL; clnCur = clnCur->next)
	{
		for (layerCur = clnCur->layerHead; layerCur != NULL; layerCur = layerCur->next)
		{
			if (layerCur->nLayerType != MAX_POOL_LAYER)
			{
				for (perceptronHeadCur = layerCur->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
				{
					for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
					{
						for (synapseCur = perceptronCur->synapseHead; synapseCur != NULL; synapseCur = synapseCur->next)
						{
							if (nMode == NETWORK_TO_MEMORY){
								synapseCur->fTempWeight = *synapseCur->fWeight;
								//fprintf(tempTest,"Layer %d, Perceptron %d, Synapse %d | Weight is: %lf\n", layerCur-> nID, perceptronCur-> nID, synapseCur-> nID, synapseCur->fTempWeight);
							}
							else
								*synapseCur->fWeight = synapseCur->fTempWeight;
						}
					}
				}
			}
		}
	}

	//counter++;
	//fclose(tempTest);
	return;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void PrintHeader_Network(structNetwork *networkMain, FILE *fpFileOut)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	int		nLength = 0;
	int		nMaxLength = 0;
	int		i, j;

	for (i = 0; i<networkMain->nParameterCount; ++i)
	{
		if (strlen(networkMain->parameterData[i].sParameter) > nMaxLength)
			nMaxLength = (int)strlen(networkMain->parameterData[i].sParameter);
	}

	printf("\n%s\n", networkMain->sTitle);

	if (strcmp(networkMain->sConfigFilePath, "NULL"))
	{
		nLength = 11;

		printf("Config File");
		for (j = 0; j<nMaxLength - nLength; ++j)
			printf(" ");

		printf(". %s\n", networkMain->sConfigFilePath);
	}

	for (i = 0; i<networkMain->nParameterCount; ++i)
	{
		nLength = (int)strlen(networkMain->parameterData[i].sParameter);

		printf("%s", networkMain->parameterData[i].sParameter);

		for (j = 0; j<nMaxLength - nLength; ++j)
		{
			if (!(i % 2))
				printf(" ");
			else
				printf(" ");
		}

		printf(". %s\n", networkMain->parameterData[i].sValue);
	}

/////////////////////////////////////////////////////////////////////////////////////////////////

	if (fpFileOut != NULL)
	{
		fprintf(fpFileOut, "%s\n", networkMain->sTitle);

		if (strcmp(networkMain->sConfigFilePath, "NULL"))
		{
			fprintf(fpFileOut, "Config File\t%s\n", networkMain->sConfigFilePath);
		}

		for (i = 0; i<networkMain->nParameterCount; ++i)
		{
			fprintf(fpFileOut, "%s\t%s\n", networkMain->parameterData[i].sParameter, networkMain->parameterData[i].sValue);
		}
	}
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void DisplayInputData(structNetwork *networkMain, FILE *fpFileOut)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	FILE			*fpOut = NULL;
	structClass		*classCur;
	int				nTotalPerceptrons = 0;
	int				nTotalSynapes = 0;
	int				nTotalWeights = 0;
	int				nTrainingSum;
	int				nVerifySum;
	int				nTestingSum;


	if (fpFileOut == NULL)
		fpOut = stdout;
	else
		fpOut = fpFileOut;

	fprintf(fpOut, "Input Data                                                                                                        \n\t");
	fprintf(fpOut, "\n      Class:");
	for (classCur = networkMain->classHead; classCur != NULL; classCur = classCur->next)
		fprintf(fpOut, "\t%s", classCur->sLabel);

	fprintf(fpOut, "\n   Training:");
	for (classCur = networkMain->classHead, nTrainingSum = 0; classCur != NULL; classCur = classCur->next)
	{
		fprintf(fpOut, "\t%d", classCur->nTrainingCount);
		nTrainingSum += classCur->nTrainingCount;
	}
	fprintf(fpOut, "\t%d", nTrainingSum);

	fprintf(fpOut, "\n     Verify:");
	for (classCur = networkMain->classHead, nVerifySum = 0; classCur != NULL; classCur = classCur->next)
	{
		fprintf(fpOut, "\t%d", classCur->nVerifyCount);
		nVerifySum += classCur->nVerifyCount;
	}
	fprintf(fpOut, "\t%d", nVerifySum);

	fprintf(fpOut, "\n    Testing:");
	for (classCur = networkMain->classHead, nTestingSum = 0; classCur != NULL; classCur = classCur->next)
	{
		fprintf(fpOut, "\t%d", classCur->nTestingCount);
		nTestingSum += classCur->nTestingCount;
	}
	fprintf(fpOut, "\t%d", nTestingSum);

	fprintf(fpOut, "\n\t\t");
	for (classCur = networkMain->classHead; classCur != NULL; classCur = classCur->next)
		fprintf(fpOut, "\t");

	fprintf(fpOut, "%d\n", (nTrainingSum + nVerifySum + nTestingSum));
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void WriteNetwork(structCLN *clnHead, char *sFilePath)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	FILE		*pFile = NULL;
	structCLN	*cln = NULL;
	int			nCount = 0;

	if ((pFile = fopen(sFilePath, "wb")) == NULL)
	{
		char	sMessage[256];

		sprintf(sMessage, "File Error: WriteNetwork() - Cannot Write To: %s", sFilePath);
		DisplayMessage(sMessage, PAUSE);
	}

	for (cln = clnHead, nCount = 0; cln != NULL; cln = cln->next, ++nCount);

	fwrite(&nCount, sizeof(int), 1, pFile);

	for (cln = clnHead; cln != NULL; cln = cln->next)
	{
		WriteClassLevelNetwork(cln, NULL, pFile);
	}

	fclose(pFile);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int ReadNetwork(structNetwork *network, char *sFilePath)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	FILE		*pFile = NULL;
	structCLN	*cln = NULL;
	structCLN	*clnA = NULL;
	int			nCount;
	int			i;

	if ((pFile = fopen(sFilePath, "rb")) == NULL)
	{
		char	sMessage[256];

		sprintf(sMessage, "File Error: ReadNetwork() - Cannot Read From: %s", sFilePath);
		DisplayMessage(sMessage, PAUSE);
	}

	fread(&nCount, sizeof(int), 1, pFile);

	for (i = 0; i < nCount; ++i)
	{
		cln = ReadClassLevelNetwork(network->fxptInputArray, &network->nID, network->sInputNetworkPath, pFile);
		cln->nClassifierMode = network->nClassifierMode;
		DescribeClassLevelNetwork(cln, NULL);
		AddNew_ClassLevelNetworks(&network->clnHead, cln);
	}

	fclose(pFile);

	return(nCount);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void WriteV2_Network(structNetwork *networkMain, char *sFilePath)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	FILE				*pFile;
	structCLN			*clnCur;
	structLayer			*layerCur = NULL;
	structPerceptron	*perceptronHead = NULL;
	structPerceptron	*perceptronCur;
	structSynapse		*synapseCur;
	int					nCount;
	int					nBiasIndex = -1;

	if ((pFile = FOpenMakeDirectory(sFilePath, "wb")) == NULL)
	{
		printf("Could not write file -- %s\n\n", sFilePath);
		while (1);
	}

	fwrite(&networkMain->nDataSource, sizeof(int), 1, pFile);
	fwrite(&networkMain->nRowCount, sizeof(int), 1, pFile);
	fwrite(&networkMain->nColumnCount, sizeof(int), 1, pFile);

	for (clnCur = networkMain->clnHead, nCount = 0; clnCur != NULL; clnCur = clnCur->next, ++nCount);
	fwrite(&nCount, sizeof(int), 1, pFile);

	for (clnCur = networkMain->clnHead; clnCur != NULL; clnCur = clnCur->next)
	{
		fwrite(&clnCur->nLabelID, sizeof(int), 1, pFile);
		fwrite(&clnCur->nSize, sizeof(int), 1, pFile);
		fwrite(&clnCur->nPerceptronLayerCount, sizeof(int), 1, pFile);
		fwrite(&clnCur->fLearningRate, sizeof(float), 1, pFile);
		fwrite(&clnCur->fInitialError, sizeof(float), 1, pFile);
		fwrite(&clnCur->fThreshold, sizeof(float), 1, pFile);

		for (layerCur = clnCur->layerHead, nCount = 0; layerCur != NULL; layerCur = layerCur->next, ++nCount);
		fwrite(&nCount, sizeof(int), 1, pFile);

		for (layerCur = clnCur->layerHead; layerCur != NULL; layerCur = layerCur->next)
		{
			fwrite(&layerCur->nID, sizeof(int), 1, pFile);
			fwrite(&layerCur->nLayerType, sizeof(int), 1, pFile);
			fwrite(&layerCur->nKernelCount, sizeof(int), 1, pFile);
			fwrite(&layerCur->nInputRowCount, sizeof(int), 1, pFile);
			fwrite(&layerCur->nInputColumnCount, sizeof(int), 1, pFile);
			fwrite(&layerCur->nKernelRowCount, sizeof(int), 1, pFile);
			fwrite(&layerCur->nKernelColumnCount, sizeof(int), 1, pFile);
			fwrite(&layerCur->nStrideRow, sizeof(int), 1, pFile);
			fwrite(&layerCur->nStrideColumn, sizeof(int), 1, pFile);
			fwrite(&layerCur->nWeightCount, sizeof(int), 1, pFile);
			fwrite(&layerCur->nOutputArraySize, sizeof(int), 1, pFile);
			fwrite(&layerCur->nOffset, sizeof(int), 1, pFile);
			fwrite(&layerCur->nPerceptronCount, sizeof(int), 1, pFile);
			fwrite(&layerCur->nConnectionCount, sizeof(int), 1, pFile);
			fwrite(&layerCur->fGamma, sizeof(float), 1, pFile);
			fwrite(&layerCur->fLambda, sizeof(float), 1, pFile);

			if (layerCur->nLayerType == INPUT_LAYER)
			{
				for (perceptronCur = layerCur->perceptronHead; perceptronCur != NULL; perceptronCur = perceptronCur->next)
				{
					fwrite(&perceptronCur->nLayerType, sizeof(int), 1, pFile);
					fwrite(&perceptronCur->nHeadIndex, sizeof(int), 1, pFile);
					fwrite(&perceptronCur->nID, sizeof(int), 1, pFile);
					fwrite(&perceptronCur->nIndex, sizeof(int), 1, pFile);
					fwrite(&perceptronCur->nWeightCount, sizeof(int), 1, pFile);
					fwrite(&perceptronCur->fLearningRate, sizeof(float), 1, pFile);
					fwrite(&perceptronCur->fThreshold, sizeof(float), 1, pFile);

					fwrite(&perceptronCur->synapseHead->nIndex, sizeof(int), 1, pFile);
					fwrite(&*perceptronCur->synapseHead->fWeight, sizeof(float), 1, pFile);
				}
			}
			else if (layerCur->nLayerType == FULLY_CONNECTED_LAYER || layerCur->nLayerType == SPARSELY_CONNECTED_LAYER || layerCur->nLayerType == CLASSIFIER_LAYER)
			{
				for (perceptronCur = layerCur->perceptronHead; perceptronCur != NULL; perceptronCur = perceptronCur->next)
				{
					fwrite(&perceptronCur->nLayerType, sizeof(int), 1, pFile);
					fwrite(&perceptronCur->nHeadIndex, sizeof(int), 1, pFile);
					fwrite(&perceptronCur->nID, sizeof(int), 1, pFile);
					fwrite(&perceptronCur->nIndex, sizeof(int), 1, pFile);
					fwrite(&perceptronCur->nWeightCount, sizeof(int), 1, pFile);
					fwrite(&perceptronCur->fLearningRate, sizeof(float), 1, pFile);
					fwrite(&perceptronCur->fThreshold, sizeof(float), 1, pFile);

					for (synapseCur = perceptronCur->synapseHead, nCount = 0; synapseCur != NULL; synapseCur = synapseCur->next, ++nCount);
					fwrite(&nCount, sizeof(int), 1, pFile);

					for (synapseCur = perceptronCur->synapseHead; synapseCur != NULL; synapseCur = synapseCur->next)
					{
						if (synapseCur->perceptronConnectTo == NULL)
						{
							fwrite(&nBiasIndex, sizeof(int), 1, pFile);
							fwrite(&nBiasIndex, sizeof(int), 1, pFile);
						}
						else
						{
							fwrite(&synapseCur->perceptronConnectTo->nHeadIndex, sizeof(int), 1, pFile);
							fwrite(&synapseCur->perceptronConnectTo->nIndex, sizeof(int), 1, pFile);
						}

						fwrite(&synapseCur->nIndex, sizeof(int), 1, pFile);
						fwrite(&*synapseCur->fWeight, sizeof(float), 1, pFile);
					}
				}
			}
			else if (layerCur->nLayerType == SINGLE_CONV_LAYER || layerCur->nLayerType == MULTIPLE_CONV_LAYER)
			{
				for (perceptronCur = layerCur->perceptronHead; perceptronCur != NULL; perceptronCur = perceptronCur->next)
				{
					fwrite(&perceptronCur->nLayerType, sizeof(int), 1, pFile);
					fwrite(&perceptronCur->nHeadIndex, sizeof(int), 1, pFile);
					fwrite(&perceptronCur->nID, sizeof(int), 1, pFile);
					fwrite(&perceptronCur->nIndex, sizeof(int), 1, pFile);
					fwrite(&perceptronCur->nWeightCount, sizeof(int), 1, pFile);
					fwrite(&perceptronCur->fLearningRate, sizeof(float), 1, pFile);
					fwrite(&perceptronCur->fThreshold, sizeof(float), 1, pFile);

					for (synapseCur = perceptronCur->synapseHead, nCount = 0; synapseCur != NULL; synapseCur = synapseCur->next, ++nCount);
					fwrite(&nCount, sizeof(int), 1, pFile);

					for (synapseCur = perceptronCur->synapseHead; synapseCur != NULL; synapseCur = synapseCur->next)
					{
						if (synapseCur->perceptronConnectTo == NULL)
						{
							fwrite(&nBiasIndex, sizeof(int), 1, pFile);
							fwrite(&nBiasIndex, sizeof(int), 1, pFile);
						}
						else
						{
							fwrite(&synapseCur->perceptronConnectTo->nHeadIndex, sizeof(int), 1, pFile);
							fwrite(&synapseCur->perceptronConnectTo->nIndex, sizeof(int), 1, pFile);
						}

						fwrite(&synapseCur->nIndex, sizeof(int), 1, pFile);
						fwrite(&*synapseCur->fWeight, sizeof(float), 1, pFile);
					}
				}
			}
		}
	}

	fclose(pFile);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void ReadV2_Network(structNetwork *networkMain, char *sFilePath)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	FILE				*pFile;
	structCLN			*clnNew = NULL;
	structLayer			*layerNew = NULL;
	structPerceptron	*perceptronHeadCur = NULL;
	structPerceptron	*perceptronHeadNew = NULL;
	structPerceptron	*perceptronNewHead = NULL;
	structPerceptron	*perceptronNew = NULL;
	structPerceptron	*perceptronCur = NULL;
	structSynapse		*synapseNew = NULL;
	int					nCLNCount;
	int					nLayerCount;
	int					nHeadPerceptronCount;
	int					nPerceptronCount;
	int					nSynapseCount;
	int					nPlaceHolder;
	int					nIndex;
	int					nSynapseIndex;

	int					i, j, k, m, n;



	if ((pFile = FOpenMakeDirectory(sFilePath, "rb")) == NULL)
	{
		printf("Could not read file -- %s\n\n", sFilePath);
		while (1);
	}



	fread(&networkMain->nDataSource, sizeof(int), 1, pFile);
	fread(&networkMain->nRowCount, sizeof(int), 1, pFile);
	fread(&networkMain->nColumnCount, sizeof(int), 1, pFile);

	if (networkMain->fInputArray != NULL)
		free(networkMain->fInputArray);

	if ((networkMain->fInputArray = (float *)calloc(networkMain->nRowCount * networkMain->nColumnCount, sizeof(float))) == NULL)
		exit(0);

	fread(&nCLNCount, sizeof(int), 1, pFile);

	for (i = 0; i < nCLNCount; ++i)
	{
		if ((clnNew = (structCLN *)calloc(1, sizeof(structCLN))) == NULL)
			exit(0);

		fread(&clnNew->nLabelID, sizeof(int), 1, pFile);
		fread(&clnNew->nSize, sizeof(int), 1, pFile);
		fread(&clnNew->nPerceptronLayerCount, sizeof(int), 1, pFile);
		fread(&clnNew->fLearningRate, sizeof(float), 1, pFile);
		fread(&clnNew->fInitialError, sizeof(float), 1, pFile);
		fread(&clnNew->fThreshold, sizeof(float), 1, pFile);

		//AddNew_ClassLevelNetworks(&networkMain->clnHead, clnNew);

		fread(&nLayerCount, sizeof(int), 1, pFile);
		for (j = 0; j < nLayerCount; ++j)
		{
			if ((layerNew = (structLayer *)calloc(1, sizeof(structLayer))) == NULL)
				exit(0);

			fread(&layerNew->nID, sizeof(int), 1, pFile);
			fread(&layerNew->nLayerType, sizeof(int), 1, pFile);
			fread(&layerNew->nKernelCount, sizeof(int), 1, pFile);
			fread(&layerNew->nInputRowCount, sizeof(int), 1, pFile);
			fread(&layerNew->nInputColumnCount, sizeof(int), 1, pFile);
			fread(&layerNew->nKernelRowCount, sizeof(int), 1, pFile);
			fread(&layerNew->nKernelColumnCount, sizeof(int), 1, pFile);
			fread(&layerNew->nStrideRow, sizeof(int), 1, pFile);
			fread(&layerNew->nStrideColumn, sizeof(int), 1, pFile);
			fread(&layerNew->nWeightCount, sizeof(int), 1, pFile);
			fread(&layerNew->nOutputArraySize, sizeof(int), 1, pFile);
			fread(&layerNew->nOffset, sizeof(int), 1, pFile);
			fread(&layerNew->nPerceptronCount, sizeof(int), 1, pFile);
			fread(&layerNew->nConnectionCount, sizeof(int), 1, pFile);
			fread(&layerNew->fGamma, sizeof(float), 1, pFile);
			fread(&layerNew->fLambda, sizeof(float), 1, pFile);

			layerNew->fWeightArray = (float *)calloc(layerNew->nWeightCount, sizeof(float));
			nSynapseIndex = 0;

			AddNew_Layer(&clnNew->layerHead, layerNew);


			fread(&nHeadPerceptronCount, sizeof(int), 1, pFile);
			for (k = 0; k < nHeadPerceptronCount; ++k)
			{
				perceptronNewHead = NULL;

				fread(&nPerceptronCount, sizeof(int), 1, pFile);
				for (m = 0; m < nPerceptronCount; ++m)
				{
					perceptronNew = (structPerceptron *)calloc(1, sizeof(structPerceptron));

					fread(&perceptronNew->nLayerType, sizeof(int), 1, pFile);
					fread(&perceptronNew->nID, sizeof(int), 1, pFile);
					fread(&perceptronNew->nHeadIndex, sizeof(int), 1, pFile);
					fread(&perceptronNew->nIndex, sizeof(int), 1, pFile);
					fread(&perceptronNew->fOutput, sizeof(float), 1, pFile);
					fread(&perceptronNew->nDimX, sizeof(int), 1, pFile);
					fread(&perceptronNew->nDimY, sizeof(int), 1, pFile);
					fread(&perceptronNew->nConnectionCount, sizeof(int), 1, pFile);
					fread(&perceptronNew->nWeightCount, sizeof(int), 1, pFile);
					fread(&perceptronNew->fBias, sizeof(float), 1, pFile);
					fread(&perceptronNew->fError, sizeof(float), 1, pFile);
					fread(&perceptronNew->fDifferential, sizeof(float), 1, pFile);
					fread(&perceptronNew->fLearningRate, sizeof(float), 1, pFile);
					fread(&perceptronNew->fThreshold, sizeof(float), 1, pFile);
					fread(&perceptronNew->nLayerID, sizeof(int), 1, pFile);

					fread(&nSynapseCount, sizeof(int), 1, pFile);

					for (n = 0; n < nSynapseCount; ++n)
					{
						synapseNew = (structSynapse *)calloc(1, sizeof(structSynapse));

						fread(&synapseNew->nID, sizeof(int), 1, pFile);
						fread(&synapseNew->nIndex, sizeof(int), 1, pFile);

						if (layerNew->nLayerType == FULLY_CONNECTED_LAYER || layerNew->nLayerType == CLASSIFIER_LAYER)
							synapseNew->nIndex = nSynapseIndex++;

						fread(&synapseNew->nCluster, sizeof(int), 1, pFile);
						fread(&synapseNew->bAdjust, sizeof(int), 1, pFile);
						fread(&layerNew->fWeightArray[synapseNew->nIndex], sizeof(float), 1, pFile);
						fread(&synapseNew->nInputArrayIndex, sizeof(int), 1, pFile);

						synapseNew->fWeight = &layerNew->fWeightArray[synapseNew->nIndex];

						if (synapseNew->nInputArrayIndex == -1) // Bias
						{
							fread(&nPlaceHolder, sizeof(int), 1, pFile);
							synapseNew->fInput = (float *)calloc(1, sizeof(float));
							*(synapseNew->fInput) = 1;
						}
						else if (synapseNew->nInputArrayIndex != -999) // Connect to input array
						{
							fread(&synapseNew->nInputArrayIndex, sizeof(int), 1, pFile);
							synapseNew->fInput = &networkMain->fInputArray[synapseNew->nInputArrayIndex];
						}
						else
						{
							fread(&nPlaceHolder, sizeof(int), 1, pFile);

							for (perceptronHeadCur = layerNew->prev->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
							{
								for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
								{
									if (perceptronCur->nID == nPlaceHolder)
									{
										synapseNew->fInput = &perceptronCur->fOutput;
										synapseNew->perceptronConnectTo = perceptronCur;

										break;
									}
								}

								if (perceptronCur != NULL)
									break;
								else
									nIndex = 0;
							}

							if (perceptronHeadCur == NULL)
							{
								nIndex = 0;
							}
						}

						AddNew_Synapse(&perceptronNew->synapseHead, synapseNew);
					}

					AddNew_Perceptron(&perceptronNewHead, perceptronNew);
				}

				AddToLayerV2_Perceptron(&layerNew->perceptronHead, perceptronNewHead);
			}

			if (layerNew->nLayerType == CLASSIFIER_LAYER)
			{
				clnNew->perceptronClassifier = layerNew->perceptronHead;
				clnNew->layerClassifier = layerNew;
			}
		}

		AddNew_ClassLevelNetworks(&networkMain->clnHead, clnNew);
	}

	networkMain->clnCur = networkMain->clnHead;

	fclose(pFile);

	return;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void SetFilePath(structNetwork *network, char *sFilePath)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	FILE	*pFile = NULL;
	char	sTemp[256];

	if ((pFile = fopen(sFilePath, "rb")) == NULL)
	{
		sprintf(sTemp, "%s/%s", network->sDrive, sFilePath);
		strcpy(sFilePath, sTemp);
	}
	else
	{
		fclose(pFile);
	}

	return;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int DumpWeights(structNetwork *networkMain)
{
	structCLN			*clnCur = NULL;
	structLayer			*layerCur;
	structPerceptron	*perceptronHeadCur = NULL;
	structPerceptron	*perceptronHead = NULL;
	structPerceptron	*perceptronCur;
	structSynapse		*synapseCur;

	FILE				*pFile = NULL;

	if ((pFile = fopen("weights.txt", "wb")) == NULL)
	{
		exit(0);
	}

	for (clnCur = networkMain->clnHead; clnCur != NULL; clnCur = clnCur->next)
	{
		for (layerCur = clnCur->layerHead; layerCur != NULL; layerCur = layerCur->next)
		{
			for (perceptronHeadCur = layerCur->perceptronHead; perceptronHeadCur != NULL; perceptronHeadCur = perceptronHeadCur->nextHead)
			{
				for (perceptronCur = perceptronHeadCur; perceptronCur != NULL; perceptronCur = perceptronCur->next)
				{
					for (synapseCur = perceptronCur->synapseHead; synapseCur != NULL; synapseCur = synapseCur->next)
					{
						if(synapseCur->fWeight != NULL)
							fprintf(pFile, "%f\n", *(synapseCur->fWeight));
						else
							fprintf(pFile, "0.0\n");
					}
				}
			}
		}
	}
	
	fclose(pFile);

	return(0);
}
