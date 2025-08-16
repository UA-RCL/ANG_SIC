#include "main.h"

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
structLayer *Create_Layer(structLayer **layerHead, int nLayerType, int nInputRowCount, int nInputColumnCount, int nPerceptronCount)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structLayer *layerNew;
	
	if ((layerNew = (structLayer *)calloc(1, sizeof(structLayer))) == NULL)
	exit(0);

	layerNew->nLayerType = nLayerType;
	layerNew->nInputRowCount = nInputRowCount;
	layerNew->nInputColumnCount = nInputColumnCount;
	layerNew->nPerceptronCount = nPerceptronCount;
	layerNew->nOutputArraySize = layerNew->nPerceptronCount;
	layerNew->nWeightCount = 0;

	AddNew_Layer(layerHead, layerNew);

	return(layerNew);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void AddNew_Layer(structLayer **head, structLayer *newLayer)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structLayer *cur = *head;
	int			nID = 0;

	if (*head == NULL)
	{
		if(newLayer->nID < 0)
			newLayer->nID = 0;
		*head = newLayer;
	}
	else
	{
		while (cur->next != NULL)
		{
			cur = cur->next;
			
			if (newLayer->nID == 0)
				nID = cur->nID;
		}

		if (newLayer->nID == 0)
			newLayer->nID= ++nID;

		cur->next = newLayer;
		newLayer->prev = cur;
	}

	newLayer->nIndex = newLayer->nID;

	return;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
structLayer *Delete_Layer(structLayer **head, int nID)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structLayer	*cur = NULL;
	structLayer	*prev = NULL;

	if ((*head)->nID == nID)
	{
		if ((*head)->next != NULL)
			cur = (*head)->next;

///////////////////////////////////////////////////
		
		free((*head)->fWeightArray);
		
		DeleteV2_Perceptron(&(*head)->perceptronHead);
		free((*head));
///////////////////////////////////////////////////

		(*head) = cur;

		return(*head);
	}
	else
	{
		for (cur = *head; cur != NULL; prev = cur, cur = cur->next)
		{
			if (cur->nID == nID)
			{
				if (prev == NULL)
				{
					*head = cur->next;

///////////////////////////////////////////////////
					free(cur->fWeightArray);
					DeleteV2_Perceptron(&cur->perceptronHead);
					free(cur);
///////////////////////////////////////////////////

					return(*head);
				}
				else
				{
					prev->next = cur->next;

///////////////////////////////////////////////////
					free(cur->fWeightArray);
					DeleteV2_Perceptron(&cur->perceptronHead);
					free(cur);
///////////////////////////////////////////////////
					
					return(prev);
				}
			}
		}
	}

	return(NULL);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void DeleteAll_Layer(structLayer **head)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structLayer	*cur = *head;
	structLayer	*next = NULL;

	while (cur != NULL)
	{
		next = cur->next;
		
		DeleteV2_Perceptron(&cur->perceptronHead);
		free(cur->fWeightArray);
		cur->fWeightArray = NULL;

		free(cur);
		cur = NULL;
		///////////////////////////////////////////////////
		
		cur = next;
	}

	*head = NULL;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void Free_Layer(structPerceptron *perceptronHead)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structSynapse		*synapseCur;
	structSynapse		*synapseTemp;
	structPerceptron	*perceptronCur;
	structPerceptron	*perceptronTemp;

	while (perceptronHead != NULL)
	{
		perceptronCur = perceptronHead->next;

		while (perceptronCur != NULL)
		{
			perceptronTemp = perceptronCur;
			perceptronCur = perceptronCur->next;


			synapseCur = perceptronTemp->synapseHead;
			while (synapseCur != NULL)
			{
				synapseTemp = synapseCur;
				synapseCur = synapseCur->next;

				if (perceptronHead->nLayerType == INPUT_LAYER)
					free(synapseTemp->fWeight);
				else if (!synapseTemp->nID || synapseTemp->perceptronConnectTo == NULL)
				{
					if (perceptronHead->nLayerType != SINGLE_CONV_LAYER && perceptronHead->nLayerType != MULTIPLE_CONV_LAYER)
						free(synapseTemp->fWeight);

					free(synapseTemp->fInput);
				}

				free(synapseTemp);
			}

			free(perceptronTemp);
		}

		synapseCur = perceptronHead->synapseHead;
		while (synapseCur != NULL)
		{
			synapseTemp = synapseCur;
			synapseCur = synapseCur->next;

			if (perceptronHead->nLayerType == INPUT_LAYER)
				free(synapseTemp->fWeight);
			else if (!synapseTemp->nID || synapseTemp->perceptronConnectTo == NULL)
			{
				if (perceptronHead->nLayerType != SINGLE_CONV_LAYER && perceptronHead->nLayerType != MULTIPLE_CONV_LAYER)
					free(synapseTemp->fWeight);

				free(synapseTemp->fInput);
			}

			free(synapseTemp);
		}



		perceptronTemp = perceptronHead;
		perceptronHead = perceptronHead->nextHead;

		free(perceptronTemp);
	}


}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void AddLayer(structLayer **head, structLayer *newLayer)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structLayer *cur = *head;

	if (*head == NULL)
	{
		*head = newLayer;
	}
	else
	{
		while (cur->next != NULL)
		{
			cur = cur->next;
		}

		cur->next = newLayer;
		newLayer->prev = cur;
	}

	return;
}
