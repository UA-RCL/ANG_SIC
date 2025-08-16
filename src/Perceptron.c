
#include "main.h"

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void AddNew_Perceptron(structPerceptron **head, structPerceptron *newPerceptron)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structPerceptron	*cur = *head;
	int					nID = 0;

	if(newPerceptron->nClusterCount == 0)
		newPerceptron->nClusterCount = 1;
	
	if (*head == NULL)
	{
		if (newPerceptron->nID < 1)
			newPerceptron->nID = 0;

		*head = newPerceptron;
	}
	else
	{
		while (cur->next != NULL)
		{
			cur = cur->next;

			if (newPerceptron->nID < 1)
				nID = cur->nID;
		}

		if (newPerceptron->nID < 1)
			newPerceptron->nID = ++nID;

		cur->next = newPerceptron;
	}

	return;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void AddNewV2_Perceptron(structPerceptron **perceptronHead, structPerceptron *perceptronNew)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structPerceptron *nodeCurrent = *perceptronHead;

	if (perceptronNew->nClusterCount == 0)
		perceptronNew->nClusterCount = 1;

	if (!perceptronNew->nIndex)
	{
		perceptronNew->prev = NULL;
		*perceptronHead = perceptronNew;
	}
	else
	{
		while (nodeCurrent->next != NULL)
			nodeCurrent = nodeCurrent->next;

		nodeCurrent->next = perceptronNew;
		perceptronNew->prev = nodeCurrent;
	}
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void AddToLayer_Perceptron(structPerceptron **perceptronHead, structPerceptron *perceptronNew, int nIndex)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structPerceptron *nodeCurrent=*perceptronHead;

	if(!nIndex)
		*perceptronHead=perceptronNew;
	else
	{
		while(nodeCurrent->nextHead != NULL)
			nodeCurrent = nodeCurrent->nextHead;

		nodeCurrent->nextHead=perceptronNew;
	}
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void AddToLayerV2_Perceptron(structPerceptron **head, structPerceptron *newPerceptron)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structPerceptron	*cur = *head;
	int					nID = 0;

	if (*head == NULL)
	{
		if(newPerceptron->nID == 0)
			newPerceptron->nID = 0;
		*head = newPerceptron;
	}
	else
	{
		while (cur->nextHead != NULL)
		{
			cur = cur->nextHead;
			nID = cur->nID;
		}

		if (newPerceptron->nID == 0)
			newPerceptron->nID = ++nID;

		cur->nextHead = newPerceptron;
		newPerceptron->prev = cur;
	}

	return;
}


/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void Delete_Perceptron(structPerceptron **perceptronHead)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structSynapse	**synapseCur;
	
	if(*perceptronHead)
	{
		while((*perceptronHead)->synapseHead)
		{
			for(synapseCur=&((*perceptronHead)->synapseHead); (*synapseCur)->next != NULL; *synapseCur=(*synapseCur)->next);

			free(*synapseCur);
			*synapseCur=NULL;
		}

		
		if((*perceptronHead)->next)
			Delete_Perceptron(&((*perceptronHead)->next));
		
		if((*perceptronHead)->nextHead)
			Delete_Perceptron(&((*perceptronHead)->nextHead));
		
		free(*perceptronHead);
		
		*perceptronHead=NULL;
	}
}






/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
void DeleteV2_Perceptron(structPerceptron **head)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structPerceptron	*cur = *head;
	structPerceptron	*next = NULL;


	while (cur != NULL)
	{
		DeleteAll_Synapse(&cur->synapseHead);

		if (cur->fWeights != NULL)
			free(cur->fWeights);

		next = cur->next;
		free(cur);
		cur = NULL;

		cur = next;
	}

	*head = NULL;
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int CalculateThreshold_Perceptron(structPerceptron *perceptronClassifierHead, float fThreshold, float *fRatio, int bDisplayMode, FILE *pFile)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structPerceptron	*perceptronCur = NULL;
	int					nWinner = 0;
	int					nCount;
	structSort			sortArray[100];

	// Classifier Layer

	for (perceptronCur = perceptronClassifierHead, nCount = 0; perceptronCur != NULL; perceptronCur = perceptronCur->next, ++nCount)
	{
		sortArray[nCount].nClassID = perceptronCur->nIndex;
		sortArray[nCount].fValue = perceptronCur->fOutput;
	}

	BubbleSort(sortArray, nCount);
	*fRatio = (sortArray[0].fValue + (sortArray[1].fValue * -1.0f));


	if (*fRatio >= fThreshold)
		nWinner = sortArray[0].nClassID;
	else
		nWinner = -1;

	return(nWinner);
}
