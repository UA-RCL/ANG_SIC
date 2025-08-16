#include "main.h"

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
structClass *DeleteAll_Classes(structClass **classHead)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structClass	*clnCur = NULL;

	clnCur = *classHead;
	while (clnCur != NULL)
	{
		clnCur = DeleteClass_Classes(classHead, clnCur->nID);
	}

	return(*classHead);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
structClass *DeleteClass_Classes(structClass **classHead, int nID)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/ 
{
	structClass	*clnCur = NULL;
	structClass	*clnPrev = NULL;

	if ((*classHead)->nID == nID)
	{
		if ((*classHead)->next != NULL)
			clnCur = (*classHead)->next;

		free((*classHead)->sLabel);
		free((*classHead));
		(*classHead) = clnCur;

		return(*classHead);
	}
	else
	{
		for (clnCur = *classHead; clnCur != NULL; clnPrev = clnCur, clnCur = clnCur->next)
		{
			if (clnCur->nID == nID)
			{
				if (clnPrev == NULL)
				{
					*classHead = clnCur->next;
					
					free(clnCur->sLabel);
					free(clnCur);

					return(*classHead);
				}
				else
				{
					clnPrev->next = clnCur->next;
					
					free(clnCur->sLabel);
					free(clnCur);

					return(clnPrev);
				}
			}
		}
	}

	return(NULL);
}

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
int AddNew_Classes(structClass **clnHead, char *sLabel)
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
{
	structClass	*classNew;
	structClass	*nodeCurrent = NULL;
	int			nID = 0;

	if ((classNew = (structClass *)calloc(1, sizeof(structClass))) == NULL)
		exit(0);

	if ((classNew->sLabel = (char *)calloc(32, sizeof(char))) == NULL)
		exit(0);

	strcpy(classNew->sLabel, sLabel);

	if (*clnHead == NULL)
	{
		classNew->nID = 0;
		*clnHead = classNew;
	}
	else
	{
		nodeCurrent = *clnHead;
		nID = nodeCurrent->nID;

		while (nodeCurrent->next != NULL)
		{
			nodeCurrent = nodeCurrent->next;
			nID = nodeCurrent->nID;
		}

		classNew->nID = ++nID;
		nodeCurrent->next = classNew;
	}

	return(classNew->nID);
}

