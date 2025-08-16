
#pragma once

typedef struct structClass
{
	int					nID;
	char				*sLabel;
	int					nMode;
	int					nTrainingCount;
	int					nVerifyCount;
	int					nTestingCount;


	struct structClass	*next;
} structClass;


int			AddNew_Classes(structClass **clnHead, char *sLabel);
structClass *DeleteAll_Classes(structClass **classHead);
structClass *DeleteClass_Classes(structClass **classHead, int nID);
