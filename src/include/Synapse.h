
	typedef struct structSynapse
	{
		int						nID;
		int						nIndex;
		int						nInputArrayIndex;
		
		int						nCluster;
		int						bAdjust;
		int						nCount;
		float					fAverage;
		float					fSumSquares;
		float					fMax;
		float					fMin;
		
		float					*fInput;
		float					**fInputArray;
		float					*fWeight;
		float					fTempWeight;
		float					fFeedBackWeight;
		fxpt					*fxptWeight;
		fxpt					*fxptInput;

		int						nInputIndex;
		int						nIndexCount;
		int						*nIndexArray;
		
		int						nInputCount;
		int						*nInputArray;

		struct structPerceptron	*perceptronConnectTo;
		struct structPerceptron	*perceptronSource;
		struct structSynapse	*next;

	} structSynapse;	


void				AddNew_Synapse(structSynapse **head, structSynapse *newSynpase);
structSynapse		*Delete_Synapse(structSynapse **head, int nID);
void				DeleteAll_Synapse(structSynapse **head);
void				Free_Synapse(structSynapse **synapseHead);


typedef struct structInputSynapse
{
	int			nID;
	float		*fInput;
	float		fWeight;
	float		fTempWeight;
	float		fMultiplier;
	float		fSum;
	int			nInputCount;
	float		**fInputArray;

	struct structInputSynapse	*next;

} structInputSynapse;