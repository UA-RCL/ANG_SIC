

typedef struct structPerceptron
{
	int					nLayerType;
	int					nID;
	int					nHeadIndex;
	int					nIndex;
	structSynapse		*synapseHead;
	float				*fWeights;
	float				fOutput;
	int					nDimX;
	int					nDimY;
	int					*nConnectionArray;
	int					nConnectionCount;
	int					nWeightCount;
	float				fBias;
	float				fError;
	float				fDifferential;
	float				fLearningRate;
	float				fUpperThreshold;
	float				fLowerThreshold;
	float				fThreshold;
	int					nLayerID;
	int					nSynapseCount;
	int					nClusterCount;
	int					nClassID;

	float				fSDLower;
	float				fSDUpper;
	fxpt				fxptOutput;


	struct structPerceptron	*prev;
	struct structPerceptron	*next;
	struct structPerceptron	*nextHead;

} structPerceptron;	


void				AddNew_Perceptron(structPerceptron **head, structPerceptron *newPerceptron);

void				AddNewV2_Perceptron(structPerceptron **perceptronHead, structPerceptron *perceptronNew);
void				Delete_Perceptron(structPerceptron **perceptronHead);
void				DeleteV2_Perceptron(structPerceptron **head);


void				AddToLayer_Perceptron(structPerceptron **perceptronHead, structPerceptron *perceptronNew, int nIndex);
void				AddToLayerV2_Perceptron(structPerceptron **head, structPerceptron *newPerceptron);
int					CalculateThreshold_Perceptron(structPerceptron *perceptronClassifierHead, float fThreshold, float *fRatio, int bDisplayMode, FILE *pFile);


typedef struct structInputPerceptron
{
	int		nID;
	int		nClassID;
	float	fOutput;
	float	fUpperThreshold;
	float	fLowerThreshold;
	float	fThreshold;
	int		nSynapseCount;

	structInputSynapse				*synapseHead;
	struct structInputPerceptron	*prev;
	struct structInputPerceptron	*next;

} structInputPerceptron;
