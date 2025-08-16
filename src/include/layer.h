#pragma once

typedef struct structLayer
{
	int					nID;
	int					nLayerType;
	int					nKernelCount;
	int					nInputRowCount;
	int					nInputColumnCount;
	int					nKernelRowCount;
	int					nKernelColumnCount;
	int					nInputMapCount;
	int					nOutputMapCount;
	float				*fWeightArray;
	fxpt				*fxptWeightArray;

	float				**fInputArray;
	float				*fOutputArray;
//	float				*fDifferentialArray;
//	float				*fLearningRateArray;
	int					nWeightCount;
	int					nOutputArraySize;
	int					nOffset;
	int					nPerceptronCount;
	int					nConnectionCount;
	int					nAdditionCount;
	float				fGamma;
	float				fLambda;

	int					nIndex;
	int					nPaddingMode;
	int					nOutputRowCount;
	int					nOutputColumnCount;
	int					nStrideRow;
	int					nStrideColumn;
	int					nStrideRowOffset;
	int					nStrideColumnOffset;
	char				sLayerName[32];
	int					nInitializeWeights;
	int					nWeightIndex;
	int					nActivationMode;
	fxpt				*fxptActivationArray;
	int					nSegmentCount;
	int					nExecutionMode;
	int					nNumberFormat;

	float				fMaxSum;
	float				fMinSum;
	float				fMaxWeight;
	float				fMinWeight;
	float				fLearningRate;

	PHYS_ADDR			physAddrWeightArray;

	int			nAdditions;
	int			nMultiplications;


	structPerceptron	*perceptronHead;

	struct structLayer	*next;
	struct structLayer	*prev;

} structLayer;

void			AddNew_Layer(structLayer **head, structLayer *newLayer);
structLayer		*Delete_Layer(structLayer **head, int nID);
void			DeleteAll_Layer(structLayer **head);
structLayer		*Create_Layer(structLayer **layerHead, int nLayerType, int nInputRowCount, int nInputColumnCount, int nPerceptronCount);
void			Free_Layer(structPerceptron *perceptronHead);


void			AddLayer(structLayer **head, structLayer *newLayer);



typedef struct structInputLayer
{
	int						nID;
	int						nPerceptronCount;
	structInputPerceptron	*perceptronHead;

	struct structInputLayer	*next;
	struct structInputLayer	*prev;

} structInputLayer;
