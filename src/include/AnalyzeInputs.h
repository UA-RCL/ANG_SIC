
void					AnalyzeInputs(float *fInputArray, structInput *inputTrainData, structInput *inputValidateData);
structInputPerceptron	*NewInputPerceptron(structInputLayer *layer, int *nPerceptronID, int *nSynapseID, int nSynapseCount, int nClassID);
structInputLayer		*NewInputLayer(structCLN *cln);
void					GetUntrainedClassMember(structInput *inputData, int nClassID, int *nIndex);
int						AssignInputConnections(structInputPerceptron *perceptron, float *fInputArray, int nArrayCount);
void					LoadInputArray(structInput *inputData, int nIndex, float *fInputArray);
void					AddInputPerceptron(structInputPerceptron **head, structInputPerceptron *newPerceptron);
int						TuneInputPerceptron(structInputPerceptron *perceptron, structInput *inputData, float *fInputArray);
float					ForwardPropagateInputPerceptron(structInputPerceptron *perceptron);
int						MarkCorrectPredictions(structInputPerceptron *perceptron, structInput *inputData, float *fInputArray);
void					TestInputLayer(structInputLayer *layer, structInput *inputData, float *fInputArray);
void					DumpLayerWeights(structInputLayer *layer);
int						AssignInputError(structInput *inputData, int nTarget);
void					ResetData(structInput *inputData);


void					AddInputSynapse(structInputSynapse **head, structInputSynapse *newSynapse);
int						GroupInputError(structInput *inputData, int nTarget, int nOutputCount);
void					GetUntrainedClassMemberFromCluster(structInput *inputData, int nClassID, int *nIndex, int nCluster);
void					CreateTrainingDataSubGroup(structInput *inputData, structInput **inputSubGroup, int nTargetCount, int nClusterCount, int nClassCount);
int						MarkAllCorrectPredictions(structInputPerceptron *perceptronHead, structInput *inputData, float *fInputArray, int *nRight, int *nWrong);
