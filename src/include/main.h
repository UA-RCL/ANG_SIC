
//#define	_WINDOWS
// #define	_LINUX
//#define	_VXWORKS





#ifdef _WINDOWS
	#include <windows.h>

	#define M_PI			3.141592F
	#define	STRICMP			_stricmp
	#define	STRREV			_strrev
	#define	STRIDUP			_strdup
#endif

#ifdef _LINUX
	#define	STRICMP			strcasecmp
	#define	STRREV			ReverseString
	#define	STRIDUP			strdup
#endif



#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <stdlib.h>
#include <time.h>  
#include <stdint.h>
#include <string.h>


#define MNIST		100
#define CIFAR		200
#define IMAGENET	960



typedef		int32_t	fxpt;
typedef int	PHYS_ADDR;

#define FXPT32_BITS				32
#define FXPT32_WBITS			6
#define FXPT32_FBITS			(FXPT32_BITS - FXPT32_WBITS)

#define FloatToFXPT32(R)		(fxpt)(R * FXPT32_ONE + (R >= 0 ? 0.5 : -0.5))
#define FXPT32_ONE				(fxpt)((fxpt)1 << FXPT32_FBITS)
#define FXPT32_Mult(A,B)		(fxpt)(((int64_t)A * (int64_t)B) >> FXPT32_FBITS)
#define FXPT32_Divd(A,B)		(fxpt)(((int64_t)A << FXPT32_FBITS) / (int64_t)B)

#define BILLION  1000000000L

/* Keras Activation Functions */
#define LINEAR				0
#define	TANH				1
#define	MTANH				2


typedef struct structArchitecture
{
	int		nID;
	int		nLayerType;
	float	fAccuracy;
	float	fPercentResponse;
	int		nKernelCount;
	int		nRowKernelSize;
	int		nColumnKernelSize;
	int		nStrideRow;
	int		nStrideColumn;
	int		nOutputRows;
	int		nOutputColumns;
	int		nWeightCount;
	int		bKeep;

	int		nInputRowCount;
	int		nInputColumnCount;
	float	fLearningRate;
	int		nPaddingMode;
	int		nActivationMode;
	int		nNumberFormat;

	struct structArchitecture	*next;
	struct structArchitecture	*prev;


} structArchitecture;

typedef struct structConfigParameters
{
	char	sParameter[256];
	char	sValue[256];
} structConfigParameters;


#define MARK_CORRECT_CLASSIFICATION					100
#define INFER_CORRECT_ONLY							200
#define BREAK_ON_BAD_CLASSIFICATION					300
#define SYNAPSE_STATISTICS_CORRECT_CLASSIFICATION	400
#define SYNAPSE_STATISTICS							500





#include "Activation.h"
#include "Bicubic.h"
#include "FixedPoint.h"
#include "Classes.h"
#include "InputData.h"

#include "MAC.h"
#include "Synapse.h"
#include "Perceptron.h"
#include "layer.h"
#include "ClassLevelNetworks.h"
#include "Network.h"

#include "InputFileISAR.h"
#include "JenkFish.h"
#include "GaborFilter.h"
#include "Train.h"
#include "NeuroGenesis.h"
#include "Inference.h"
#include "Functions.h"
#include "ConstructNetworks.h"
#include "Convolution.h"
#include "InputFileIR.h"
#include "InputFileCIFAR.h"
#include "InputFileImagenette.h"
#include "Config.h"
#include "LRI.h"
#include "Prune.h"
#include "AnalyzeInputs.h"

#define	TRAINING				100
#define	VERIFY					200
#define	TESTING					300

#define INPUT_LAYER					1
#define FULLY_CONNECTED_LAYER		2
#define SINGLE_CONV_LAYER			3
#define CONV_2D_LAYER				3
#define MULTIPLE_CONV_LAYER			4
#define CONV_3D_LAYER				4
#define CLASSIFIER_LAYER			5
#define SPARSELY_CONNECTED_LAYER	6
#define COMBINING_CLASSIFIER_LAYER	7
#define MAX_POOLING_LAYER			8
#define AVERAGE_POOLING_LAYER		9


#define DENSE_LAYER					2
#define SPARCE_LAYER				6
#define MAX_POOL_LAYER				8
#define COMBINING_DENSE_LAYER		10
#define DROPOUT_LAYER				11
#define FLATTEN_LAYER				12
#define CROPPING_2D_LAYER			13
#define ZERO_PADDING_2D_LAYER		14
#define BATCH_NORMALIZATION_LAYER	15
#define ACTIVATION_LAYER			16
#define AVERAGE_LAYER				17
#define MAXIMUM_LAYER				18
#define MINIMUM_LAYER				19
#define ADD_LAYER					20
#define SUBTRACT_LAYER				21
#define MULTIPLY_LAYER				22
#define CENTER_CROP_LAYER			23
#define BORDER_NORMALIZATION_LAYER	24
#define CONV_2D_NORMALIZED_LAYER	25
#define UPSAMPLE_LAYER				26
#define MASK_NORMALIZATION_LAYER	27
#define APPLY_MASK_LAYER			28
#define LAMBDA_CARRY_GATE_LAYER		29
#define CONCATENATE_LAYER			30
#define SUMMATION_LAYER				31






#define RANDOMIZE				100
#define ORDERED					200
#define SORT					300
#define SHUFFLE					400
#define GABOR					500
#define SAME_WEIGHTS			600

#define DEG_2_RAD				M_PI / 180.0F
#define TWO_THIRDS				2.0f / 3.0F
#define	SIG_PARM				1.71589F
#define UNIFORM_PLUS_MINUS_ONE	( (float)(2.0F * rand())/RAND_MAX - 1.0F)
#define UNIFORM_ZERO_THRU_ONE	( (float)(rand())/(RAND_MAX + 1UL)) 
#define SIGMOID(x)				(float)((1.7159F*tanh(TWO_THIRDS*x)))
#define DSIGMOID(S)				(float)((TWO_THIRDS/1.7159F*(1.7159F+(S))*(1.7159F-(S))))  // derivative of the sigmoid as a function of the sigmoid's output
#define	SIG_PARM_2				SIG_PARM/2.0F
#define	SQUARED(x)				(float)(x * x)
#define	MULTIPLIER				0.05f

#define	LEARNING_RATE_MULTIPLIER	0.0001F
#define	LEARNING_RATE_LOWER_BOUND	0.00F


#define	MAX_WEIGHT	65504
#define	MIN_WEIGHT	-65504

#define SHOW_MATRIX				1
#define HIDE_MATRIX				0

#define SHOW_DATA				1
#define HIDE_DATA				0


#define ADJUST					1
#define DO_NOT_ADJUST			0

#define NETWORK_TO_MEMORY		1
#define MEMORY_TO_NETWORK		0

#define CUR_CLN					100
#define ALL_CLN					200

#define COMPLETE_NETWORK		100
#define CLASS_NETWORK			200
#define COMBINE_NETWORK			300

#define MARK					100
#define DO_NOT_MARK				200

#define THRESHOLD				100
#define NO_THRESHOLD			200

#define CALCULATE_SD					100
#define CALCULATE_AVERAGE				200
#define CALCULATE_AVERAGE_NO_SATURATE	300
#define CALCULATE_LEARNING_RATE			400
#define FIXED_LEARNING_RATE				500



#define BUILD_COMPLETE_NETWORK			0
#define REBUILD_COMPLETE_NETWORK		1
#define REBUILD_CLASS_LEVEL_NETWORK		2
#define BUILD_CLASS_LEVEL_NETWORK		3
#define NEUROGENESIS					4
#define REBUILD_NEUROGENESIS			5
#define ANALYZE_INPUT					6

#define COMPRESS						100
#define FULL_RANGE						200


#define PRUNE_AFTER						100
#define PRUNE_EACH_CYCLE				200
#define PRUNE_INTERVAL_CYCLE			300
#define PRUNE_AFTER_ZERO				400

#define EXECUTE_TEST_INFERENCE			100
#define DO_NOT_EXECUTE_TEST_INFERENCE	200

#define	PAUSE					1
#define	CONTINUE				2

#define FLOAT_POINT				100
#define FIXED_POINT				200
#define FLOAT_FIXED				300

#define	HARDMAX				100
#define	SOFTMAX				200

#define	VALID				100
#define	SAME				200
#define	BACK_UP				300

#define	TANH				1
#define	MTANH				2

#define	EXECUTION_MODE_CPU		0
#define	EXECUTION_MODE_FPGA		1

#define	WRITE_DNA_FILE			1
#define	WRITE_DNX_FILE			2

structArchitecture	*AllocateArchitecture(int *nID);
void				AddArchitecture(structArchitecture **head, structArchitecture *newArchitecture);
structArchitecture	*DeleteArchitecture(structArchitecture **head, int nID);
int					DeleteAllArchitecture(structArchitecture **head);
float				SIC_Individual(structNetwork *networkMain, structInput *inputData, structInput *inputTestingData, structInput *inputVerifyData, structInput *inputTrainingData);
void				ConvertToDNX(structNetwork *networkMain, float *fInputArray);
void				SaveNetwork(structNetwork* network, char* sFilePath);
void				SIC_V3(structNetwork* network, structInput* inputTrain, structInput* inputTest, float* fAccuracy, int nMaxCluster);
void				ModifyArchitecture(structCLN* cln);
void				CopyPerceptronWeights(structPerceptron* perceptron, int nMode);
void				DeleteZeroWeights(structCLN* cln);
void				SIC_V4(structNetwork* network, structInput* inputTrain, structInput* inputTest, int nMaxCluster, float fTargetAccuracy);
