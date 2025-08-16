

typedef int32_t	fx32;
typedef int16_t	fx16;
typedef int8_t	fx8;

typedef union FIXED6_26tag 
{
    fx32 full;
    struct part6_26tag 
	{
		fx32 fraction: 26;
		fx32 integer: 6;
    } part;

} FIXED6_26;



#define FX32_BITS		32
#define FX32_WBITS		7
#define FX32_MBITS		FX32_WBITS - 2
#define FX32_FBITS		(FX32_BITS - FX32_WBITS)
#define FX32_ONE		(fx32)((fx32)1 << FX32_FBITS)
#define MULT_FX32(A,B)	(fx32)(((int64_t)A * (int64_t)B) / ( 1 << FX32_FBITS))
#define DIVD_FX32(A,B)	(fx32)((((int64_t)A * (1 << FX32_FBITS)) / B))
#define FloatToFx32(R)	(fx32)(R * FX32_ONE + (R >= 0 ? 0.5 : -0.5))

#define SIG_MIN			FloatToFx32(-SIG_PARM)
#define SIG_MAX			FloatToFx32(SIG_PARM)


#define SIG_A			FloatToFx32(-4.0f)
#define SIG_B			FloatToFx32(-1.6f)
#define SIG_C			FloatToFx32(-0.2f)
#define SIG_D			FloatToFx32(0.2f)
#define SIG_E			FloatToFx32(1.6f)
#define SIG_F			FloatToFx32(4.0f)

#define SIG_A_R			FloatToFx32(-1.60f)
#define SIG_B_R			FloatToFx32(-1.35f)
#define SIG_C_R			FloatToFx32(-0.23f)
#define SIG_D_R			FloatToFx32(0.23f)
#define SIG_E_R			FloatToFx32(1.35f)
#define SIG_F_R			FloatToFx32(1.60f)



float					Fx32ToFloat(fx32 nValue);
fx32					Sigmoid(fx32 fx32Value);
fx32					DSigmoid(fx32 fx32Value);

#define FX32_MAX		FloatToFx32((powf(2.0f, (float)FX32_MBITS) - 1.0f))
#define FX32_MIN		FloatToFx32((-powf(2.0f, (float)FX32_MBITS) + 1.0f))


#define MULTX_Y(A,B) (int64_t)(A.full*B.full+     2^(Y-1))>>Y 
#define DIVX_Y(A,B)  (int64_t)((A.full<<Y+1)/     B.full)+1)/2
#define FIXEDX_YCONST(A,B) (fx32)((A<<26) + ((B + 0.000000007450580596923828125)*67108864))



#define FX16_BITS		16
#define FX16_WBITS		3
#define FX16_FBITS		(FX16_BITS - FX16_WBITS)
#define FX16_ONE		(fx16)((fx16)1 << FX16_FBITS)
#define MULT_FX16(A,B)	(fx16)(((int32_t)A * (int32_t)B) >> FX16_FBITS)
#define FloatToFx16(R)	(fx16)(R * FX16_ONE + (R >= 0 ? 0.5 : -0.5))
float					Fx16ToFloat(fx16 nValue);

#define FX8_BITS		8
#define FX8_WBITS		1
#define FX8_FBITS		(FX8_BITS - FX8_WBITS)
#define FX8_ONE			(fx8)((fx8)1 << FX8_FBITS)
#define MULT_FX8(A,B)	(fx8)(((int16_t)A * (int16_t)B) >> FX8_FBITS)
#define FloatToFx8(R)	(fx8)(R * FX8_ONE + (R >= 0 ? 0.5 : -0.5))
float					Fx8ToFloat(fx8 nValue);


