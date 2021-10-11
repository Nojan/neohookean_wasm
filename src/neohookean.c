//emcc src/neohookean.c -s TOTAL_MEMORY=32MB -O3 --no-entry -o html/neohookean.wasm
#include <math.h>

__attribute__((used)) 
void vecSetZero(float* a, int anr) {
    anr *= 3;
    a[anr++] = 0.f;
    a[anr++] = 0.f;
    a[anr]   = 0.f;
}

__attribute__((used)) 
void vecCopy(float* a, int anr, float* b, int bnr) {
    anr *= 3; bnr *= 3;
    a[anr++] = b[bnr++]; 
    a[anr++] = b[bnr++]; 
    a[anr]   = b[bnr];
}

__attribute__((used)) 
void vecAdd(float* a, int anr, float* b, int bnr, double s) {
    float scale = s;
    anr *= 3; bnr *= 3;
    a[anr++] += b[bnr++] * scale; 
    a[anr++] += b[bnr++] * scale; 
    a[anr]   += b[bnr] * scale;
}

__attribute__((used)) 
void vecSetDiff(float* dst, int dnr, float* a, int anr, float* b, int bnr, double s) {
    float scale = s;
    dnr *= 3; anr *= 3; bnr *= 3;
    dst[dnr++] = (a[anr++] - b[bnr++]) * scale;
    dst[dnr++] = (a[anr++] - b[bnr++]) * scale;
    dst[dnr]   = (a[anr] - b[bnr]) * scale;
}

__attribute__((used)) 
double vecLengthSquared(float* a, int anr) {
    anr *= 3;
    double a0 = a[anr], a1 = a[anr + 1], a2 = a[anr + 2];
    return a0 * a0 + a1 * a1 + a2 * a2;
}

__attribute__((used)) 
void vecSetCross(float* a,int anr, float* b, int bnr, float* c, int cnr) {
    anr *= 3; bnr *= 3; cnr *= 3;
    a[anr++] = b[bnr + 1] * c[cnr + 2] - b[bnr + 2] * c[cnr + 1];
    a[anr++] = b[bnr + 2] * c[cnr + 0] - b[bnr + 0] * c[cnr + 2];
    a[anr]   = b[bnr + 0] * c[cnr + 1] - b[bnr + 1] * c[cnr + 0];
}

__attribute__((used)) 
double matIJ(float* A, int anr, int row, int col) {
    return A[9*anr + 3 * col + row];
}

__attribute__((used)) 
void matSetVecProduct(float* dst, int dnr, float* A, int anr, float* b, int bnr) {
    bnr *= 3; anr *= 3;
    const double b0 = b[bnr++];
    const double b1 = b[bnr++];
    const double b2 = b[bnr];
    vecSetZero(dst,dnr);
    vecAdd(dst,dnr, A,anr++, b0);
    vecAdd(dst,dnr, A,anr++, b1);
    vecAdd(dst,dnr, A,anr,   b2);
}

__attribute__((used)) 
void matSetMatProduct(float* Dst, int dnr, float* A, int anr, float* B, int bnr) {
    dnr *= 3; bnr *= 3;
    matSetVecProduct(Dst,dnr++, A,anr, B,bnr++);
    matSetVecProduct(Dst,dnr++, A,anr, B,bnr++);
    matSetVecProduct(Dst,dnr++, A,anr, B,bnr++);
}

__attribute__((used)) 
double matGetDeterminant(float* A, int anr) {
    anr *= 9;
    double a11 = A[anr + 0], a12 = A[anr + 3], a13 = A[anr + 6];
    double a21 = A[anr + 1], a22 = A[anr + 4], a23 = A[anr + 7];
    double a31 = A[anr + 2], a32 = A[anr + 5], a33 = A[anr + 8];
    return a11*a22*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31 - a12*a21*a33 - a11*a23*a32;
}

__attribute__((used)) 
void applyToElem(int elemNr, double C, double compliance, double dt, float* grads, float* invMass, float* invRestVolume, int* tetIds, float* pos) 
{
    if (C == 0.0)
        return;
    float* g = grads;

    vecSetZero(g,0);
    vecAdd(g,0, g,1, -1.0);
    vecAdd(g,0, g,2, -1.0);
    vecAdd(g,0, g,3, -1.0);

    double w = 0.0;
    for (int i = 0; i < 4; i++) {
        int id = tetIds[4 * elemNr + i];
        w += vecLengthSquared(g,i) * invMass[id];
    }

    if (w == 0.0) 
        return;
    double alpha = compliance / dt / dt * invRestVolume[elemNr];
    double dlambda = -C / (w + alpha);

    for (int i = 0; i < 4; i++) {
        int id = tetIds[4 * elemNr + i];
        vecAdd(pos,id, g,i, dlambda * invMass[id]);
    }
}

__attribute__((used)) 
double solveElem(int elemNr, double dt, double devCompliance, double volCompliance, float* grads, float* P, float* F, float* dF, float* invMass, float* invRestVolume, float* invRestPose, int* tetIds, float* pos, double volError) 
{
    double C = 0.0;
    float* g = grads;
    float* ir = invRestPose;

    // tr(F) = 3

    int id0 = tetIds[4 * elemNr];
    int id1 = tetIds[4 * elemNr + 1];
    int id2 = tetIds[4 * elemNr + 2];
    int id3 = tetIds[4 * elemNr + 3];

    vecSetDiff(P,0, pos,id1, pos,id0, 1.0);
    vecSetDiff(P,1, pos,id2, pos,id0, 1.0);
    vecSetDiff(P,2, pos,id3, pos,id0, 1.0);

    matSetMatProduct(F,0, P,0, invRestPose,elemNr);

    double r_s = sqrt(vecLengthSquared(F,0) + vecLengthSquared(F,1) + vecLengthSquared(F,2));
    double r_s_inv = 1.0 / r_s;

    vecSetZero(g,1);
    vecAdd(g,1, F,0, r_s_inv * matIJ(ir,elemNr, 0,0));
    vecAdd(g,1, F,1, r_s_inv * matIJ(ir,elemNr, 0,1));
    vecAdd(g,1, F,2, r_s_inv * matIJ(ir,elemNr, 0,2));

    vecSetZero(g,2);
    vecAdd(g,2, F,0, r_s_inv * matIJ(ir,elemNr, 1,0));
    vecAdd(g,2, F,1, r_s_inv * matIJ(ir,elemNr, 1,1));
    vecAdd(g,2, F,2, r_s_inv * matIJ(ir,elemNr, 1,2));

    vecSetZero(g,3);
    vecAdd(g,3, F,0, r_s_inv * matIJ(ir,elemNr, 2,0));
    vecAdd(g,3, F,1, r_s_inv * matIJ(ir,elemNr, 2,1));
    vecAdd(g,3, F,2, r_s_inv * matIJ(ir,elemNr, 2,2));

    C = r_s; 


    applyToElem(elemNr, C, devCompliance, dt, grads, invMass, invRestVolume, tetIds, pos);
    
    // det F = 1

    vecSetDiff(P,0, pos,id1, pos,id0, 1.0);
    vecSetDiff(P,1, pos,id2, pos,id0, 1.0);
    vecSetDiff(P,2, pos,id3, pos,id0, 1.0);

    matSetMatProduct(F,0, P,0, invRestPose,elemNr);

    vecSetCross(dF,0, F,1, F,2);
    vecSetCross(dF,1, F,2, F,0);
    vecSetCross(dF,2, F,0, F,1);

    vecSetZero(g,1);
    vecAdd(g,1, dF,0, matIJ(ir,elemNr, 0,0));
    vecAdd(g,1, dF,1, matIJ(ir,elemNr, 0,1));
    vecAdd(g,1, dF,2, matIJ(ir,elemNr, 0,2));

    vecSetZero(g,2);
    vecAdd(g,2, dF,0, matIJ(ir,elemNr, 1,0));
    vecAdd(g,2, dF,1, matIJ(ir,elemNr, 1,1));
    vecAdd(g,2, dF,2, matIJ(ir,elemNr, 1,2));

    vecSetZero(g,3);
    vecAdd(g,3, dF,0, matIJ(ir,elemNr, 2,0));
    vecAdd(g,3, dF,1, matIJ(ir,elemNr, 2,1));
    vecAdd(g,3, dF,2, matIJ(ir,elemNr, 2,2));

    const double lambda = 1.0/volCompliance;
	const double mu = 1.0/devCompliance;

    const double vol = matGetDeterminant(F,0);
    C = vol - 1.0 - mu/lambda;

    volError += vol - 1.0;
    
    applyToElem(elemNr, C, volCompliance, dt, grads, invMass, invRestVolume, tetIds, pos);

    return volError;
}