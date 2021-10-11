//clang src/neohookean.c --target=wasm32-unknown-unknown-wasm --optimize=3 -nostdlib -Wl,--export-all -Wl,--no-entry -Wl,--allow-undefined --output html/neohookean.wasm

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
