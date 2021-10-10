//clang src/neohookean.c --target=wasm32-unknown-unknown-wasm --optimize=3 -nostdlib -Wl,--export-all -Wl,--no-entry -Wl,--allow-undefined --output html/neohookean.wasm

__attribute__((used)) 
void vecCopy(float* a, int anr, float* b, int bnr) {
    anr *= 3; bnr *= 3;
    a[anr++] = b[bnr++]; 
    a[anr++] = b[bnr++]; 
    a[anr]   = b[bnr];
}
