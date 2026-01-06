WASM_FLAGS = --target=wasm32 \
	-O3 \
	-flto \
	-nostdlib \
	-Wl,--no-entry \
	 -Wl,--export-all \
	-Wl,--lto-O3 \
	-DNDEBUG

monoscule.wasm: monoscule.c ../../monoscule.h
	$(CC) $(WASM_FLAGS) monoscule.c  -o monoscule.wasm
